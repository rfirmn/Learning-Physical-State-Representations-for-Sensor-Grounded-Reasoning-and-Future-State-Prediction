"""Small, shared reproducibility boundary for the M4Human v3 pipeline.

Source LMDBs are never written here. Checkpoints are tensor/plain-container
payloads loaded with weights_only=True; no legacy pickle fallback is permitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

CONTRACT_VERSION = "m4human_kinetok_v3"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=_json_default).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path, newline="lf"):
    """UTF-8 completion text hash; newline normalization only, not JSON rewriting."""
    digest = hashlib.sha256()
    with open(path, encoding="utf-8", newline="") as handle:
        for line in handle:
            normalized = line.replace("\r\n", "\n")
            digest.update((normalized.replace("\n", "\r\n") if newline == "crlf" else normalized).encode("utf-8"))
    return digest.hexdigest()


def completion_text_matches(path, expected, algorithm=None):
    if algorithm == "utf8_lf_sha256":
        return expected == text_sha256(path)
    if algorithm is not None:
        return False
    # Old completion markers hashed platform text bytes. Accept only the same
    # UTF-8 content with LF/CRLF conversion; lineage always hashes exact bytes.
    return expected in (file_sha256(path), text_sha256(path), text_sha256(path, "crlf"))


def state_dict_hash(module_or_state):
    state = module_or_state.state_dict() if isinstance(module_or_state, torch.nn.Module) else module_or_state
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {key} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_write(path, writer, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publication is atomic and cannot overwrite another writer.
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, payload, overwrite=False):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False, default=_json_default) + "\n"
    def write(temporary):
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    _atomic_write(path, write, overwrite)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield value


def require_lineage(metadata, expected):
    if metadata.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Foreign or missing M4Human contract")
    lineage = metadata.get("lineage", metadata)
    for key, value in expected.items():
        if value is None or value == "" or lineage.get(key) != value:
            raise ValueError(f"Missing/foreign lineage: {key}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def adamw_parameters(module, weight_decay=0.01):
    parameters = [p for p in module.parameters() if p.requires_grad]
    return [{"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0}]


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state((state["numpy"][0], np.asarray(state["numpy"][1], dtype=np.uint32),
                         *state["numpy"][2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume CUDA RNG on a CPU-only runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, metadata, **extra):
    metadata = dict(metadata)
    metadata.setdefault("contract_version", CONTRACT_VERSION)
    require_lineage(metadata, {})
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {"state_dict": state, "metadata": metadata, "state_hash": state_dict_hash(state), **extra}
    # best/last are mutable only within the caller's unique run directory.
    _atomic_write(path, lambda temporary: torch.save(payload, temporary), overwrite=True)
    return file_sha256(path)


def load_checkpoint(path, model, expected_lineage=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"state_dict", "metadata", "state_hash"} <= payload.keys():
        raise ValueError("Not a M4Human checkpoint")
    require_lineage(payload["metadata"], expected_lineage or {})
    if state_dict_hash(payload["state_dict"]) != payload["state_hash"]:
        raise ValueError("Checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


def environment_snapshot():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    untracked = git("ls-files", "--others", "--exclude-standard") or ""
    code_files = [Path(name) for name in untracked.splitlines()
                  if name.startswith("eksperimen_model/") and Path(name).suffix in {".py", ".yaml", ".json"}]
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "git_diff": git("diff", "--no-ext-diff"),
            "untracked_code_sha256": {str(path): file_sha256(path) for path in code_files if path.is_file()}}


def process_tree_resources():
    """Sample system headroom and process-tree accounting, never physical RSS usage."""
    import psutil
    process = psutil.Process()
    processes = [process] + process.children(recursive=True)
    rows, partial = [], False
    for member in processes:
        try:
            rss = member.memory_info().rss
            try:
                uss = getattr(member.memory_full_info(), "uss", None)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                uss = None
            rows.append({"pid": member.pid, "rss_bytes": rss, "uss_bytes": uss})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            partial = True
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
            "process_scope": "parent_and_recursive_descendants",
            "processes": rows, "process_sample_partial": partial,
            "process_tree_rss_sum_bytes": sum(row["rss_bytes"] for row in rows),
            "rss_sum_semantics": "sum_of_process_RSS; shared_pages_may_be_counted_multiple_times; not_physical_RAM_usage",
            "process_tree_uss_sum_bytes": sum(row["uss_bytes"] for row in rows)
                if rows and all(row["uss_bytes"] is not None for row in rows) and not partial else None,
            "swap_used_bytes": swap.used,
            "swap_in_bytes": None if sys.platform == "win32" else getattr(swap, "sin", None),
            "swap_out_bytes": None if sys.platform == "win32" else getattr(swap, "sout", None)}


def resource_snapshot():
    current, peak = None, None
    try:
        import psutil
        info = psutil.Process().memory_info()
        current = info.rss
        peak = getattr(info, "peak_wset", None)
    except ImportError:
        pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        pass
    devices = {str(i): {"peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)}
               for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else {}
    active = str(torch.cuda.current_device()) if devices else None
    return {"process_current_rss_bytes": current, "process_peak_rss_bytes": peak,
            "cuda_device_peaks": devices,
            "cuda_peak_allocated_bytes": devices[active]["peak_allocated_bytes"] if active else None,
            "cuda_peak_reserved_bytes": devices[active]["peak_reserved_bytes"] if active else None}


def plot_run_history(root):
    history = root / "history.jsonl"
    if not history.exists():
        return False
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    series = {}
    for index, row in enumerate(iter_jsonl(history)):
        terms = row.get("terms", row.get("loss", {}))
        if isinstance(terms, dict):
            for name, value in terms.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    series.setdefault(name, []).append((index, value))
        for name in ("answer_only_ce", "loss"):
            value = row.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                series.setdefault(name, []).append((index,value))
    if not series:
        return False
    fig, axis = plt.subplots(figsize=(8,4))
    for name, points in series.items():
        axis.plot(*zip(*points), label=name)
    axis.set(xlabel="Logged training event", ylabel="Objective value (term-specific units)", title="Training objectives; validation metrics are reported separately")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "loss_curve.png", dpi=150)
    plt.close(fig)
    return True


class RunLogger:
    """Append-only events, explicit completion, no scientific success by implication."""
    def __init__(self, output_dir, config, stage):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.stage, self.config = stage, config
        self.started = time.monotonic()
        self.closed = False
        self.run_id = self.output_dir.name
        self.data_kind = config.get("data_kind", config.get("data", {}).get("data_kind", "m4human"))
        with open(self.output_dir / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        atomic_json(self.output_dir / "lineage.json", {
            "contract_version": CONTRACT_VERSION, "stage": stage, "run_id": self.run_id,
            "data_kind": self.data_kind, "config_hash": canonical_hash(config),
            "lineage": config.get("lineage", {}), "environment": environment_snapshot()})
        self.log("start")

    def log(self, event, **values):
        row = {"event": event, "stage": self.stage, "run_id": self.run_id,
               "data_kind": self.data_kind, "utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_s": time.monotonic() - self.started, **values}
        encoded = json.dumps(row, default=_json_default, allow_nan=False) + "\n"
        names = ["events.jsonl"]
        if event in {"epoch", "validation", "history", "train_epoch", "update"}:
            names.append("history.jsonl")
        for name in names:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / name, "a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
        return row

    def finish(self, summary, status="complete"):
        if self.closed:
            raise RuntimeError("Run already finalized")
        if status not in {"complete", "failed", "incomplete"}:
            raise ValueError("Unknown run status")
        result = {"contract_version": CONTRACT_VERSION, "stage": self.stage,
                  "run_id": self.run_id, "data_kind": self.data_kind, "status": status,
                  "elapsed_s": time.monotonic() - self.started,
                  "resources": resource_snapshot(), **summary}
        atomic_json(self.output_dir / "metrics.json", result)
        self.log(status, summary=summary)
        try:
            result["loss_curve_created"] = plot_run_history(self.output_dir)
        except ImportError:
            result["loss_curve_created"] = False
            result["plot_status"] = "matplotlib_missing"
        artifact_issues = validate_run_artifacts(self.output_dir, require_completion=False)
        result["analysis_ready"] = not artifact_issues
        result["artifact_issues"] = artifact_issues
        atomic_json(self.output_dir / "metrics.json", result, overwrite=True)
        with open(self.output_dir / "report.md", "w", encoding="utf-8") as handle:
            handle.write(f"# M4Human {self.stage}: {self.run_id}\n\nStatus: {status}. Data: {self.data_kind}.\n\n")
            handle.write("Synthetic checks are implementation evidence only.\n\n" if self.data_kind == "synthetic" else "Real-data gates and scientific claims require separate validation.\n\n")
            handle.write("```json\n" + json.dumps(result, default=_json_default, indent=2, allow_nan=False) + "\n```\n")
        atomic_json(self.output_dir / "completion.json", {
            "status": status, "analysis_ready": not artifact_issues,
            "text_hash_algorithm": "utf8_lf_sha256",
            "metrics_hash": text_sha256(self.output_dir / "metrics.json"),
            "history_hash": text_sha256(self.output_dir / "history.jsonl") if (self.output_dir / "history.jsonl").exists() else None})
        self.closed = True
        if self.data_kind == "m4human":
            report_root = next((p for p in self.output_dir.parents if p.name == "report_training"), None)
            if report_root is not None:
                relative = (self.output_dir / "report.md").relative_to(report_root).as_posix()
                index = report_root / "INDEX.md"
                previous = index.read_text(encoding="utf-8") if index.exists() else "# Training reports\n"
                if relative not in previous:
                    heading = "\n\n## M4Human v3 runs\n\n| Run | Stage | Status | Analysis ready | Report |\n|---|---|---|---|---|\n"
                    with index.open("a", encoding="utf-8") as handle:
                        if "## M4Human v3 runs" not in previous:
                            handle.write(heading)
                        handle.write(f"| {self.run_id} | {self.stage} | {status} | {not artifact_issues} | [Report]({relative}) |\n")
        return result

    def fail(self, exception):
        self.log("error", reason=str(exception), traceback="".join(traceback.format_exception(exception)))
        return self.finish({"reason": str(exception)}, status="failed")


def validate_run_artifacts(output_dir, require_completion=True):
    """Check artifact integrity and recompute recorded physical sums or QA scores."""
    root = Path(output_dir)
    required = ["config_snapshot.yaml", "lineage.json", "history.jsonl", "events.jsonl",
                "predictions_val.jsonl", "diagnostic_samples.npz", "diagnostic_samples.json",
                "metrics.json", "best.pt", "last.pt"]
    issues = [f"missing:{name}" for name in required if not (root / name).is_file()]
    if (root / "lineage.json").exists():
        lineage = read_json(root / "lineage.json")
        if lineage.get("contract_version") != CONTRACT_VERSION:
            issues.append("foreign_contract")
        if not lineage.get("config_hash"):
            issues.append("missing_config_hash")
        elif (root / "config_snapshot.yaml").exists():
            with open(root / "config_snapshot.yaml", encoding="utf-8") as handle:
                if canonical_hash(yaml.safe_load(handle)) != lineage["config_hash"]:
                    issues.append("config_hash_mismatch")
    if (root / "predictions_val.jsonl").exists():
        ids, row_count, missing_groups, invalid_ids = set(), 0, False, False
        for row in iter_jsonl(root / "predictions_val.jsonl"):
            row_count += 1
            identity = row.get("qa_id", row.get("window_id", row.get("frame_uid")))
            if identity is None or identity in ids:
                invalid_ids = True
            ids.add(identity)
            missing_groups |= not all(key in row for key in ("subject_id", "recording_id"))
        if not row_count or invalid_ids:
            issues.append("prediction_ids_missing_or_duplicate")
        if missing_groups:
            issues.append("prediction_group_ids_missing")
    for name in ("best.pt", "last.pt"):
        if (root / name).exists():
            try:
                checkpoint = torch.load(root / name, map_location="cpu", weights_only=True)
                require_lineage(checkpoint["metadata"], {})
                if state_dict_hash(checkpoint["state_dict"]) != checkpoint["state_hash"]:
                    issues.append(f"checkpoint_state_hash_mismatch:{name}")
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                issues.append(f"checkpoint_invalid:{name}:{type(exc).__name__}")
    if (root / "diagnostic_samples.json").exists() and (root / "best.pt").exists():
        diagnostic = read_json(root / "diagnostic_samples.json")
        if diagnostic.get("checkpoint_hash") != file_sha256(root / "best.pt"):
            issues.append("diagnostic_checkpoint_mismatch")
        if not diagnostic.get("records"):
            issues.append("diagnostic_ids_missing")
    if (root / "diagnostic_samples.npz").exists():
        try:
            with np.load(root / "diagnostic_samples.npz", allow_pickle=False) as diagnostic:
                if not diagnostic.files or any(diagnostic[k].dtype.hasobject for k in diagnostic.files):
                    issues.append("diagnostic_arrays_invalid")
        except (ValueError, OSError):
            issues.append("diagnostic_arrays_invalid")
    if require_completion:
        if not (root / "completion.json").exists():
            issues.append("incomplete:no_completion_marker")
        else:
            completion = read_json(root / "completion.json")
            if completion.get("status") != "complete":
                issues.append("not_complete")
            if (root / "metrics.json").exists() and not completion_text_matches(root / "metrics.json", completion.get("metrics_hash"), completion.get("text_hash_algorithm")):
                issues.append("completion_hash_mismatch")
            if (root / "history.jsonl").exists() and not completion_text_matches(root / "history.jsonl", completion.get("history_hash"), completion.get("text_hash_algorithm")):
                issues.append("completion_history_hash_mismatch")
    # Recompute from saved records, rather than trusting a 'verified' flag.
    if not (root / "metric_recomputation.json").exists():
        issues.append("metric_recomputation_missing")
    elif (root / "predictions_val.jsonl").exists():
        witness = read_json(root / "metric_recomputation.json")
        if witness.get("status") != "verified" or witness.get("predictions_sha256") != file_sha256(root / "predictions_val.jsonl"):
            issues.append("metric_recomputation_hash_or_status_invalid")
        elif witness.get("kind") == "qa":
            reference = Path(witness.get("qa_reference_path", ""))
            if not reference.is_absolute():
                reference = root / reference
            if not reference.is_file() or file_sha256(reference) != witness.get("qa_reference_sha256"):
                issues.append("qa_reference_missing_or_changed")
            else:
                from eksperimen_model.m4human_evaluation import score_rows
                class Predictions:
                    def __iter__(self):
                        return iter_jsonl(root / "predictions_val.jsonl")
                    def __len__(self):
                        return row_count
                computed = score_rows(list(iter_jsonl(reference)), Predictions(), witness.get("selected_tasks"))
                if canonical_hash(computed) != canonical_hash(witness.get("metrics")):
                    issues.append("qa_metric_recomputation_mismatch")
                if (root / "metrics.json").exists() and canonical_hash(read_json(root / "metrics.json").get("selected_metrics")) != canonical_hash(computed):
                    issues.append("reported_qa_metrics_mismatch")
        else:
            totals = {}
            for row in iter_jsonl(root / "predictions_val.jsonl"):
                for key, metric in row.get("metrics", {}).items():
                    if not isinstance(metric, dict) or "count" not in metric:
                        continue
                    value = metric.get("sum", metric.get("error_sum"))
                    count = metric["count"]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or type(count) is not int or count < 0:
                        issues.append("metric_sum_count_invalid")
                        continue
                    total = totals.setdefault(key, [0.0, 0])
                    total[0] += value
                    total[1] += count
            expected = witness.get("metrics", {})
            summary = read_json(root / "metrics.json") if (root / "metrics.json").exists() else {}
            reported = summary.get("validation_physical", summary.get("validation", {}))
            if not totals or set(totals) != set(expected):
                issues.append("metric_recomputation_keys_missing")
            for key, (value, count) in totals.items():
                check = expected.get(key, {})
                if count != check.get("count") or not math.isclose(value, check.get("sum", check.get("error_sum", -1)), rel_tol=1e-5, abs_tol=1e-5):
                    issues.append(f"metric_recomputation_mismatch:{key}")
                public = reported.get(key, {})
                mean = public.get("mean", public.get("mean_m"))
                if count != public.get("count") or not math.isclose(value, public.get("sum", public.get("error_sum", -1)), rel_tol=1e-5, abs_tol=1e-5):
                    issues.append(f"reported_metric_mismatch:{key}")
                if (count and (not isinstance(mean, (int,float)) or not math.isclose(mean, value/count, rel_tol=1e-5, abs_tol=1e-5))) or (not count and mean is not None):
                    issues.append(f"reported_mean_mismatch:{key}")
    return issues
