"""Run CPU implementation checks without real data or pretrained downloads.

Use --output to retain the exact checks, runtime versions and source hashes.
This is a development gate; it never authorizes scientific training.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from eksperimen_model.utils.m4human_runtime import atomic_json, file_sha256

CHECKS = (
    "test_m4human_kinematics_contract",
    "test_m4human_encoder_contract",
    "test_m4human_motion_contract",
    "test_m4human_language_contract",
    "test_m4human_physical_contract",
    "test_m4human_integration",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New JSON report path; never overwrite prior evidence")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root/"eksperimen_model").rglob("*m4human*.py"))
    sources += sorted((root/"eksperimen_model/configs").glob("m4human*.yaml"))
    sources += [root/"requirements-m4human.txt", root/"eksperimen_model/utils/__init__.py"]
    source_hashes = {str(path.relative_to(root)): file_sha256(path) for path in sources}
    results = []
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="m4human_test_runtime_") as temporary:
        environment = dict(os.environ, MPLCONFIGDIR=str(Path(temporary)/"matplotlib"),
                           XDG_CACHE_HOME=temporary, TOKENIZERS_PARALLELISM="false")
        for name in CHECKS:
            command = [sys.executable, "-m", "eksperimen_model." + name]
            before = time.perf_counter()
            try:
                process = subprocess.run(command, cwd=root, env=environment, text=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
                result = {"module": name, "exit_code": process.returncode, "output": process.stdout}
            except subprocess.TimeoutExpired as error:
                result = {"module": name, "exit_code": -1, "output": "timeout after 600s"}
            result["seconds"] = time.perf_counter() - before
            results.append(result)
            print(f"{'PASS' if result['exit_code'] == 0 else 'FAIL'} {name} ({result['seconds']:.1f}s)", flush=True)
            if result["exit_code"]:
                print(result["output"], flush=True)
    changed = any(file_sha256(root/path) != digest for path,digest in source_hashes.items())
    passed = all(result["exit_code"] == 0 for result in results) and not changed
    packages = ("torch", "numpy", "scipy", "lmdb", "msgpack", "transformers", "accelerate", "smplx", "matplotlib", "pyyaml")
    report = {"status": "passed" if passed else "failed", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "data_kind": "synthetic",
              "scientific_gates": "not_run", "python": platform.python_version(),
              "platform": platform.platform(), "device": "cpu",
              "versions": {package: importlib.metadata.version(package) for package in packages},
              "source_sha256": source_hashes, "source_changed_during_checks": changed,
              "checks": results}
    if args.output:
        atomic_json(args.output, report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
