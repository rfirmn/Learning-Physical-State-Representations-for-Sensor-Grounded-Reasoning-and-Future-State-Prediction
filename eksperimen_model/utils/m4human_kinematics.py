"""Single authority for v3 physical targets, losses, metrics and task recipes.

Offline derivatives use actual seconds and float64. Masks are never inferred
from model errors: non-finite values on valid support are failures.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .m4human_runtime import canonical_hash

TASK_DEFINITIONS = {
    "root_speed_trend": {
        "answers": ("speeding_up", "slowing_down", "no_overall_trend", "unknown"),
        "question": "Selama interval ini, apakah speed pelvis secara keseluruhan cenderung meningkat, menurun, atau tidak menunjukkan tren naik/turun yang cukup kuat?",
        "body_parts": (),
    },
    "relative_limb_motion": {
        "answers": ("faster_left", "faster_right", "tie", "unknown"),
        "question": "Selama interval ini, sisi mana dari {body_part} bergerak lebih cepat relatif terhadap pelvis?",
        "body_parts": ("arms", "legs"),
    },
    "limb_onset_order": {
        "answers": ("left_before_right", "right_before_left", "approximately_simultaneous", "unknown"),
        "question": "Selama interval ini, sisi mana dari {body_part} mulai bergerak lebih dahulu relatif terhadap pelvis?",
        "body_parts": ("arms", "legs"),
    },
    "root_radial_direction": {
        "answers": ("approaching", "receding", "stationary", "unknown"),
        "question": "Selama interval ini, apakah pelvis mendekati radar, menjauhi radar, atau memiliki gerak radial yang kecil?",
        "body_parts": (),
    },
}
DEFAULT_RECIPE = {
    "label_recipe_version": "kinematic_recipe_v3",
    "derivative_recipe_id": "aligned12hz_endpoint_sg_v2",
    "length": 32, "derivative_window": 7, "degree": 2,
    "min_coverage": 0.75, "trend_deadband_mps2": 0.05,
    "trend_residual_mad_max_mps": 0.10, "limb_tie_margin_mps": 0.05,
    "onset_enabled": False, "onset_on_mps": 0.15, "onset_off_mps": 0.10,
    "onset_quiet": 3, "onset_confirmation": 3, "onset_tie_margin_s": 2 / 12,
    "radial_enabled": False, "radial_range_guard_m": 0.05,
    "radial_deadband_mps": 0.05, "radial_min_fraction": 0.80,
}
TARGET_STATUS = {"defined": "valid", "unknown": "unknown", "undefined": "excluded"}


def task_definition(task):
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unsupported task: {task}")
    return TASK_DEFINITIONS[task]


def build_reference_derivatives(position, time_s, validity, *, max_gap_s, window=7):
    """Causal degree-two fit independently inside this window; never use prehistory.

    position [T,...,3], validity [T,...], time_s [T]. Duplicate/reversed times,
    gaps, rank/conditioning failures invalidate only affected supports.
    """
    x = np.asarray(position, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    mask = np.asarray(validity)
    if window != 7 or x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError("v3 derivative requires seven samples and XYZ vectors")
    if mask.dtype != np.bool_ or mask.shape != x.shape[:-1] or times.shape != (len(x),):
        raise ValueError("Derivative shape/mask mismatch")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0 or not np.isfinite(times).all():
        raise ValueError("Finite timestamps and audited positive max_gap_s required")
    if not np.isfinite(x[mask]).all():
        raise ValueError("Non-finite position on valid derivative support")
    velocity = np.zeros_like(x)
    derivative_valid = np.zeros_like(mask)
    support = np.full((len(x), window), -1, dtype=np.int64)
    reasons = ["warmup"] * len(x)
    flat = x.reshape(len(x), -1, 3)
    flat_mask = mask.reshape(len(x), -1)
    flat_out = velocity.reshape(len(x), -1, 3)
    flat_valid = derivative_valid.reshape(len(x), -1)
    for end in range(window - 1, len(x)):
        indices = np.arange(end - window + 1, end + 1)
        support[end] = indices
        dt = np.diff(times[indices])
        if np.any(dt <= 0) or np.any(dt > max_gap_s):
            reasons[end] = "time_gap_or_nonmonotonic"
            continue
        tau = times[indices] - times[end]
        design = np.stack((np.ones(window), tau, tau ** 2), axis=-1)
        if np.linalg.matrix_rank(design) != 3 or np.linalg.cond(design) > 1e8:
            reasons[end] = "ill_conditioned_time_support"
            continue
        valid = flat_mask[indices].all(axis=0)
        if valid.any():
            observed = flat[indices][:, valid, :].reshape(window, -1)
            coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
            flat_out[end, valid] = coefficients[1].reshape(-1, 3)
            flat_valid[end, valid] = True
        reasons[end] = "valid" if valid.any() else "missing_support"
    return {"velocity": velocity, "derivative_valid": derivative_valid,
            "support_indices": support, "support_reason": reasons,
            "derivative_recipe_id": DEFAULT_RECIPE["derivative_recipe_id"]}


def derivative_support_mask(mask, window=7, time_dim=1):
    """Seven consecutive valid sensor states. Supports torch [B,T,...] masks."""
    if isinstance(mask, np.ndarray):
        return derivative_support_mask(torch.from_numpy(mask), window, time_dim).numpy()
    if mask.dtype != torch.bool or window != 7:
        raise ValueError("Expected boolean mask and v3 seven-frame support")
    if mask.ndim == 1:
        time_dim = 0
    result = torch.zeros_like(mask)
    if mask.shape[time_dim] >= window:
        reduced = mask.unfold(time_dim, window, 1).all(dim=-1)
        index = [slice(None)] * mask.ndim
        index[time_dim] = slice(window - 1, None)
        result[tuple(index)] = reduced
    return result


def build_physical_targets(P_gt_relative_m, r_gt_m, time_s, joint_valid, root_valid, *, max_gap_s):
    p, r = np.asarray(P_gt_relative_m), np.asarray(r_gt_m)
    jm, rm = np.asarray(joint_valid), np.asarray(root_valid)
    if p.shape != (len(r), 22, 3) or r.shape != (len(r), 3):
        raise ValueError("Expected [T,22,3] relative joints and [T,3] root")
    if jm.dtype != np.bool_ or rm.dtype != np.bool_ or jm.shape != p.shape[:-1] or rm.shape != r.shape[:-1]:
        raise ValueError("Invalid annotation masks")
    jm = jm & rm[:, None]
    pv = build_reference_derivatives(p, time_s, jm, max_gap_s=max_gap_s)
    rv = build_reference_derivatives(r, time_s, rm, max_gap_s=max_gap_s)
    return {"P_relative_m": p.copy(), "r_m": r.copy(),
            "v_relative_mps": pv["velocity"].astype(np.float32),
            "v_root_mps": rv["velocity"].astype(np.float32),
            "position_valid_joint": jm.copy(), "position_valid_root": rm.copy(),
            "velocity_valid_joint": pv["derivative_valid"],
            "velocity_valid_root": rv["derivative_valid"]}


def _selected(prediction, target, mask):
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("Prediction/target XYZ shapes differ")
    if mask.dtype != torch.bool or mask.shape != prediction.shape[:-1]:
        raise ValueError("Physical mask shape/type mismatch")
    prediction, target = prediction.float()[mask], target.float()[mask]
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite physical value on eligible support")
    return prediction, target


def kinematic_loss(prediction, target, scales, pelvis_index=0):
    """Four separately normalized groups; no NaN*0 or denominator invention."""
    if not 0 <= pelvis_index < 22:
        raise ValueError("Invalid audited pelvis index")
    terms, counts = {}, {}
    groups = (("p_joint", "P_relative_m", "position_valid_joint", 1.0),
              ("p_root", "r_m", "position_valid_root", 1.0),
              ("v_joint", "v_relative_mps", "velocity_valid_joint", scales["s_v_joint"]),
              ("v_root", "v_root_mps", "velocity_valid_root", scales["s_v_root"]))
    for name, key, mask_key, scale in groups:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid physical scale: {name}")
        mask = target[mask_key].clone()
        if name.endswith("joint"):
            mask[..., pelvis_index] = False
        p, y = _selected(prediction[key], target[key], mask)
        counts[name] = p.numel()
        # Empty selection remains connected to its prediction graph, finite even
        # when invalid padded entries contain NaN.
        terms["L_" + name] = ((p - y) / scale).square().mean() if p.numel() else p.sum()
    terms["loss"] = terms["L_p_joint"] + terms["L_p_root"] + 0.25 * (terms["L_v_joint"] + terms["L_v_root"])
    terms["counts"] = counts
    return terms


def physical_metrics(prediction, target, scales=None, pelvis_index=0):
    """Unaligned Euclidean errors, sums/counts and units for exact recomputation."""
    result = {}
    def metric(name, p, y, mask, unit, per_joint=False):
        selected_p, selected_y = _selected(p, y, mask)
        errors = (selected_p - selected_y).norm(dim=-1)
        entry = {"sum": errors.double().sum().item(), "count": errors.numel(), "unit": unit}
        entry["mean"] = entry["sum"] / entry["count"] if entry["count"] else None
        if "velocity" in name:
            speed_errors = (selected_p.norm(dim=-1) - selected_y.norm(dim=-1)).abs()
            entry["speed_error_sum"] = speed_errors.double().sum().item()
            entry["speed_error_mean"] = entry["speed_error_sum"] / entry["count"] if entry["count"] else None
        if per_joint:
            entry["per_joint"] = []
            for j in range(p.shape[-2]):
                a, b = _selected(p[..., j, :], y[..., j, :], mask[..., j])
                e = (a - b).norm(dim=-1)
                entry["per_joint"].append({"sum": e.double().sum().item(), "count": e.numel()})
        result[name] = entry
    joint_p = target["position_valid_joint"].clone()
    joint_v = target["velocity_valid_joint"].clone()
    joint_p[..., pelvis_index] = False
    joint_v[..., pelvis_index] = False
    metric("mpjpe_relative_m", prediction["P_relative_m"], target["P_relative_m"], joint_p, "m", True)
    metric("root_error_m", prediction["r_m"], target["r_m"], target["position_valid_root"], "m")
    global_mask = target["position_valid_joint"] & target["position_valid_root"].unsqueeze(-1)
    # Sanitize BEFORE addition, because invalid root may be NaN even at a valid joint.
    def global_positions(values):
        return torch.where(global_mask[..., None], values["P_relative_m"], 0) + torch.where(
            global_mask[..., None], values["r_m"].unsqueeze(-2), 0)
    metric("mpjpe_global_m", global_positions(prediction), global_positions(target), global_mask, "m", True)
    metric("velocity_relative_mps", prediction["v_relative_mps"], target["v_relative_mps"], joint_v, "m/s", True)
    metric("velocity_root_mps", prediction["v_root_mps"], target["v_root_mps"], target["velocity_valid_root"], "m/s")
    return result


def _recipe(recipe):
    supplied = recipe or {}
    if set(supplied) - set(DEFAULT_RECIPE) - {"radar_origin_m", "max_gap_s"}:
        raise ValueError("Unknown kinematic recipe field")
    result = {**DEFAULT_RECIPE, **supplied}
    if result["length"] != 32 or result["derivative_window"] != 7 or result["degree"] != 2:
        raise ValueError("Foreign derivative/window contract")
    if result["label_recipe_version"] != "kinematic_recipe_v3" or result["derivative_recipe_id"] != "aligned12hz_endpoint_sg_v2":
        raise ValueError("Foreign recipe identifier")
    if not 0 < result["min_coverage"] <= 1:
        raise ValueError("Coverage must be in (0,1]")
    for key in ("trend_deadband_mps2", "trend_residual_mad_max_mps", "limb_tie_margin_mps",
                "onset_on_mps", "onset_off_mps", "onset_tie_margin_s", "radial_range_guard_m", "radial_deadband_mps"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Invalid threshold: {key}")
    return result


def _first_onset(speed, times, recipe):
    quiet, confirm = recipe["onset_quiet"], recipe["onset_confirmation"]
    off, on = recipe["onset_off_mps"], recipe["onset_on_mps"]
    if not isinstance(quiet, int) or not isinstance(confirm, int) or min(quiet, confirm) < 1 or on <= off:
        raise ValueError("Invalid onset hysteresis recipe")
    if len(speed) < quiet or not (speed[:quiet] <= off).all():
        return {"onset_time_s": None, "confirmed_at_s": None, "reason": "left_censored"}
    armed, candidate, quiet_count, spikes = True, None, 0, 0
    for index in range(quiet, len(speed)):
        if not armed:
            quiet_count = quiet_count + 1 if speed[index] <= off else 0
            armed = quiet_count >= quiet
            continue
        if candidate is None and speed[index] >= on:
            candidate = index
        if candidate is not None:
            if speed[index] < on:
                candidate, armed, quiet_count = None, False, int(speed[index] <= off)
                spikes += 1
            elif index - candidate + 1 >= confirm:
                return {"onset_time_s": float(times[candidate]), "confirmed_at_s": float(times[index]),
                        "reason": None, "failed_spikes": spikes}
    return {"onset_time_s": None, "confirmed_at_s": None,
            "reason": "right_censored" if candidate is not None else "no_confirmed_onset", "failed_spikes": spikes}


def build_evidence(physical_values, time_s, source_validity, recipe=None, task="root_speed_trend",
                   body_part=None, joint_map=None, source="sensor_prediction"):
    definition = task_definition(task)
    settings = _recipe(recipe)
    if source not in {"sensor_prediction", "gt_reference"}:
        raise ValueError("Unknown evidence source")
    if definition["body_parts"] and body_part not in definition["body_parts"]:
        raise ValueError("Task requires an audited arms/legs group")
    if not definition["body_parts"] and body_part is not None:
        raise ValueError("Unexpected body_part")
    times = np.asarray(time_s, dtype=np.float64)
    if times.shape != (32,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Evidence requires exactly 32 increasing, finite sensor timestamps")
    eligible = np.arange(6, 32)
    result = {"source": source, "task": task, "body_part": body_part,
              "diagnostic_answer": "unknown", "evidence_status": "undefined", "reason": None,
              "numeric_aggregates": {}, "valid_count": 0, "eligible_count": 26,
              "task_interval_s": [float(times[6] - times[0]), float(times[-1] - times[0])],
              "support_frame_ids": [], "uncertainty": None, "recipe_hash": canonical_hash(settings),
              "label_recipe_version": settings["label_recipe_version"],
              "derivative_recipe_id": settings["derivative_recipe_id"]}
    def values(key, mask_key, shape):
        array, mask = np.asarray(physical_values[key], dtype=np.float64), np.asarray(source_validity[mask_key])
        if array.shape != shape or mask.dtype != np.bool_ or mask.shape != shape[:-1]:
            raise ValueError(f"Invalid physical evidence {key}")
        if not np.isfinite(array[mask]).all():
            raise ValueError(f"Non-finite valid evidence: {key}")
        return array[eligible], mask[eligible]
    if task in {"root_speed_trend", "root_radial_direction"}:
        velocity, valid = values("v_root_mps", "velocity_valid_root", (32, 3))
        speed = np.linalg.norm(velocity, axis=-1)
    else:
        if not joint_map or body_part not in joint_map.get("task_groups", {}):
            raise ValueError("Audited task_groups required; joint indices are never guessed")
        groups = joint_map["task_groups"][body_part]
        left, right = groups["left"], groups["right"]
        if len(left) != 2 or len(right) != 2 or len(set(left + right)) != 4 or any(type(j) is not int or j < 0 or j >= 22 for j in left + right):
            raise ValueError("Expected two distinct audited joints on each limb side")
        velocity, mask = values("v_relative_mps", "velocity_valid_joint", (32, 22, 3))
        valid = mask[:, left + right].all(axis=-1)
        left_speed = np.linalg.norm(velocity[:, left], axis=-1)
        right_speed = np.linalg.norm(velocity[:, right], axis=-1)
    if task == "root_radial_direction":
        if not settings["radial_enabled"] or "radar_origin_m" not in settings:
            raise ValueError("Radial task requires explicit origin/reliability gate")
        root, root_valid = values("r_m", "position_valid_root", (32, 3))
        origin = np.asarray(settings["radar_origin_m"], dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid radar origin")
        relative = root - origin
        distance = np.linalg.norm(relative, axis=-1)
        valid = valid & root_valid & (distance > settings["radial_range_guard_m"])
    result["valid_count"] = int(valid.sum())
    result["support_frame_ids"] = eligible[valid].tolist()
    if valid.sum() < math.ceil(26 * settings["min_coverage"]):
        result["reason"] = "insufficient_support"
        return result
    valid_time = (times[eligible] - times[0])[valid]
    if task == "root_speed_trend":
        speed = speed[valid]
        a, b = np.triu_indices(len(speed), k=1)
        slope = float(np.median((speed[b] - speed[a]) / (valid_time[b] - valid_time[a])))
        intercept = float(np.median(speed - slope * valid_time))
        residual = speed - (slope * valid_time + intercept)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        result["numeric_aggregates"] = {"slope_mps2": slope, "residual_mad_mps": mad, "intercept_mps": intercept}
        if mad > settings["trend_residual_mad_max_mps"]:
            result["reason"] = "unreliable_trend_fit"
            return result
        result["diagnostic_answer"] = "speeding_up" if slope > settings["trend_deadband_mps2"] else "slowing_down" if slope < -settings["trend_deadband_mps2"] else "no_overall_trend"
    elif task == "relative_limb_motion":
        a, b = float(np.median(left_speed[valid], axis=0).mean()), float(np.median(right_speed[valid], axis=0).mean())
        difference = a - b
        result["numeric_aggregates"] = {"speed_left_mps": a, "speed_right_mps": b, "difference_mps": difference}
        result["diagnostic_answer"] = "faster_left" if difference > settings["limb_tie_margin_mps"] else "faster_right" if difference < -settings["limb_tie_margin_mps"] else "tie"
    elif task == "limb_onset_order":
        if not settings["onset_enabled"] or "max_gap_s" not in settings:
            raise ValueError("Onset task requires reliability gate and audited time gap")
        if not valid.all() or np.any(np.diff(times) > settings["max_gap_s"]):
            result["reason"] = "onset_requires_complete_contiguous_support"
            return result
        a = _first_onset(left_speed.mean(axis=-1), valid_time, settings)
        b = _first_onset(right_speed.mean(axis=-1), valid_time, settings)
        result["numeric_aggregates"] = {"left": a, "right": b}
        if a["onset_time_s"] is None or b["onset_time_s"] is None:
            result["reason"] = a["reason"] or b["reason"]
            return result
        delta = a["onset_time_s"] - b["onset_time_s"]
        margin = settings["onset_tie_margin_s"]
        result["diagnostic_answer"] = "left_before_right" if delta < -margin else "right_before_left" if delta > margin else "approximately_simultaneous"
    else:
        radial = (velocity[valid] * relative[valid]).sum(axis=-1) / distance[valid]
        band = settings["radial_deadband_mps"]
        fractions = {"approaching": float((radial < -band).mean()), "receding": float((radial > band).mean()),
                     "stationary": float((np.abs(radial) <= band).mean())}
        result["numeric_aggregates"] = fractions
        answer = max(fractions, key=fractions.get)
        if fractions[answer] < settings["radial_min_fraction"]:
            result["reason"] = "mixed_radial_direction"
            return result
        result["diagnostic_answer"] = answer
    result["evidence_status"] = "defined"
    return result
