"""CPU contract checks: .venv/bin/python -m eksperimen_model.test_m4human_kinematics_contract"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_coeffs

from eksperimen_model.utils.m4human_kinematics import (
    DEFAULT_RECIPE, build_evidence, build_physical_targets,
    build_reference_derivatives, derivative_support_mask, kinematic_loss, physical_metrics,
)
from eksperimen_model.utils.m4human_runtime import (
    RunLogger, canonical_hash, load_checkpoint, rng_state, restore_rng_state,
    save_checkpoint, seed_everything, state_dict_hash, validate_run_artifacts,
)


def main():
    torch.set_num_threads(1)
    for times in (np.arange(32) / 12, np.cumsum(np.linspace(.071, .091, 32))):
        for degree in (0, 1, 2):
            position = np.repeat((times ** degree)[:, None], 3, axis=1)
            output = build_reference_derivatives(position, times, np.ones(32, bool), max_gap_s=.12)
            assert not output["derivative_valid"][:6].any() and output["derivative_valid"][6:].all()
            expected = degree * times ** max(degree - 1, 0)
            assert np.allclose(output["velocity"][6:, 0], expected[6:], atol=1e-8)
    times = np.arange(32) / 12
    random = np.random.default_rng(17).normal(size=(32, 3))
    derivative = build_reference_derivatives(random, times, np.ones(32, bool), max_gap_s=.12)
    reference = savgol_coeffs(7, 2, deriv=1, delta=1/12, pos=6, use="dot") @ random[:7]
    assert np.allclose(reference, derivative["velocity"][6], atol=1e-8)
    modified = random.copy()
    modified[15:] += 100
    assert np.array_equal(derivative["velocity"][:15], build_reference_derivatives(
        modified, times, np.ones(32, bool), max_gap_s=.12)["velocity"][:15])
    for bad_times in (np.concatenate((times[:16], times[16:] + 1)), np.concatenate((times[:16], times[15:31]))):
        result = build_reference_derivatives(random, bad_times, np.ones(32, bool), max_gap_s=.12)
        assert not result["derivative_valid"][16:22].any()
    valid = np.ones(32, bool)
    valid[12] = False
    random[12] = np.nan
    result = build_reference_derivatives(random, times, valid, max_gap_s=.12)
    assert np.isfinite(result["velocity"]).all() and not result["derivative_valid"][12:19].any()
    mask = torch.ones(2, 32, 22, dtype=torch.bool)
    mask[0, 12, 4] = False
    support = derivative_support_mask(mask)
    assert not support[:, :6].any() and not support[0, 12:19, 4].any() and support[1, 6:].all()

    values = {"v_root_mps": np.zeros((32, 3)), "v_relative_mps": np.zeros((32, 22, 3))}
    masks = {"velocity_valid_root": np.ones(32, bool), "velocity_valid_joint": np.ones((32, 22), bool)}
    values["v_root_mps"][6:, 0] = [.2] * 10 + [.3, .5, .8, .8, .5, .3] + [.2] * 10
    evidence = build_evidence(values, times, masks)
    assert evidence["diagnostic_answer"] == "no_overall_trend"
    assert evidence["numeric_aggregates"]["slope_mps2"] == 0
    for speed, expected in ((.2 + times, "speeding_up"), (3 - times, "slowing_down"), (np.ones(32), "no_overall_trend")):
        values["v_root_mps"][:, 0] = speed
        assert build_evidence(values, times, masks)["diagnostic_answer"] == expected
    masks["velocity_valid_root"][6:13] = False
    assert build_evidence(values, times, masks)["diagnostic_answer"] == "unknown"
    groups = {"task_groups": {"arms": {"left": [18, 20], "right": [19, 21]}}}
    values["v_relative_mps"][:, [18, 20], 0] = 1
    assert build_evidence(values, times, masks, task="relative_limb_motion", body_part="arms", joint_map=groups)["diagnostic_answer"] == "faster_left"
    masks["velocity_valid_joint"][6:13, 19] = False
    assert build_evidence(values, times, masks, task="relative_limb_motion", body_part="arms", joint_map=groups)["diagnostic_answer"] == "unknown"
    masks["velocity_valid_joint"][:] = True
    values["v_relative_mps"][:] = 0
    values["v_relative_mps"][12:, [18, 20], 0] = .3
    values["v_relative_mps"][18:, [19, 21], 0] = .3
    onset_recipe = {"onset_enabled": True, "max_gap_s": .12}
    onset = build_evidence(values, times, masks, onset_recipe, "limb_onset_order", "arms", groups)
    assert onset["diagnostic_answer"] == "left_before_right"
    assert onset["numeric_aggregates"]["left"]["confirmed_at_s"] > onset["numeric_aggregates"]["left"]["onset_time_s"]
    values["v_relative_mps"][6, [18, 20], 0] = .3
    assert build_evidence(values, times, masks, onset_recipe, "limb_onset_order", "arms", groups)["reason"] == "left_censored"

    target_np = build_physical_targets(np.zeros((32, 22, 3)), np.zeros((32, 3)), times,
                                      np.ones((32, 22), bool), np.ones(32, bool), max_gap_s=.12)
    target = {k: torch.as_tensor(v).unsqueeze(0) for k, v in target_np.items()}
    prediction = {k: target[k].float().clone().requires_grad_() for k in ("P_relative_m", "r_m", "v_relative_mps", "v_root_mps")}
    losses = kinematic_loss(prediction, target, {"s_v_joint": 1., "s_v_root": 1.})
    assert losses["loss"] == 0 and losses["counts"]["p_joint"] == 32 * 21 * 3
    losses["loss"].backward()
    metrics = physical_metrics(prediction, target)
    assert metrics["mpjpe_global_m"]["count"] == 32 * 22
    assert metrics["mpjpe_relative_m"]["count"] == 32 * 21
    assert metrics["velocity_relative_mps"]["count"] == 26 * 21
    empty = {key: value.clone() for key, value in target.items()}
    for key in empty:
        if "valid" in key:
            empty[key][:] = False
        else:
            empty[key][:] = float("nan")
    losses = kinematic_loss(prediction, empty, {"s_v_joint": 1., "s_v_root": 1.})
    assert torch.isfinite(losses["loss"]) and losses["loss"] == 0
    losses["loss"].backward()
    invalid = {**target, "r_m": torch.full_like(target["r_m"], float("nan"))}
    try:
        kinematic_loss(prediction, invalid, {"s_v_joint": 1., "s_v_root": 1.})
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite valid targets must fail")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        metadata = {"lineage": {"source_hash": canonical_hash("fixture")}, "data_kind": "synthetic"}
        save_checkpoint(root / "model.pt", model, metadata, optimizer=optimizer.state_dict(), rng=rng_state())
        clone = torch.nn.Linear(3, 2)
        loaded = load_checkpoint(root / "model.pt", clone, metadata["lineage"])
        assert state_dict_hash(model) == state_dict_hash(clone)
        restore_rng_state(loaded["rng"])
        try:
            load_checkpoint(root / "model.pt", clone, {"source_hash": "foreign"})
        except ValueError:
            pass
        else:
            raise AssertionError("Foreign lineage accepted")
        logger = RunLogger(root / "run", {"data_kind": "synthetic"}, "fixture")
        logger.log("epoch", loss=0.)
        result = logger.finish({"checks_passed": True})
        assert not result["analysis_ready"]
        assert "metric_recomputation_missing" in validate_run_artifacts(root / "run")
        # A fake completion marker cannot upgrade absent provenance/predictions.
        assert "missing:predictions_val.jsonl" in validate_run_artifacts(root / "run")
    seed_everything(42)
    print("PASS: derivative, support, task recipes, losses/metrics, checkpoint and incomplete-run guards (synthetic CPU)")


if __name__ == "__main__":
    main()
