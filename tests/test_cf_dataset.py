"""Unit tests for the cf_data pipeline and the cf_balanced handler.

Plain ``test_*`` functions (pytest-style) plus a ``__main__`` runner, so they run
both under ``pytest tests/`` and ``python tests/test_cf_dataset.py``. These tests are
env-free: they exercise core helpers, the action-chunk/delta convention, the balance
math, and the handler against a synthetic anchors.jsonl + npz built in a temp dir.
The full env-dependent end-to-end check lives in ``cf_data/smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `python tests/test_cf_dataset.py`
# resolves the `cf_data` / `models` / `simvla_datasets` packages (pytest run from the
# repo root does this automatically; this makes the __main__ runner self-contained).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import tempfile

import numpy as np
import torch

from cf_data.core import (
    GRASP,
    PLACE,
    REACH_PICK,
    REACH_PLACE,
    STAGE_TO_PHASE,
    OBJECTIVE_COLORS,
    qpos_to_row,
    split_for_scene,
    stage_to_phase,
)
from models.action_hub import SO101DeltaActionSpace
from simvla_datasets.domain_handler.cf_balanced import CFBalancedHandler
from simvla_datasets.utils import action_slice

D = 6  # proprio/action dim: 5 arm joints (deg) + gripper %


# --------------------------------------------------------------------------- #
# Phase mapping (plan §3)                                                     #
# --------------------------------------------------------------------------- #
def test_stage_to_phase_mapping_covers_all_eight_stages():
    expected = {
        "approach_object": REACH_PICK,
        "descend_to_object": REACH_PICK,
        "close_gripper": GRASP,
        "lift_object": GRASP,
        "move_above_target": REACH_PLACE,
        "lower_to_target": PLACE,
        "release_object": PLACE,
        "retreat": PLACE,
    }
    for stage, phase in expected.items():
        assert stage_to_phase(stage) == phase
    # All eight oracle stages are mapped (no orphan, no duplicate target check here).
    assert len(STAGE_TO_PHASE) == 8


def test_stage_to_phase_finished_maps_to_place():
    assert stage_to_phase("finished") == PLACE


def test_stage_to_phase_unknown_stage_raises():
    try:
        stage_to_phase("not_a_real_stage")
    except ValueError:
        return
    raise AssertionError("unknown stage did not raise ValueError")


# --------------------------------------------------------------------------- #
# Scene-level split (plan §11)                                                #
# --------------------------------------------------------------------------- #
def test_split_for_scene_is_contiguous_and_covers_all_scenes():
    n = 10
    splits = [split_for_scene(i, n) for i in range(n)]
    # Exactly three contiguous blocks: test, then val, then train.
    assert set(splits) <= {"test", "val", "train"}
    assert all(s in ("test", "val", "train") for s in splits)
    # Contiguity: each label forms one run.
    for label in ("test", "val", "train"):
        idxs = [i for i, s in enumerate(splits) if s == label]
        assert idxs == list(range(min(idxs), max(idxs) + 1)), f"{label} not contiguous"


def test_split_for_scene_guarantees_test_and_val_for_n_ge_3():
    for n in (3, 5, 10, 20):
        splits = [split_for_scene(i, n) for i in range(n)]
        assert "test" in splits, f"no test scene for n={n}"
        assert "val" in splits, f"no val scene for n={n}"
        assert "train" in splits, f"no train scene for n={n}"


def test_split_for_scene_does_not_crash_and_keeps_train_for_n_ge_2():
    # n=1 is fully degenerate: forcing >=1 test scene leaves no room for train, so the
    # single scene is test. n=2 forces >=1 test and keeps the remaining scene as train.
    splits1 = [split_for_scene(i, 1) for i in range(1)]
    assert all(s in ("test", "val", "train") for s in splits1)
    splits2 = [split_for_scene(i, 2) for i in range(2)]
    assert "train" in splits2 and "test" in splits2


def test_split_for_scene_is_deterministic_per_scene():
    n = 10
    for i in range(n):
        assert split_for_scene(i, n) == split_for_scene(i, n)


def test_split_for_scene_all_anchors_of_a_scene_share_split():
    # A scene's split depends only on scene_index, so all anchors of that scene match.
    n = 10
    for i in range(n):
        s = split_for_scene(i, n)
        # Calling with the same index always returns the same value -> same split.
        assert split_for_scene(i, n) == s


# --------------------------------------------------------------------------- #
# Proprio encoding (plan §4: arm deg, gripper 0-100 %)                         #
# --------------------------------------------------------------------------- #
def test_qpos_to_row_converts_arm_to_degrees_and_gripper_to_percent():
    lower, upper = math.radians(-10.0), math.radians(100.0)
    qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, math.radians(40.0)])
    row = qpos_to_row(qpos)
    assert row.shape == (D,)
    # Arm joints: radians -> degrees.
    np.testing.assert_allclose(row[:5], np.degrees(qpos[:5]), rtol=1e-9, atol=1e-9)
    # Gripper: linear map of the rad value onto [0, 100] over [lower, upper].
    expected_gripper = (qpos[5] - lower) / (upper - lower) * 100.0
    np.testing.assert_allclose(row[5], expected_gripper, rtol=1e-9, atol=1e-9)
    # Closed (rad(-10)) -> 0 %, open (rad(100)) -> 100 %.
    assert qpos_to_row(np.array([0, 0, 0, 0, 0, lower]))[5] == 0.0
    assert qpos_to_row(np.array([0, 0, 0, 0, 0, upper]))[5] == 100.0


# --------------------------------------------------------------------------- #
# Action-chunk convention (plan §8)                                            #
# --------------------------------------------------------------------------- #
def test_action_slice_splits_abs_trajectory_into_proprio_and_action():
    H = 10
    abs_traj = torch.randn(H + 1, D)
    out = action_slice(abs_traj)
    torch.testing.assert_close(out["proprio"], abs_traj[0])
    torch.testing.assert_close(out["action"], abs_traj[1:].clone())
    assert out["action"].shape == (H, D)


def test_so101_delta_matches_plan_delta_convention():
    # Plan §8: delta_q[k] = future_q[k] - current_q for arm; gripper absolute.
    current = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # anchor proprio
    future = torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 0.0],   # k=1, gripper closes
                          [12.0, 24.0, 36.0, 48.0, 60.0, 100.0]])  # k=2, gripper opens
    space = SO101DeltaActionSpace()
    delta = space._to_delta(current, future)
    # Arm: future - current (per plan §8).
    torch.testing.assert_close(delta[..., :5], future[..., :5] - current[..., None, :5])
    # Gripper: absolute future value, NOT differenced.
    torch.testing.assert_close(delta[..., 5], future[..., 5])


# --------------------------------------------------------------------------- #
# Balance math (plan §9)                                                      #
# --------------------------------------------------------------------------- #
def test_balance_math_is_50_50_and_phase_balanced():
    # N_rp RP groups (1 nominal + 4 CF each) and N_rp nominal-only anchors per other
    # phase. nominal = 4*N_rp (one nominal branch per group/anchor), cf = 4*N_rp.
    N_rp = 5
    nominal_rp = N_rp          # 1 nominal branch per RP group
    cf_rp = 4 * N_rp           # 4 CF branches per RP group
    nominal_other = 3 * N_rp   # GRASP + REACH_PLACE + PLACE, 1 nominal branch each
    nominal_total = nominal_rp + nominal_other  # 4*N_rp
    cf_total = cf_rp                              # 4*N_rp
    total = nominal_total + cf_total
    assert nominal_total == 4 * N_rp
    assert cf_total == 4 * N_rp
    assert nominal_total == cf_total                       # 50/50
    assert abs(nominal_total / total - 0.5) < 1e-9        # ratio 0.5
    # Phase distribution: 12.5 % each nominal phase + 50 % CF REACH_PICK.
    expected = {
        "nominal_REACH_PICK": nominal_rp / total,
        "nominal_GRASP": N_rp / total,
        "nominal_REACH_PLACE": N_rp / total,
        "nominal_PLACE": N_rp / total,
        "cf_REACH_PICK": cf_rp / total,
    }
    for k, v in expected.items():
        assert abs(v - 0.125 if k != "cf_REACH_PICK" else v - 0.5) < 1e-9, k


# --------------------------------------------------------------------------- #
# Handler against a synthetic anchors.jsonl + npz (plan §6/§7/§10)             #
# --------------------------------------------------------------------------- #
def _identity_aug(image):
    """Deterministic PIL -> [C,H,W] float tensor (no resize) for unit tests."""
    arr = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(arr).permute(2, 0, 1)


def _build_synthetic_dataset(root: Path) -> dict:
    """One RP group (5 branches, 5 objectives) + one nominal-only anchor (1 branch)."""
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    (root / "cf_anchors").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    T, H, HW = 6, 4, 8
    anchor_frame = 2
    anchor_proprio = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 20.0], dtype=np.float32)
    # Nominal episode: state[t] and distinct per-frame images so we can check frame t is used.
    state = np.stack(
        [np.array([t * 1.0, t * 2.0, t * 3.0, t * 4.0, t * 5.0, t * 10.0], dtype=np.float32) for t in range(T)]
    )
    assert np.allclose(state[anchor_frame], anchor_proprio)
    overhead = np.stack([np.full((HW, HW, 3), t, dtype=np.uint8) for t in range(T)])
    wrist = np.stack([np.full((HW, HW, 3), t + 100, dtype=np.uint8) for t in range(T)])
    ep_path = "episodes/ep_nominal.npz"
    np.savez_compressed(
        root / ep_path,
        **{
            "observation.state": state,
            "observation.images.overhead": overhead,
            "observation.images.wrist": wrist,
        },
    )
    # RP group: nominal branch future == nominal continuation state[3:7] (repeat-last pad);
    # CF branches diverge per objective.
    nominal_future = np.concatenate(
        [state[3:7], np.repeat(state[5:6], max(0, H - (T - 3)), axis=0)], axis=0
    )[:H].astype(np.float32)
    rp_futures = np.stack(
        [nominal_future]
        + [anchor_proprio[None] + np.stack([np.arange(1, H + 1) * (j + 1)] * D, axis=1) for j in range(1, 5)]
    ).astype(np.float32)
    assert rp_futures.shape == (5, H, D)
    cf_path = "cf_anchors/an_rp_group.npz"
    np.savez_compressed(
        root / cf_path,
        future_chunks=rp_futures,
        objective_ids=np.arange(5, dtype=np.int8),
        is_counterfactual=np.array([False, True, True, True, True]),
        anchor_proprio=anchor_proprio,
    )
    # Nominal-only anchor: one GRASP branch.
    nom_only_future = (anchor_proprio[None] + np.stack([np.arange(1, H + 1)] * D, axis=1)).astype(np.float32)
    nom_only_cf = "cf_anchors/an_nominal_only.npz"
    np.savez_compressed(
        root / nom_only_cf,
        future_chunks=nom_only_future[None],
        objective_ids=np.array([2], dtype=np.int8),
        is_counterfactual=np.array([False]),
        anchor_proprio=anchor_proprio,
    )
    anchors = [
        {
            "anchor_id": "rp_group",
            "nominal_episode_path": ep_path,
            "cf_path": cf_path,
            "anchor_frame": anchor_frame,
            "phase": REACH_PICK,
            "n_branches": 5,
            "branches": [
                {"branch_id": f"rp_group_obj{j}", "objective_id": j,
                 "instruction": f"pick up the {OBJECTIVE_COLORS[j]} block and place it on the white target",
                 "is_counterfactual": j != 0}
                for j in range(5)
            ],
        },
        {
            "anchor_id": "nominal_only",
            "nominal_episode_path": ep_path,
            "cf_path": nom_only_cf,
            "anchor_frame": anchor_frame,
            "phase": GRASP,
            "n_branches": 1,
            "branches": [
                {"branch_id": "nominal_only_obj2", "objective_id": 2,
                 "instruction": "pick up the green block and place it on the white target",
                 "is_counterfactual": False}
            ],
        },
    ]
    (root / "meta" / "anchors.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in anchors) + "\n", encoding="utf-8"
    )
    return {"T": T, "H": H, "HW": HW, "anchor_frame": anchor_frame,
            "anchor_proprio": anchor_proprio, "nominal_future": nominal_future}


def test_handler_yields_exactly_the_six_model_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        h = CFBalancedHandler(meta, num_views=2)
        gen = h.iter_episode(0, num_actions=4, image_aug=_identity_aug)
        sample = next(gen)
        assert set(sample.keys()) == {
            "language_instruction", "image_input", "image_mask",
            "proprio", "abs_trajectory", "flow_group_id",
        }


def test_handler_branches_share_image_proprio_and_group_id():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        info = _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        h = CFBalancedHandler(meta, num_views=2)
        gen = h.iter_episode(0, num_actions=info["H"], image_aug=_identity_aug)
        rp = [next(gen) for _ in range(5)]  # first 5 = RP group branches
        # Shared anchor image (criterion 6) and proprio across the 5 branches.
        for s in rp[1:]:
            torch.testing.assert_close(s["image_input"], rp[0]["image_input"])
            torch.testing.assert_close(s["proprio"], rp[0]["proprio"])
        # Shared flow_group_id (all branches of an anchor).
        gids = {int(s["flow_group_id"]) for s in rp}
        assert gids == {0}
        # Proprio == anchor_proprio.
        torch.testing.assert_close(rp[0]["proprio"], torch.as_tensor(info["anchor_proprio"], dtype=torch.float32))
        # Image == nominal frame at anchor_frame (overhead filled with frame idx;
        # _identity_aug keeps raw uint8 values as float, no normalization).
        frame_val = info["anchor_frame"]
        torch.testing.assert_close(rp[0]["image_input"][0, 0, 0, 0],
                                   torch.tensor(float(frame_val)))


def test_handler_nominal_branch_equals_nominal_continuation_and_cf_diverge():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        info = _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        h = CFBalancedHandler(meta, num_views=2)
        gen = h.iter_episode(0, num_actions=info["H"], image_aug=_identity_aug)
        rp = [next(gen) for _ in range(5)]
        nominal = rp[0]["abs_trajectory"][1:]                      # nominal branch future
        torch.testing.assert_close(nominal, torch.as_tensor(info["nominal_future"], dtype=torch.float32))
        # CF branches differ from the nominal continuation and from each other.
        for j in range(1, 5):
            cf = rp[j]["abs_trajectory"][1:]
            assert not torch.allclose(cf, nominal)
            for k in range(j + 1, 5):
                assert not torch.allclose(cf, rp[k]["abs_trajectory"][1:])


def test_handler_abs_trajectory_shape_and_repeat_last_padding():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        info = _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        h = CFBalancedHandler(meta, num_views=2)
        # num_actions > stored H -> repeat-last padding.
        num_actions = info["H"] + 3
        gen = h.iter_episode(0, num_actions=num_actions, image_aug=_identity_aug)
        s = next(gen)
        assert s["abs_trajectory"].shape == (num_actions + 1, D)
        torch.testing.assert_close(s["abs_trajectory"][0], s["proprio"])
        # Padded tail repeats the last real action.
        torch.testing.assert_close(s["abs_trajectory"][info["H"] + 1],
                                   s["abs_trajectory"][info["H"]])


def test_handler_image_mask_marks_first_two_views():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        for num_views in (2, 3):
            h = CFBalancedHandler(meta, num_views=num_views)
            gen = h.iter_episode(0, num_actions=4, image_aug=_identity_aug)
            s = next(gen)
            assert s["image_mask"].shape == (num_views,)
            assert s["image_mask"][:2].all()
            assert not s["image_mask"][2:].any()


def test_handler_rejects_missing_image_aug():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_synthetic_dataset(root)
        meta = {"dataset_name": "cf_balanced", "dataset_root": str(root),
                "anchors_file": "meta/anchors.jsonl"}
        h = CFBalancedHandler(meta, num_views=2)
        try:
            next(h.iter_episode(0, num_actions=4))
        except ValueError:
            return
        raise AssertionError("missing image_aug did not raise ValueError")


# --------------------------------------------------------------------------- #
# Runner so the file is executable without pytest.                             #
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 -- test runner reports any failure
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())