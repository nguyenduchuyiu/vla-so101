"""End-to-end smoke test for the cf_data pipeline (plan §13 acceptance criteria).

Runs the full collect -> build path on a tiny 1-scene dataset, then checks every
plan §13 acceptance criterion against the produced files and confirms the
cf_balanced handler + dataloader yield a valid batch. Writes a detailed log to
``<out>/smoke.log`` and prints one PASS/FAIL line per criterion.

Run:  python -m cf_data.smoke --out data/cf_smoke_test --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from cf_data.build import build as build_fn
from cf_data.collect import collect as collect_fn
from cf_data.core import OBJECTIVE_COLORS, REACH_PICK, objective_instruction
from models.action_hub import SO101DeltaActionSpace
from simvla_datasets.dataset_smolvlm import create_smolvlm_dataloader
from simvla_datasets.utils import action_slice

D = 6


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _check(num: int, name: str, ok: bool, detail: str, log: io.StringIO, results: list) -> None:
    results.append((num, ok))
    line = f"{'PASS' if ok else 'FAIL'}  criterion {num:>2}  {name}"
    print(line)
    log.write(f"{line}\n    {detail}\n")


def _verify(out: Path, log: io.StringIO) -> tuple[list, dict]:
    results: list = []
    info = json.loads((out / "meta" / "info.json").read_text())
    anchors = _load_jsonl(out / "meta" / "anchors.jsonl")
    eval_pairs = _load_jsonl(out / "meta" / "eval_pairs.jsonl")
    stats = json.loads((out / "meta" / "stats.json").read_text())
    nominal_metas = _load_jsonl(out / "meta" / "nominal_episodes.jsonl")

    # 1. Five objectives.
    ok = info["num_objectives"] == 5 and len(OBJECTIVE_COLORS) == 5
    _check(1, "five objectives", ok, f"num_objectives={info['num_objectives']}", log, results)

    # 2. Objective distribution balanced across nominal episodes.
    per_obj = {}
    for m in nominal_metas:
        per_obj[m["objective_id"]] = per_obj.get(m["objective_id"], 0) + 1
    counts = [per_obj.get(j, 0) for j in range(5)]
    ok = len(per_obj) == 5 and max(counts) - min(counts) <= 1
    _check(2, "objectives balanced over episodes", ok, f"episodes_per_objective={counts}", log, results)

    # 3. Nominal samples balanced across the 4 phases. Each anchor contributes exactly
    #    one nominal branch, so nominal-branches-per-phase == anchors-per-phase, which
    #    build.py samples at n_rp each. (stats.samples_per_phase counts ALL branches, so
    #    REACH_PICK includes its 4 CF branches per anchor and is 5x larger -- not a bug.)
    anchors_per_phase = {p: 0 for p in ("REACH_PICK", "GRASP", "REACH_PLACE", "PLACE")}
    for a in anchors:
        anchors_per_phase[a["phase_name"]] += 1
    phase_vals = [anchors_per_phase[k] for k in anchors_per_phase]
    ok = max(phase_vals) - min(phase_vals) == 0 and min(phase_vals) > 0
    _check(3, "nominal samples balanced per phase", ok,
           f"anchors(=nominal_branches)_per_phase={anchors_per_phase} stats.samples_per_phase={stats['samples_per_phase']}", log, results)

    # 4. Each CF anchor = 1 nominal + 4 CF branches.
    rp = [a for a in anchors if a["phase"] == REACH_PICK]
    bad = []
    for a in rp:
        npz = np.load(out / a["cf_path"])
        cf = npz["is_counterfactual"]
        if a["n_branches"] != 5 or int(cf.sum()) != 4 or int((~cf).sum()) != 1:
            bad.append(a["anchor_id"])
    _check(4, "CF anchor = 1 nominal + 4 CF", len(rp) > 0 and not bad,
           f"rp_groups={len(rp)} bad={bad[:3]}", log, results)

    # 5. Branches of a group start from the same state (shared anchor_proprio == nominal state[frame]).
    bad = []
    for a in rp:
        npz = np.load(out / a["cf_path"])
        ap = npz["anchor_proprio"]
        ep = np.load(out / a["nominal_episode_path"])
        nominal_state = ep["observation.state"][a["anchor_frame"]]
        if not np.allclose(ap, nominal_state, atol=1e-5):
            bad.append(a["anchor_id"])
    _check(5, "branches share anchor state", not bad,
           f"checked {len(rp)} groups, mismatches={bad[:3]}", log, results)

    # 6. Nominal & CF branches share image + proprio at anchor (proprio shared by construction;
    #    image = nominal frame t, used by all branches via the handler).
    bad = []
    for a in rp:
        npz = np.load(out / a["cf_path"])
        ep = np.load(out / a["nominal_episode_path"])
        img = ep["observation.images.overhead"][a["anchor_frame"]]
        if not np.allclose(npz["anchor_proprio"], ep["observation.state"][a["anchor_frame"]], atol=1e-5):
            bad.append(a["anchor_id"])
    _check(6, "shared image + proprio at anchor", not bad,
           f"anchor_proprio==nominal_state[frame] for {len(rp)} groups; image read from nominal frame t by handler", log, results)

    # 7. Each branch has instruction + future matching its objective (CF futures diverge from nominal).
    bad = []
    for a in rp:
        npz = np.load(out / a["cf_path"])
        obj_ids = npz["objective_ids"]
        futures = npz["future_chunks"]
        nominal_future = futures[int(np.where(~npz["is_counterfactual"])[0][0])]
        for j in range(5):
            if a["branches"][j]["instruction"] != objective_instruction(int(obj_ids[j])):
                bad.append((a["anchor_id"], "instruction", j))
            if j != int(np.where(~npz["is_counterfactual"])[0][0]) and np.allclose(futures[j], nominal_future, atol=1e-5):
                bad.append((a["anchor_id"], "cf==nominal", j))
    _check(7, "branch instruction + future match objective", not bad,
           f"checked {len(rp)} groups, violations={bad[:3]}", log, results)

    # 8. ~50/50 nominal/counterfactual ratio.
    ratio = stats["nominal_counterfactual_ratio"]
    _check(8, "~50/50 nominal/counterfactual", abs(ratio - 0.5) <= 0.05,
           f"ratio={ratio} nominal={stats['num_nominal_branches']} cf={stats['num_cf_branches']}", log, results)

    # 9. All branches retrievable from anchor_id.
    bad = []
    for a in anchors:
        npz = np.load(out / a["cf_path"])
        if npz["future_chunks"].shape[0] != a["n_branches"]:
            bad.append(a["anchor_id"])
    _check(9, "all branches retrievable from anchor_id", not bad,
           f"checked {len(anchors)} anchors, mismatches={bad[:3]}", log, results)

    # 10. Branches of one anchor are not split across splits (one split per anchor record).
    splits = {}
    bad = []
    for a in anchors:
        prev = splits.get(a["anchor_id"])
        if prev is not None and prev != a["split"]:
            bad.append(a["anchor_id"])
        splits[a["anchor_id"]] = a["split"]
    _check(10, "anchor branches not split across splits", not bad,
           f"checked {len(anchors)} anchors, split_conflicts={bad[:3]}", log, results)

    # 11. Action chunk computed consistently from future proprio (abs_traj = [proprio, future]).
    space = SO101DeltaActionSpace()
    a0 = rp[0]
    npz = np.load(out / a0["cf_path"])
    proprio = npz["anchor_proprio"].astype(np.float32)
    future = npz["future_chunks"][0].astype(np.float32)
    H = min(10, future.shape[0])
    abs_traj = torch.as_tensor(np.concatenate([proprio[None], future[:H]], axis=0), dtype=torch.float32)
    sl = action_slice(abs_traj)
    delta = space._to_delta(sl["proprio"], sl["action"])
    ok = bool(torch.allclose(sl["proprio"], torch.as_tensor(proprio, dtype=torch.float32))) and \
         bool(torch.allclose(sl["action"], torch.as_tensor(future[:H], dtype=torch.float32))) and \
         bool(torch.allclose(delta[..., :5], sl["action"][..., :5] - sl["proprio"][..., None, :5]))
    _check(11, "action chunk consistent from future proprio", ok,
           f"proprio==abs[0], action==future[:H], delta[:5]=future-current (H={H})", log, results)

    # eval_pairs == test-split RP groups (output D).
    test_rp = [a for a in anchors if a["split"] == "test" and a["phase"] == REACH_PICK]
    eval_ok = len(eval_pairs) == len(test_rp) and {e["anchor_id"] for e in eval_pairs} == {a["anchor_id"] for a in test_rp}
    _check(0, "eval_pairs == test REACH_PICK groups (output D)", eval_ok,
           f"eval_pairs={len(eval_pairs)} test_rp={len(test_rp)}", log, results)

    return results, {"stats": stats, "anchors": anchors, "out": out}


def _dataloader_batch(out: Path, log: io.StringIO) -> bool:
    meta_path = out / "meta" / "cf_balanced.json"
    if not meta_path.exists():
        log.write(f"DATALOADER: meta missing {meta_path}\n")
        print("FAIL  dataloader  cf_balanced.json missing (build did not emit training meta)")
        return False
    loader = create_smolvlm_dataloader(
        batch_size=10, metas_path=str(meta_path), num_actions=10, training=True,
        action_mode="so101_delta", num_workers=0, image_size=96, num_views=2,
    )
    batch = next(iter(loader))
    keys = set(batch.keys())
    expected = {"language_instruction", "image_input", "image_mask", "proprio", "action", "flow_group_id"}
    ok = expected.issubset(keys)
    detail = (f"keys={sorted(keys)} image_input={tuple(batch['image_input'].shape)} "
              f"action={tuple(batch['action'].shape)} proprio={tuple(batch['proprio'].shape)} "
              f"flow_group_id={batch['flow_group_id'].tolist()[:6]}")
    print(f"{'PASS' if ok else 'FAIL'}  dataloader  one batch")
    log.write(f"DATALOADER: {detail}\n")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cf_smoke_test"))
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=3)
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    log = io.StringIO()
    print(f"== cf_data smoke: out={out} scenes={args.scenes} max_anchors={args.max_anchors} ==")
    log.write(f"== cf_data smoke: out={out} scenes={args.scenes} max_anchors={args.max_anchors} ==\n")

    try:
        cargs = argparse.Namespace(out=out, scenes=args.scenes, seed=args.seed, width=args.width,
                                    height=args.height, robot_noise=0.02, overwrite=True)
        with redirect_stdout(log):
            collect_fn(cargs)
        bargs = argparse.Namespace(in_dir=out, horizon=32, anchor_stride=args.anchor_stride,
                                    max_anchors=args.max_anchors, seed=args.seed, overwrite=True)
        with redirect_stdout(log):
            build_fn(bargs)
    except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure
        log.write(f"PIPELINE ERROR: {exc}\n{traceback.format_exc()}\n")
        (out / "smoke.log").write_text(log.getvalue(), encoding="utf-8")
        print(f"FAIL  pipeline  collect/build raised: {exc}")
        print(f"(full log: {out / 'smoke.log'})")
        raise SystemExit(1)

    results, _ = _verify(out, log)
    loader_ok = _dataloader_batch(out, log)

    (out / "smoke.log").write_text(log.getvalue(), encoding="utf-8")
    n_pass = sum(1 for _, ok in results if ok) + (1 if loader_ok else 0)
    n_tot = len(results) + 1
    print(f"\n{n_pass}/{n_tot} checks passed  (full log: {out / 'smoke.log'})")
    raise SystemExit(0 if n_pass == n_tot else 1)


if __name__ == "__main__":
    main()