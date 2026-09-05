"""Export CF anchors to LeRobot v3 storage with an explicit future-chunk feature."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from .data import CAMERAS, JOINT_NAMES, CounterfactualDataset, to_delta_joint


def extend_futures(source, horizon):
    """Rebuild longer futures from saved snapshots, keeping the original anchors."""
    from cf_data.build import _build_nominal_branch, _build_place_group, _build_rp_group, _load_nominal
    from cf_data.collect import make_env
    from cf_data.core import REACH_PICK, REACH_PLACE

    episodes = _load_nominal(source.root)
    lookup = {ep["meta"]["episode_id"]: i for i, ep in enumerate(episodes)}
    height, width, _ = source.info["image_shape"]
    env = make_env(width, height, source_index=0, robot_init_qpos_noise=0.0)
    try:
        for i, anchor in enumerate(tqdm(source.anchors, desc="Extend oracle chunks")):
            ep_index, frame = lookup[anchor["episode_id"]], anchor["anchor_frame"]
            if anchor["phase"] == REACH_PICK:
                result = _build_rp_group(env, episodes, ep_index, frame, horizon, source.info["num_objectives"])
            elif anchor["phase"] == REACH_PLACE:
                result = _build_place_group(env, episodes, ep_index, frame, horizon, source.info["num_targets"])
            else:
                result = _build_nominal_branch(episodes, ep_index, frame, horizon)
            future, _, _, _, branches = result
            if branches != anchor["branches"]:
                raise ValueError(f"Regenerated branches differ at {anchor['anchor_id']}")
            if not np.allclose(future[:, :source.chunk_size], source.actions[i], atol=1e-4, rtol=1e-5):
                raise ValueError(f"Regenerated prefix differs at {anchor['anchor_id']}; check simulator version")
            source.actions[i] = future
    finally:
        env.close()
    source.chunk_size = horizon


def export_dataset(source_root, output, repo_id="local/so101_cf", horizon=50):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh export directory: {output}")
    anchors = [json.loads(line) for line in (Path(source_root) / "meta/anchors.jsonl").read_text().splitlines() if line]
    stored_horizon = min(row["horizon"] for row in anchors)
    source = CounterfactualDataset(source_root, chunk_size=min(stored_horizon, horizon), split="all")
    if horizon > stored_horizon:
        extend_futures(source, horizon)
    height, width, _ = source.info["image_shape"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "cf.action_chunk": {"dtype": "float32", "shape": (horizon, 6), "names": ["future_step", "joint"]},
        **{key: {"dtype": "image", "shape": (3, height, width), "names": ["channels", "height", "width"]} for key in CAMERAS},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=source.info["fps"],
                                   robot_type="so101", features=features, use_videos=False)
    manifest = []
    for i, j in tqdm(source.samples, desc="Write LeRobot episodes"):
        anchor = source.anchors[i]
        action = to_delta_joint(source.actions[i][j], source.states[i])
        dataset.add_frame({
            **source.images[i], "observation.state": source.states[i].copy(),
            "action": action[0].copy(), "cf.action_chunk": action,
            "task": anchor["branches"][j]["instruction"],
        })
        dataset.save_episode()
        manifest.append({"anchor_id": anchor["anchor_id"], "branch_id": anchor["branches"][j]["branch_id"],
                         "split": anchor["split"], "source_anchor_index": i, "source_branch_index": j})
    dataset.finalize()
    contract = {**source.contract(), "repo_id": repo_id, "image_shape": source.info["image_shape"],
                "storage": "lerobot_v3_single_observation_cf_chunk", "raw_source": str(Path(source_root).resolve())}
    (output / "meta/cf_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "meta/cf_samples.jsonl").write_text("\n".join(json.dumps(row) for row in manifest) + "\n")
    all_stats = {}
    for split in ("train", "val", "test", "all"):
        pairs = [(i, j) for i, j in source.samples if split == "all" or source.anchors[i]["split"] == split]
        if not pairs:
            continue
        states = np.stack([source.states[i] for i, _ in pairs]).astype(np.float64)
        actions = np.concatenate([to_delta_joint(source.actions[i][j], source.states[i]) for i, j in pairs]).astype(np.float64)
        all_stats[split] = {key: {"mean": values.mean(0).tolist(), "std": values.std(0).clip(1e-6).tolist()}
                            for key, values in (("observation.state", states), ("action", actions))}
    (output / "meta/cf_norm_stats.json").write_text(json.dumps(all_stats, indent=2) + "\n")
    print(f"Exported {len(source)} CF branches to {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/cf_nominal"))
    parser.add_argument("--output", type=Path, default=Path("data/lerobot_cf"))
    parser.add_argument("--repo-id", default="local/so101_cf")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    export_dataset(args.source, args.output, args.repo_id, args.chunk_size)


if __name__ == "__main__":
    main()
