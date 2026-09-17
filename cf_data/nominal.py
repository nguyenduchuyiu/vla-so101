"""Collect 5 spatial training scenes + 2 heldout scenes and export LeRobot."""

import argparse
import json
from pathlib import Path

from cf_data.collect import collect
from cf_data.layouts import generate_layouts
from smolvla_cf.export import export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/nominal_spatial'))
    parser.add_argument('--lerobot-out', type=Path, default=Path('data/lerobot_nominal_spatial'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    args = parser.parse_args()
    if args.workers < 1 or args.image_size < 32:
        raise ValueError('workers must be positive and image-size >= 32')
    for path in (args.out, args.lerobot_out):
        if path.exists():
            raise FileExistsError(f'Choose a fresh output directory: {path}')
    layouts = generate_layouts(args.seed)
    raw = collect(argparse.Namespace(
        out=args.out, overwrite=False, scenes=7, seed=args.seed,
        width=args.image_size, height=args.image_size, robot_noise=0.0,
        workers=args.workers, layouts=layouts,
    ))
    records = [json.loads(line) for line in (raw / 'meta/nominal_episodes.jsonl').read_text().splitlines()]
    counts = {i: sum(row['scene_index'] == i for row in records) for i in range(7)}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f'Empty scenes: {counts}; inspect meta/failures.jsonl before export')
    export_dataset(raw, args.lerobot_out, repo_id='local/so101_nominal_spatial',
                   horizon=83, target_fps=50, workers=args.workers,
                   episodes_per_shard=15, encoder_threads=2, nominal_only=True)
    train_count = sum(counts[i] for i in range(5))
    report = {'episodes_per_scene': counts, 'train_episodes': train_count,
              'heldout_episodes': counts[5] + counts[6],
              'failed_attempts': 105 - len(records), 'counterfactual_episodes': 0}
    (args.lerobot_out / 'meta/collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
