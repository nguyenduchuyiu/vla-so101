"""Deterministic, spatially separated layouts for nominal pretraining."""

import numpy as np


def generate_layouts(seed: int = 42) -> list[dict]:
    """Five training layouts and two heldout layouts, in metres (world XY).

    Colours have no fixed spatial slot. Every same-colour object/goal moves at
    least 6 cm between scenes; mean displacement is at least 12 cm. Unlabelled
    point sets must also differ, so a colour permutation alone is insufficient.
    """
    rng = np.random.default_rng(seed)
    layouts = []
    for attempt in range(100000):
        points = []
        for _ in range(300):
            xy = rng.uniform([0.24, -0.17], [0.40, 0.17])
            if np.linalg.norm(xy) > 0.425:
                continue
            if all(np.linalg.norm(xy - p) >= 0.075 for p in points):
                points.append(xy)
            if len(points) == 8:
                break
        if len(points) != 8:
            continue
        xy = np.asarray(points)[rng.permutation(8)]
        # Both objects and goals span depth, rather than forming two fixed rows.
        if np.ptp(xy[:5, 0]) < 0.08 or np.ptp(xy[5:, 0]) < 0.06:
            continue
        separated = True
        for layout in layouts:
            old = np.asarray(layout['objects_xy'] + layout['targets_xy'])
            displacement = np.linalg.norm(xy - old, axis=1)
            nearest = np.linalg.norm(xy[:, None] - old[None], axis=2)
            if (displacement.min() < 0.06 or displacement.mean() < 0.12
                    or (nearest.min(0).mean() + nearest.min(1).mean()) / 2 < 0.025):
                separated = False
                break
        if not separated:
            continue
        index = len(layouts)
        layouts.append(dict(scene_index=index, seed=seed + index,
                            split='train' if index < 5 else 'test',
                            objects_xy=xy[:5].tolist(), targets_xy=xy[5:].tolist()))
        if len(layouts) == 7:
            return layouts
    raise RuntimeError('Could not generate seven separated layouts')
