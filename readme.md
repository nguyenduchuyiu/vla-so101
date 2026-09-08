# SO101 counterfactual fine-tuning with LeRobot SmolVLA

Fine-tune the official `lerobot/smolvla_base` checkpoint using LeRobot 0.6.1.
The local dataset adapter supplies each 30 Hz branch observation/state, its
instruction, and its padded future action chunk to `SmolVLAPolicy`.
No Hub upload is required. Data is exported with `LeRobotDataset.create` into
LeRobot v3 Parquet/image storage. A thin reader selects the explicit CF chunk
feature for the training loop; the unmodified `lerobot-train` temporal sampler
does not understand this extension (see below).

## Environment

Use an isolated Python 3.12 environment; the previous SimVLA environment has
incompatible Torch/Transformers versions.

```bash
uv venv .venv-smolvla --python 3.12
uv pip install --python .venv-smolvla/bin/python -r requirements.txt
source .venv-smolvla/bin/activate
```

## Collect and build counterfactual data

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m cf_data.collect \
  --out data/cf_nominal --scenes 3 --seed 0 --width 256 --height 256 --workers 3

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m cf_data.build \
  --in data/cf_nominal --horizon 50 --anchor-stride 4 --max-anchors 300 --workers 4
```

Choose a new directory for a new dataset; collect/build require `--overwrite`
to replace existing outputs. A failed objective is logged and dropped by itself;
other successful objectives in the scene are retained. Train/val/test are assigned
at scene level by the builder. `--split train` fails if no train anchors exist.
Collection parallelizes whole scenes, preserving the common initial state within
each scene. Build parallelizes CF validation and anchor rollouts. Every worker owns
an independent MuJoCo environment and EGL context; lower `--workers` if GPU memory
is constrained.

The builder restores each selected anchor and rolls every counterfactual branch to
terminal at the simulator's native 50 Hz. Unlike the earlier action-only format,
each CF trajectory stores overhead/wrist RGB, joint state, absolute PD target,
phase, timestamp, and terminal at every control step in `cf_trajectories/`.

## Export LeRobot data and delta-action statistics

```bash
python -m smolvla_cf.export \
  --source data/cf_nominal --output data/lerobot_cf \
  --fps 30 --chunk-size 50 \
  --workers 4 --encoder-threads 2 --episodes-per-shard 24
```

The builder stores expert `pd_joint_pos` commands: nominal commands are recovered
from the next frame's held `snapshot.ctrl`, and counterfactual commands are taken
directly from the oracle. Use a fresh export directory.

Each CF branch becomes a complete multi-frame episode beginning at its shared
anchor and ending at terminal. The 50 Hz source trajectory is mapped to the exact 30 Hz
timeline `t_k = k/30`: RGB and gripper state use nearest samples, arm state and
absolute arm targets use linear interpolation, and the absolute gripper target
uses zero-order hold. No irregular frame dropping is used.

Each original nominal trajectory is exported once. Selected anchors only add
their genuinely counterfactual trajectories; nominal suffixes shared by hundreds
of anchors are not duplicated. Cameras use streaming video storage by default,
avoiding per-frame PNG encoding. Use `--image-storage image` only when individual
lossless image files are explicitly required. Export uses independent LeRobot
shards so trajectory loading and H.264 encoding run on multiple CPU cores; the
shards are then merged without re-encoding or changing episode boundaries. Both
`--encoder-threads` and its CPU cost are per worker.

Every frame stores its real observation and a `[50,6]` future action chunk.
Targets past terminal repeat the final action and `action_is_pad` identifies
the padded positions. `next.done` marks the final observed frame. Arm actions are
encoded relative to the current frame state; gripper actions remain absolute.

LeRobot writes `meta/info.json`, `meta/stats.json`, episode/task metadata and
Parquet data. `meta/cf_samples.jsonl` preserves branch/split provenance.
`meta/cf_norm_stats.json` separately stores **valid, unpadded delta-chunk**
state/action mean/std for each available split; these are the statistics used by training.
The standard `meta/stats.json` action statistics cover only the first-action
column, so they must not replace the full-chunk statistics.

## Fine-tune

For a dataset with a nonempty training split, run on one CUDA GPU:

```bash
python -m smolvla_cf.train \
  --data data/lerobot_cf --split all \
  --pretrained lerobot/smolvla_base \
  --device cuda --batch-size 16 --chunk-size 50 --execute-steps 12 \
  --steps 10000 --lr 1e-4 \
  --warmup-steps 1000 --decay-steps 30000 --decay-lr 2.5e-6 \
  --save-every 1000 \
  --output runs/smolvla_cf
```

The currently collected local dataset has only one saved scene and all anchors
are labeled `test`. To explicitly use that whole dataset for an **overfit
experiment**, replace `--split train` with `--split all`. This provides no held-out
evaluation. Do not report training-scene success as generalization.

On Apple Silicon use `--device mps --batch-size 1`; a short wiring check can use
`--steps 1 --output tests/smolvla/delta_run`. Every run requires a fresh output directory.
The default fine-tunes the pretrained action expert and state projection with
AdamW, gradient clipping at 10, and the official SmolVLA warmup-plus-cosine
schedule: peak LR `1e-4`, 1,000 warmup steps, 30,000 decay steps, and final LR
`2.5e-6`. LeRobot automatically scales the warmup and decay to finish within a
shorter run (for example, a 10,000-step run uses about 333 warmup steps). The LR
actually used at each update is recorded in `metrics.jsonl`. Add `--train-vlm`
to also fine-tune the text layers; the vision encoder remains frozen. The script
is single-process and saves model weights/processors, not optimizer-resume state.

Statistics are loaded for the selected split and saved in the LeRobot
pre/post-processors alongside the policy. Logs are in `metrics.jsonl`; each
`checkpoint-NNNNNN/` is loadable with `SmolVLAPolicy.from_pretrained`.

## SO101 contract

| Field | Representation |
| --- | --- |
| State and action order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| State arm channels | Absolute joint angles in degrees |
| Action arm channels | Future joint targets minus the current frame state, in degrees |
| Gripper | Absolute opening percentage, 0–100 |
| camera1 | Overhead RGB |
| camera2 | Wrist RGB |
| Input pixels | CHW float in [0, 1]; LeRobot handles resizing and SigLIP scaling |
| Normalization | Per-channel mean/std from selected CF branches |
| Chunk | 50 future actions at the 30 Hz dataset rate; terminal tail repeats the final action |
| Execution | First 12 actions, then observe and replan; configurable |

The base model's internal padded state/action dimensions stay at 32 to preserve
pretrained weights. Its configured real state/action dimensions are both 6;
LeRobot excludes unused channels from the loss. Aloha-specific transforms are
disabled. Delta conversion is specific to the first five SO101 channels; the
gripper remains absolute. There are no sign flips or gripper thresholding.

For every action in a chunk, `delta_arm[k] = future_arm[k] - current_frame_arm`.
At rollout, unnormalize the prediction then recover
`target_arm[k] = current_frame_arm + delta_arm[k]` using the **same frame state for the whole chunk**.
Do not cumulatively sum deltas or add a new measured state at every step.
The simulator is explicitly configured as `pd_joint_pos`, so recovered targets
are converted to radians and sent to `env.step`. EE deltas would require a
separate Cartesian controller/IK layer, which this environment path does not use.

The [official base config](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json)
sets `chunk_size=50` and `n_action_steps=50`. Execute 12 is this project's rollout
setting, not the published checkpoint default. This custom training loop is not
a claim to reproduce every hyperparameter of the paper's training recipe.

## Replay one recorded trajectory

```bash
python -m cf_data.replay \
  --data data/cf_nominal --episode 0 \
  --output outputs/replay_ep0.mp4
```

The collector records state and held controller target immediately before each
action. Therefore `snapshot.ctrl[t+1]` is the command applied at frame `t`.
The final unobserved command is unavailable; replay holds the last recoverable
target briefly so the PD controller can settle.

## Evaluate in simulation

```bash
python -m smolvla_cf.evaluate \
  --checkpoint runs/smolvla_cf/checkpoint-010000 \
  --device cuda --source 0 --target 1 \
  --output outputs/smolvla_red_black.mp4
```

For an experimental smooth command path, join each 30 Hz prediction chunk to
the measured state with a bounded arm B-spline on a 100 Hz internal timeline,
then sample it at the simulator/controller's native 50 Hz:

```bash
python -m smolvla_cf.evaluate \
  --checkpoint runs/smolvla_cf/checkpoint-010000 \
  --device cuda --source 0 --target 1 \
  --interpolation bspline --spline-fps 100 \
  --output outputs/smolvla_red_black_bspline.mp4
```

The gripper remains zero-order held. The default interpolation is still
`linear`; validate task success before making B-spline execution the default.

To evaluate all 15 source/target objectives in each of scene seeds 0, 1 and 2
without reloading the checkpoint between episodes:

```bash
MUJOCO_GL=egl python -m smolvla_cf.evaluate_batch \
  --checkpoint runs/smolvla_cf-resume-003000/checkpoint-010000 \
  --device cuda --seeds 0 1 2 --max-replans 120 --execute-steps 3 \
  --interpolation linear --save-videos all \
  --output outputs/eval_ckpt010000_3scenes_15objectives
```

The batch writes one flushed record per episode to `results.jsonl` and an
aggregate `summary.json`. Add `--resume` with otherwise identical arguments to
continue an interrupted run.

Sources: 0=red, 1=blue, 2=green, 3=yellow, 4=purple.
Targets: 0=white, 1=black, 2=orange.
The evaluator reloads saved normalization and checks the SO101 contract. It
uses the selected linear or B-spline arm interpolation to reach the simulator's
50 Hz control clock; gripper targets use zero-order hold. It reports success,
grasp state, object-to-target distance, and command/motion smoothness metrics.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m smolvla_cf.validate --data data/lerobot_cf
python -m cf_data.smoke --out tests/smolvla/dataset --scenes 1 --max-anchors 3
```

The dataset smoke test rebuilds its output directory; use a fresh directory.
The tests check branch alignment, shared observations, delta encode/decode,
normalization, LeRobot round-trip, and split/horizon validation. See `tests/smolvla/REPORT.md` for
the validation actually performed.

References: [SmolVLA base](https://huggingface.co/lerobot/smolvla_base),
[LeRobot SmolVLA](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/policies/smolvla).
