# Task: Build a Balanced Counterfactual Dataset for Language-Conditioned Pick-and-Place

## 1. Objective

Xây dựng pipeline thu thập và tổ chức dữ liệu cho bài toán pick-and-place có **5 language objectives**, nhằm:

* duy trì khả năng thực hiện đầy đủ trajectory pick-and-place;
* tăng supervision tại phase mà language quyết định object cần chọn;
* hạn chế việc model dự đoán action chỉ dựa trên proprioception;
* tạo các nhóm counterfactual có cùng observation và robot state nhưng instruction và future trajectory khác nhau;
* tạo dataset có tỷ lệ xấp xỉ **50% nominal và 50% counterfactual**.

## 2. Task Definition

Environment chứa 5 objective hợp lệ, ví dụ:

```text
pick object A and place it at the target
pick object B and place it at the target
pick object C and place it at the target
pick object D and place it at the target
pick object E and place it at the target
```

Mỗi nominal episode:

* bắt đầu từ một initial robot pose hợp lệ;
* bắt đầu từ một scene configuration hợp lệ;
* chọn một trong 5 objectives làm instruction của episode;
* expert hoàn thành toàn bộ task thành công;
* lưu image observations, proprioception trajectory, instruction và phase labels.

Initial robot pose và object configuration cần được randomize trong miền hợp lệ để tăng state coverage.

Objective gốc cần được phân bố cân bằng giữa 5 objectives trên toàn bộ nominal dataset.

## 3. Trajectory Phases

Mỗi nominal trajectory được chia thành 4 phase:

### `REACH_PICK`

Robot chưa giữ object và đang di chuyển tới object được instruction chỉ định.

### `GRASP`

Robot căn chỉnh gần object, hạ gripper, đóng gripper và bắt đầu nhấc object.

### `REACH_PLACE`

Robot đang giữ object và di chuyển tới vị trí đặt.

### `PLACE`

Robot hạ object, mở gripper và rút tay khỏi vùng đặt.

Phase label được lấy trực tiếp từ subgoal hiện tại của expert collector.

## 4. Raw Nominal Trajectory Storage

Mỗi timestep của nominal trajectory cần lưu:

```text
images[t]
proprioception[t]
phase[t]
timestamp/frame index
```

Mỗi episode cần có metadata:

```text
instruction
objective_id
episode_id
scene_id
initial_state_id
success
```

`proprioception[t]` bao gồm joint positions của arm và gripper tại timestep `t`.

## 5. Nominal Sampling

Nominal training anchors được lấy cân bằng giữa 4 phase:

```text
25% REACH_PICK
25% GRASP
25% REACH_PLACE
25% PLACE
```

Nominal sample tại anchor timestep `t` gồm:

```text
images[t]
proprioception[t]
instruction
future proprioception chunk
phase[t]
objective_id
episode_id
anchor_id
is_counterfactual = false
```

## 6. Counterfactual Generation

Counterfactual chỉ được sinh từ các anchor thuộc phase `REACH_PICK`.

Với mỗi nominal `REACH_PICK` anchor:

* giữ nguyên scene;
* giữ nguyên robot state;
* giữ nguyên object poses;
* giữ nguyên image observation tại anchor;
* giữ nguyên proprioception tại anchor;
* thay objective gốc bằng từng objective còn lại;
* chạy expert từ cùng anchor state để tạo future proprioception trajectory tương ứng với objective mới.

Với 5 objectives, mỗi valid anchor tạo một group gồm:

```text
1 nominal branch
4 counterfactual branches
```

Ví dụ:

```text
same state + "pick A" → future proprio trajectory toward A
same state + "pick B" → future proprio trajectory toward B
same state + "pick C" → future proprio trajectory toward C
same state + "pick D" → future proprio trajectory toward D
same state + "pick E" → future proprio trajectory toward E
```

Chỉ giữ các anchor có đầy đủ 4 counterfactual hợp lệ.

## 7. Anchor and Branch Identity

Một `anchor_id` đại diện cho một state duy nhất:

```text
same scene
same robot state
same object states
same images
same proprioception
```

Nominal và các counterfactual branches từ cùng state dùng chung `anchor_id`.

Mỗi branch có:

```text
branch_id
objective_id
instruction
is_counterfactual
future proprioception trajectory
```

## 8. Action Target

Action target được suy ra từ proprioception trajectory.

Tại anchor timestep `t`, với action horizon `H`:

```text
current_q = proprioception[t]
future_q[k] = proprioception[t + k]
```

Arm action chunk:

```text
delta_q[k] = future_q[k] - current_q
```

với:

```text
k = 1, 2, ..., H
```

Gripper target có thể dùng:

```text
future gripper position relative to current gripper position
```

hoặc:

```text
absolute normalized future gripper position
```

## 9. Dataset Ratio

Giả sử nominal pool có tổng cộng `N` samples và được cân bằng giữa 4 phase:

```text
REACH_PICK   = N / 4
GRASP        = N / 4
REACH_PLACE  = N / 4
PLACE        = N / 4
```

Mỗi `REACH_PICK` anchor sinh 4 counterfactual branches:

```text
4 × N / 4 = N counterfactual samples
```

Dataset cuối:

```text
N nominal samples
N counterfactual samples
```

Tỷ lệ:

```text
50% nominal
50% counterfactual
```

Phân phối:

```text
12.5% nominal REACH_PICK
12.5% nominal GRASP
12.5% nominal REACH_PLACE
12.5% nominal PLACE
50.0% counterfactual REACH_PICK
```

## 10. Required Training Sample Fields

Mỗi training sample gồm:

```text
images
current proprioception
future proprioception chunk
language instruction
phase
objective_id
episode_id
scene_id
anchor_id
branch_id
is_counterfactual
```

Nominal và counterfactual samples cùng group có cùng:

```text
images
current proprioception
scene_id
anchor_id
```

và khác:

```text
instruction
objective_id
future proprioception chunk
branch_id
```

## 11. Dataset Splitting

Train, validation và test được split theo `scene_id` hoặc `initial_state_id`.

Toàn bộ branches thuộc cùng một `anchor_id` phải nằm trong cùng một split.

## 12. Desired Outputs

### A. Nominal trajectory dataset

```text
images
proprioception
instruction
objective_id
phase labels
scene and episode metadata
```

### B. Counterfactual branch dataset

Các `REACH_PICK` anchor groups:

```text
1 nominal branch
4 counterfactual branches
```

### C. Balanced training dataset

```text
50% nominal
50% counterfactual
```

Nominal samples cân bằng giữa 4 phases.

### D. Paired counterfactual evaluation set

```text
same images
same current proprioception
5 different instructions
5 corresponding future proprioception chunks
```

### E. Dataset statistics report

```text
number of nominal episodes
number of scenes
number of unique anchors
number of branches
samples per phase
samples per objective
nominal/counterfactual ratio
number of valid counterfactual groups
number of rejected anchors
trajectory length distribution
```

## 13. Acceptance Criteria

1. Có đủ 5 objectives.
2. Objective gốc được phân bố cân bằng giữa nominal episodes.
3. Nominal samples được cân bằng giữa 4 phases.
4. Mỗi counterfactual anchor có 1 nominal branch và 4 counterfactual branches hợp lệ.
5. Các branches trong cùng group bắt đầu từ cùng state.
6. Nominal và counterfactual branches có cùng image và current proprioception tại anchor.
7. Mỗi branch có instruction và future proprioception trajectory đúng với objective.
8. Dataset đạt xấp xỉ tỷ lệ 50/50 nominal–counterfactual.
9. Có thể truy xuất toàn bộ branches từ một `anchor_id`.
10. Các branches của cùng anchor không bị chia giữa các split.
11. Action chunk được tính nhất quán từ future proprioception.
