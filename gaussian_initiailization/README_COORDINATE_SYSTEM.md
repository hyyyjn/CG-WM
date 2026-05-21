# Stage 1 - Stage 2 좌표계 정리

이 문서는 Stage 1에서 학습한 Gaussian PLY를 Stage 2 collision proxy로 사용할 때 필요한 좌표계 변환을 정리한 문서입니다.

## 문제 상황

Stage 1의 3D Gaussian means는 capture dataset의 world frame에서 학습됩니다. 예를 들어 MuJoCo 바닥 위에 물체가 settle된 상태에서 이미지를 찍었다면, PLY 안의 `x, y, z` 좌표는 이미 그 world 위치와 회전이 반영된 값입니다.

반면 Stage 2 contact fitting은 object-local collision primitive를 기대합니다. 기존 코드는 PLY를 읽은 뒤 bbox 중심을 빼는 방식으로 local 좌표를 만들었습니다.

```text
local_centers = ply_centers - bbox_center
world_centers = local_centers + predicted_position
```

이 방식은 bbox center가 MuJoCo body origin과 같다는 보장이 없기 때문에 불안정합니다. Stage 1 결과에 floater가 있거나, 물체가 비대칭이거나, yaw jitter가 들어간 경우 bbox center가 실제 body origin과 어긋날 수 있습니다.

## Stage 1 pose 저장

`gaussian_initiailization/generate_mujoco_synthetic_dataset.py`는 Stage 1 capture 당시 target body의 world pose를 `dataset_manifest.json`에 저장합니다.

```json
"object_pose": {
  "xpos": [0.0, 0.0, 0.09],
  "xquat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "xmat_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}
```

각 필드의 의미는 다음과 같습니다.

- `xpos`: Stage 1 capture 당시 object body의 world position
- `xquat_wxyz`: 같은 pose의 quaternion, MuJoCo 형식 `(w, x, y, z)`
- `xmat_row_major`: 같은 rotation의 3x3 matrix를 row-major로 펼친 값

## PLY world 좌표를 object-local 좌표로 변환

`load_gaussian_collision_primitives_from_ply()`는 다음 인자를 받을 수 있습니다.

```python
world_translation
world_rotation
```

이 값들이 주어지면 Stage 1 PLY의 world-frame means를 object-local frame으로 변환합니다.

```text
local = R^T * (world - t)
```

구현에서는 point를 row vector로 다루기 때문에 다음처럼 계산합니다.

```python
centers_np = centers_np - t[None, :]
centers_np = centers_np @ R
```

명시적인 pose가 없으면 기존 방식처럼 bbox recenter를 fallback으로 사용합니다.

## Stage 2에서 다시 world 좌표로 변환

Stage 2 fitting 중에는 local collision centers를 다시 world frame으로 올립니다.

```text
world = R0 * local + position
```

구현에서는 다음과 같습니다.

```python
rotated_local = local_centers @ world_rotation.T
gaussian_centers = rotated_local + position.unsqueeze(0)
```

현재 `R0`는 Stage 1 capture 당시 저장된 orientation을 고정해서 사용합니다. 즉 이 경로는 아직 translation dynamics 중심입니다. time-varying quaternion rollout, torque, inertia를 포함한 full angular dynamics는 아직 연결되어 있지 않습니다.

## 실행 방법

Stage 1 capture tree와 Stage 2 episode tree가 다른 위치에 있다면 `--stage1_dataset_root`를 명시합니다.

```bash
python gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <stage2_episode_root> \
  --stage1_ply <stage1_point_cloud.ply> \
  --stage1_dataset_root <stage1_dataset_root>
```

`--stage1_dataset_root`를 생략하면 fit script는 `episode_root` 주변에서 `dataset_manifest.json`을 찾고, `object_pose`를 찾지 못하면 bbox recenter 방식으로 fallback합니다.

## 변경 파일 요약

- `gaussian_initiailization/generate_mujoco_synthetic_dataset.py`
  - `dataset_manifest.json`에 `object_pose`를 저장합니다.
- `gaussian_initiailization/stage2/differentiable_collision_detection.py`
  - Stage 1 PLY loader에 pose 기반 world-to-local 변환을 추가했습니다.
- `gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py`
  - `--stage1_dataset_root` 옵션을 추가했습니다.
  - Stage 1 `object_pose`를 읽습니다.
  - pose-aware PLY loading을 사용합니다.
  - collision center를 world frame으로 복원할 때 Stage 1 capture rotation을 적용합니다.
