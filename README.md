# CG-WM

한양대학교 졸프용 ContactGaussian-WM 실험 저장소입니다.

현재 구현의 중심은 `gaussian_initiailization/` 아래에 있습니다. 폴더 이름은 기존 저장소 상태를 따라
`initiailization` 오타가 남아 있습니다.

이 저장소는 ContactGaussian-WM 전체를 완성한 코드라기보다, 아래 두 축을 구현하고 검증하는 연구용
프로토타입입니다.

- Stage 1: multi-view 관측에서 object-aware spherical Gaussian scene initialization
- Stage 2: spherical Gaussian proxy 기반 differentiable collision detection 및 contact dynamics

## Repository Layout

```text
gaussian_initiailization/
  train.py                                      # Stage 1 학습
  render.py                                     # 학습 결과 렌더 / foreground debug render
  metrics.py                                    # 렌더 결과 metric 계산
  run_scene_initialization_pipeline.py          # mask, COLMAP, visual hull, SAM2, train wrapper
  generate_mujoco_synthetic_dataset.py          # MuJoCo synthetic multi-view dataset 생성
  build_visual_hull.py                          # mask 기반 visual hull seed 생성
  extract_sam2_features.py                      # SAM2 feature map 추출
  extract_object_masks.py                       # 입력 이미지에서 object mask 생성
  estimate_masked_colmap.py                     # mask 반영 COLMAP 재추정
  assign_object_ids.py                          # 수동 object id 할당
  auto_assign_object_ids.py                     # 자동 object id 할당
  export_physics_scene.py                       # physics-friendly intermediate export
  scene/gaussian_model.py                       # spherical Gaussian model, foreground/object fields
  gaussian_renderer/                            # 3DGS rasterizer wrapper
  tools/                                        # Blender/MuJoCo/helper scripts
  stage2/
    differentiable_collision_detection.py
    differentiable_contact_graph.py
    differentiable_complementarity_free_contact_dynamics.py
    gaussian_splatting_rendering.py
    tune_collision_proxy.py
    _smoke_test_*.py
```

## Stage 1: Scene Initialization

Stage 1은 multi-view RGB, mask, camera pose, optional SAM2 feature를 받아 spherical Gaussian 표현을
학습합니다. 목표는 이후 physics stage에서 쓰기 쉬운 object-centric Gaussian representation을 만드는
것입니다.

### 구현된 내용

- 기본 3D Gaussian Splatting 학습, 렌더, metric 계산
- isotropic spherical Gaussian 제약
  - 3축 scale 평균 사용
  - rotation은 identity에 가깝게 고정
- geometry / appearance decoupled optimization
  - alternating optimization
  - joint optimization 옵션
- Stage 1 preset
  - `--stage1_preset contactwm`
  - SG-GS strict mode
  - SAM feature 필수화
  - geometry RGB pressure 제어
  - densification 조기 종료
  - 후반 appearance refine
- SAM2 feature supervision
  - `.npy` feature map 로딩
  - `--geometry_feature_dim`으로 3채널 이상 supervision 지원
- object mask prior supervision
  - `--masks_dir`
  - foreground mask BCE + L1 loss
- per-Gaussian learned foreground score
  - `foreground_logit`
  - checkpoint / PLY save-load round trip
  - foreground score render
  - foreground threshold render
- object id 저장 / 복구 / PLY export
- manual / automatic object grouping
- visual hull seed initialization
- masked COLMAP preprocessing path
- physics export용 metadata 생성
- densification statistics logging

### Stage 1 입력

지원하는 대표 입력 구조는 Blender/NeRF-style synthetic dataset입니다.

```text
<scene>/
  images/
    train/
    test/
  masks/
    train/
    test/
  transforms_train.json
  transforms_test.json
  points3d.ply
  visual_hull/
    visual_hull.ply
```

COLMAP 형식도 일부 지원합니다.

```text
<scene>/
  images/
  sparse/0/
    cameras.bin or cameras.txt
    images.bin or images.txt
```

### Stage 1 사용 예시

MuJoCo synthetic dataset 생성:

```bash
MUJOCO_GL=egl conda run -n mujoco python gaussian_initiailization/generate_mujoco_synthetic_dataset.py \
  --output_root gaussian_initiailization/output/mujoco_data \
  --scene_name box_eval_v24_t8_r512 \
  --object_type box \
  --train_views 24 \
  --test_views 8 \
  --width 512 \
  --height 512
```

visual hull seed 생성:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/build_visual_hull.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512/masks \
  --grid_resolution 128 \
  --max_points 200000
```

SAM2 feature 추출:

```bash
conda run -n sam2cpu python gaussian_initiailization/extract_sam2_features.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --output_dir sam_features_sam2 \
  --output_channels 9 \
  --feature_source high_res0
```

Stage 1 학습:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --model_path gaussian_initiailization/output/stage1_box_contactwm \
  --images images \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512/masks \
  --sam_features sam_features_sam2 \
  --geometry_feature_dim 9 \
  --sam_feature_weight 0.1 \
  --object_mask_weight 1.0 \
  --object_mask_bce_weight 2.0 \
  --init_mode visual_hull \
  --init_ply_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512/visual_hull/visual_hull.ply \
  --stage1_preset contactwm \
  --iterations 10000 \
  --eval \
  --disable_viewer
```

렌더:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --model_path gaussian_initiailization/output/stage1_box_contactwm \
  --images images \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512/masks \
  --sam_features sam_features_sam2 \
  --geometry_feature_dim 9 \
  --iteration 10000 \
  --skip_train \
  --eval
```

foreground threshold render:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --model_path gaussian_initiailization/output/stage1_box_contactwm \
  --iteration 10000 \
  --skip_train \
  --eval \
  --foreground_threshold 0.5
```

physics export:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/export_physics_scene.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_eval_v24_t8_r512 \
  --model_path gaussian_initiailization/output/stage1_box_contactwm \
  --iteration 10000
```

## Stage 2: Collision And Dynamics

Stage 2는 Stage 1에서 얻은 spherical Gaussian proxy를 이용해 differentiable collision detection과
contact dynamics를 구성하는 실험 코드입니다.

### 구현된 내용

Collision detection:

- spherical Gaussian union SDF
  - primitive distance
  - LogSumExp smooth min
  - sigmoid-blended inside penalty
  - analytic surface normal
- `evaluate_gaussian_union_sdf(...)`
  - `(Q, 3)` query 지원
  - `(B, Q, 3)` batch query 지원
  - `(G, 3)` / `(B, G, 3)` Gaussian centers 지원
- `aggregate_gaussian_union_contacts(...)`
  - batch별 aggregate contact
  - top-k / spatial contact patch 선택
- `GaussianCollisionBody`
  - local Gaussian proxy
  - optional local query points
  - quaternion / rotation matrix pose transform
  - world bounding sphere
  - world AABB
- `DifferentiableCollisionEngine`
  - world query vs Gaussian body contact
  - object-object bidirectional contact
  - bounding sphere / AABB broad phase
- `BodyPairContacts`
  - `a_to_b`, `b_to_a`
  - merged `patch_points`
  - `patch_normals`
  - `patch_weights`
  - `patch_penetrations`
  - `patch_signed_distances`

Contact dynamics:

- 기존 floor/sphere smoke dynamics 유지
- `ImpedanceFloorContactDynamics`
  - paper-style frictionless one-pair impedance contact
- `RigidBodyState`
  - position
  - quaternion
  - linear velocity
  - angular velocity
- `PairwiseGaussianBodyImpedanceDynamics`
  - object-object multi-contact patch 사용
  - patch별 impedance force
  - patch별 tangent basis / tangential velocity diagnostics
  - Coulomb-limited tangential damping force
  - diagonal 또는 full 3x3 inertia tensor 기반 angular update
  - linear velocity update
  - torque / angular velocity update
  - quaternion integration
  - dynamic/static body option

### Stage 2 사용 예시

cube-floor collision detection smoke:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_cube_floor_collision.py \
  --frames 120 \
  --query_resolution 17 \
  --proxy_resolution 5 \
  --num_contact_patches 4
```

collision proxy tuning sweep:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/tune_collision_proxy.py \
  --proxy_resolutions 3,4,5,6,7 \
  --radius_scales 0.70,0.85,1.00,1.15,1.30 \
  --query_resolution 17
```

결과:

```text
gaussian_initiailization/stage2/_outputs/cube_floor_collision/
  cube_floor_collision_summary.json
  cube_floor_collision_frames.json
  cube_floor_collision_plot.png

gaussian_initiailization/stage2/_outputs/collision_proxy_tuning/
  collision_proxy_tuning_summary.json
  collision_proxy_tuning_results.csv
```

object-object pairwise dynamics smoke:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_pairwise_contact_dynamics.py
```

multi-object pairwise contact graph smoke:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_pairwise_contact_graph.py
```

MuJoCo bouncing ball GT fit smoke:

```bash
conda run -n mujoco python gaussian_initiailization/stage2/_smoke_test_mujoco_bouncing_ball_fit.py \
  --mode generate_gt \
  --output_dir gaussian_initiailization/stage2/_outputs/mujoco_bouncing_ball_fit

conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_mujoco_bouncing_ball_fit.py \
  --mode fit \
  --gt_json gaussian_initiailization/stage2/_outputs/mujoco_bouncing_ball_fit/mujoco_bouncing_ball_trajectory.json \
  --output_dir gaussian_initiailization/stage2/_outputs/mujoco_bouncing_ball_fit
```

pairwise MuJoCo GT comparison:

```bash
conda run -n mujoco python gaussian_initiailization/stage2/_smoke_test_pairwise_mujoco_compare.py \
  --mode generate_gt \
  --output_dir gaussian_initiailization/stage2/_outputs/pairwise_mujoco_compare

conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_pairwise_mujoco_compare.py \
  --mode compare \
  --gt_json gaussian_initiailization/stage2/_outputs/pairwise_mujoco_compare/mujoco_pairwise_gt_trajectory.json \
  --output_dir gaussian_initiailization/stage2/_outputs/pairwise_mujoco_compare
```

결과:

```text
gaussian_initiailization/stage2/_outputs/pairwise_contact_dynamics/
  pairwise_contact_dynamics_summary.json
  pairwise_contact_dynamics_frames.json
  pairwise_contact_dynamics_plot.png
  pairwise_contact_dynamics_rollout_plot.png

gaussian_initiailization/stage2/_outputs/pairwise_contact_graph/
  pairwise_contact_graph_summary.json

gaussian_initiailization/stage2/_outputs/mujoco_bouncing_ball_fit/
  mujoco_bouncing_ball_trajectory.json
  stage2_bouncing_ball_fit_summary.json
  stage2_bouncing_ball_fit_frames.json
  stage2_bouncing_ball_fit_plot.png

gaussian_initiailization/stage2/_outputs/pairwise_mujoco_compare/
  mujoco_pairwise_gt_trajectory.json
  pairwise_mujoco_compare_summary.json
  pairwise_mujoco_compare_frames.json
  pairwise_mujoco_compare_plot.png
```

검증된 항목:

- contact patch 생성
- broad phase overlap
- patch별 impedance force
- linear/angular update
- finite gradient

직접 API 사용 예시:

```python
import torch

from gaussian_initiailization.stage2.differentiable_collision_detection import (
    CollisionEngineConfig,
    DifferentiableCollisionEngine,
    GaussianCollisionBody,
    make_box_surface_query_points,
)
from gaussian_initiailization.stage2.differentiable_complementarity_free_contact_dynamics import (
    PairwiseGaussianBodyImpedanceDynamics,
    PairwiseImpedanceDynamicsConfig,
    RigidBodyState,
)

queries = make_box_surface_query_points([0.1, 0.1, 0.1], grid_resolution=3)
radii = torch.full((queries.shape[0],), 0.025)
body = GaussianCollisionBody(queries, radii, queries)

engine = DifferentiableCollisionEngine(CollisionEngineConfig(num_contact_patches=4))
contacts = engine.body_pair_contacts(
    body,
    torch.tensor([0.0, 0.0, 0.0]),
    body,
    torch.tensor([0.15, 0.0, 0.0]),
)

dynamics = PairwiseGaussianBodyImpedanceDynamics(
    body,
    body,
    stiffness=torch.tensor(250.0),
    damping=torch.tensor(10.0),
    config=PairwiseImpedanceDynamicsConfig(gravity=(0.0, 0.0, 0.0)),
)

state_a = RigidBodyState(
    position=torch.tensor([0.0, 0.0, 0.0]),
    quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    linear_velocity=torch.tensor([1.0, 0.0, 0.0]),
    angular_velocity=torch.zeros(3),
)
state_b = RigidBodyState(
    position=torch.tensor([0.15, 0.0, 0.0]),
    quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    linear_velocity=torch.zeros(3),
    angular_velocity=torch.zeros(3),
)

next_a, next_b, diagnostics = dynamics.step(state_a, state_b)
```

## Current Verification

최근 확인한 smoke tests:

```bash
python -m py_compile \
  gaussian_initiailization/stage2/differentiable_collision_detection.py \
  gaussian_initiailization/stage2/differentiable_complementarity_free_contact_dynamics.py \
  gaussian_initiailization/stage2/_smoke_test_cube_floor_collision.py \
  gaussian_initiailization/stage2/_smoke_test_pairwise_contact_dynamics.py
```

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_cube_floor_collision.py \
  --frames 20 \
  --query_resolution 9 \
  --proxy_resolution 4 \
  --num_contact_patches 4
```

```bash
conda run -n gaussian_splatting python gaussian_initiailization/stage2/_smoke_test_pairwise_contact_dynamics.py
```

현재 smoke 기준으로는 collision patch, broad phase, pairwise dynamics, gradient가 정상 출력됩니다.
cube analytic SDF 대비 Gaussian union SDF 오차는 `tune_collision_proxy.py`로 radius scale / proxy
resolution sweep을 돌려 후보를 고를 수 있습니다.

## TODO

### P0

완료:

- pairwise dynamics 결과를 MuJoCo GT trajectory와 비교
- Stage 2 pairwise dynamics를 `tools/run_stage2_mujoco_stage1_fit.py`에 mode로 연결
- Stage 1 실제 PLY에서 `GaussianCollisionBody`를 구성하는 helper 추가
- collision proxy tuning
  - radius scale sweep
  - proxy resolution sweep
  - foreground score / opacity 기반 primitive filtering
- object-object dynamics smoke를 여러 step rollout으로 확장

### P1

완료:

- broad phase를 bounding sphere에서 AABB / spatial hash로 확장
  - `CollisionEngineConfig.broad_phase_mode`에 `sphere` / `aabb` 지원
  - `GaussianCollisionBody.world_aabb()` 추가
  - contact graph에서 `candidate_pair_mode="spatial_hash"` 지원
  - `tools/run_stage2_mujoco_stage1_fit.py`에 `--pairwise_broad_phase_mode` 추가
- multi-object scene에서 pairwise contact graph 구성
  - `stage2/differentiable_contact_graph.py` 추가
  - multi-body node / pairwise edge / adjacency matrix / active edge serialization 지원
  - `_smoke_test_pairwise_contact_graph.py`로 3-body graph와 gradient 연결 검증
- quaternion / angular velocity fit loop 연결
  - `tools/run_stage2_mujoco_stage1_fit.py`의 `pairwise_impedance` mode가 trajectory의 `quaternion_wxyz` / `angular_velocity`를 읽음
  - `--orientation_loss_weight`, `--fit_initial_angular_velocity`, `--angular_velocity_l2_weight` 추가
  - `predicted_trajectory.json`에 target/predicted quaternion 기록
- inertia tensor를 diagonal 이상으로 확장
- friction Jacobian 추가
- tangential damping / Coulomb-like friction approximation 추가

### P2

- foreground/background 분리 품질 metric 추가
  - mask IoU
  - precision / recall
  - threshold sweep
  - foreground histogram
- Stage 1 visual hull bounds 개선
  - object bounds 수동 지정
  - mask ray 기반 bounds 추정
- Stage 1 output quality report 자동 생성
- train + render + metrics + physics export 통합 runner 정리

### P3

- differentiable image loss 연결
- real-world video 입력 지원 강화
- multi-instance object-aware Gaussian 학습
- ContactGaussian-WM paper 전체 pipeline과 실험표 정렬

## Demo 실행 방법

`run_demo.ps1`로 Stage 1 학습부터 side-by-side 비교 GIF 생성까지 한 번에 실행할 수 있다.

### 구체(sphere) 튀기 데모 (기본값)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
```

기본값이 `sphere_demo`로 설정되어 있다. 구체가 바닥에서 여러 번 튀는 상황을 MuJoCo GT와 Stage 2 예측이 얼마나 일치하는지 비교한다.

### 박스(box) 데모

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 `
    -SceneName box_demo -ObjectType box -ForegroundThreshold 0.55
```

### 빠른 테스트 (Stage 1 낮은 iter, 결과 품질 낮음)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 `
    -Stage1Iters 2000 -Stage2FitIters 100
```

### Stage 1 재사용 (이미 학습된 모델이 있을 때)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 -SkipStage1
```

### 결과물 위치

```text
demo_output\sphere_demo_stage2_fit\
  comparison_gt_vs_3dgs.gif   ← 메인 결과 (GT vs 3DGS side-by-side)
  gt_episode.gif              ← MuJoCo ground truth 영상
  stage2_fit_follow_view.gif  ← 위치 trajectory 비교
  fit_summary.json            ← 학습된 물리 파라미터 (restitution 등)
```

### Prerequisites

**환경 설치 (처음 한 번만):**

```powershell
# 1. conda 환경 생성
conda env create -f environment.yml

# 2. 환경 이름 확인 후 활성화 (기본: cg-wm)
conda activate cg-wm
```

환경 이름이 다르면 `-CondaEnv` 파라미터로 지정:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 -CondaEnv cg-wm
```

**요구 사항:**
- Windows, PowerShell 5.1 이상
- NVIDIA GPU (VRAM 8GB 이상 권장, CUDA 11.8)
- Visual Studio Build Tools (submodule CUDA 빌드용)

---

## Git / Output Policy

커밋 대상:

- source code
- root `README.md`
- lightweight config files

커밋 제외 대상:

- `gaussian_initiailization/output/`
- `gaussian_initiailization/stage2/_outputs/`
- `sam_features_sam2/`
- `masked_colmap/`
- `visual_hull/`
- `physics_export/`
- checkpoint, render image, GIF, PNG, NPZ, cache, `__pycache__`
