# CG-WM

ContactGaussian-WM 논문 아이디어를 바탕으로 Stage 1 Gaussian scene initialization과 Stage 2 differentiable contact simulation을 실험하는 연구용 저장소입니다.

현재 코드는 논문 전체를 완전히 재현한 공식 구현체가 아니라, 주사위/박스 같은 rigid object를 대상으로 object-aware Gaussian proxy를 만들고, 그 proxy를 이용해 differentiable collision/contact dynamics를 검증하는 프로토타입입니다.

## ContactGaussian-WM 재현 단계 1–12

현재 구현한 Stage-II 재현 범위는 다음과 같습니다.

1. **Unified learnable Gaussian geometry**: object-local Gaussian center와 isotropic
   radius로 collision SDF와 Gaussian renderer의 형상을 일관되게 구성합니다. 최종값은
   `refined_geometry.json`에 저장됩니다.
2. **Image-space phys-geo objective**: predicted position/quaternion으로 Gaussian을
   렌더링하고 L1, MSE 또는 L1+D-SSIM loss를 계산합니다. foreground mask, sparse
   frame, calibrated multi-view와 image-only optimization을 지원합니다.
3. **Initial-state prefit**: 첫 RGB 프레임으로 position/quaternion을 맞추고, 처음 N개
   접촉 전 프레임의 constant-velocity motion으로 linear/angular velocity를 추정합니다.
   이 초기 상태는 본 physics fit 동안 고정됩니다.
4. **Complementarity-free dual-cone contact**: Appendix C의
   `J̃ = Jⁿ - μJᵈ`와 `SoftPlus(-K(hJ̃b+φ̃)-D(J̃b))`를 구현했습니다. K, D, μ가
   학습 가능하며 normal/friction force와 torque를 하나의 closed-form 식으로 계산합니다.
5. **Learnable mass–inertia**: 양수로 제약된 body mass와 body-frame diagonal inertia를
   K/D/μ와 함께 학습합니다. prior regularization을 지원하고 최종 M/I를 결과 JSON과
   train/holdout protocol에 보존합니다.
6. **Action-conditioned dynamics**: episode의 time-varying world-frame force/torque를
   비접촉 generalized force `τ(q,v,a)`에 포함합니다. Action은 학습된 mass/inertia를
   통해 선형·각가속도를 만들고, 이후 동일한 collision/contact step으로 이어집니다.
7. **Kinematic robot/environment contact**: 정적 floor/body B 대신 시간별
   position/quaternion trajectory를 갖는 kinematic Gaussian collider를 지원합니다.
   finite-difference 또는 기록된 linear/angular velocity가 contact Jacobian의 상대속도에
   들어가므로 움직이는 end-effector·hand link가 물체에 전달하는 접촉을 표현할 수 있습니다.
8. **Articulated multi-link kinematics**: revolute/prismatic/fixed joint로 구성한
   kinematic tree의 differentiable forward kinematics를 지원합니다. Joint trajectory에서
   모든 link pose와 `v,ω`를 계산하고, multi-body contact graph가 dynamic object와 여러
   Gaussian link의 접촉력을 한 step에서 합산합니다.
9. **URDF/MJCF and joint-trajectory adapter**: URDF 또는 MuJoCo XML/MJCF의
   kinematic tree를 `ArticulatedLink`로 변환하고, joint position trajectory에서 link별
   pose와 velocity JSON을 생성합니다. URDF origin/rpy와 MJCF body pos/quat/euler,
   joint axis 및 offset joint anchor를 지원합니다.
10. **Collision-only geometry gradient routing**: 논문 Stage II와 같이 renderer와
    직접 geometry supervision에는 detached center/radius를 전달합니다. Image loss는
    predicted state와 contact dynamics를 거쳐서만 Gaussian geometry를 수정합니다.
11. **L1 + LoFTR correspondence loss**: frozen pretrained LoFTR로 예측/관측 영상의
    대응점을 찾고, 해당 위치의 정규화 RGB patch feature 차이를 L1 photometric loss에
    더합니다. Hard matching은 gradient graph 밖에서 수행하고, 선택된 rendered patch는
    graph에 남겨 pose, dynamics와 collision geometry까지 gradient가 전달됩니다.
12. **Paper spherical-Gaussian convention**: Stage-I 렌더 geometry를 isotropic
    scale과 identity rotation으로 고정해 저장하고, 하나의 Gaussian scale `s`를 renderer와
    collision이 공유합니다. Renderer에는 `s`, collision detector에는 논문의 `r=2s`를
    전달합니다. Legacy anisotropic PLY를 위한 mean/max 변환과 이전 `r=s` 호환 모드도
    제공합니다.

1–12단계를 함께 사용하는 권장 실행 예시는 다음과 같습니다.

```bash
conda run -n gaussian_splatting python \
   tools/compare_query_modes.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --output_root <comparison_output> \
  --variants floor_disk axis6 fib26 analytic \
  --prefit_initial_state \
  --prefit_velocity_frames 3 \
  --refine_geometry \
  --geometry_gradient_route collision_only \
  --gaussian_radius_convention paper_r2s \
  --gaussian_scale_reduction strict \
  --image_only_objective \
  --gaussian_rgb_loss_weight 1.0 \
  --gaussian_render_loss l1_loftr \
  --gaussian_render_loftr_weight 0.1 \
  --loftr_pretrained outdoor \
  --gaussian_rgb_dir <rgb_dir> \
  --gaussian_mask_dir <mask_dir> \
  --pairwise_contact_model dual_cone \
  --pairwise_dual_cone_directions 4 \
  --pairwise_friction_mode learned \
  --pairwise_friction_coefficient 0.1 \
  --pairwise_mass_mode learned \
  --mass 1.0 \
  --pairwise_inertia_mode learned \
  --pairwise_inertia_diag 1,1,1 \
  --actions_json <episode_root/actions/trajectory.json> \
  --pairwise_body_b_ply <robot_or_environment_gaussians.ply> \
  --pairwise_body_b_trajectory_json <body_b_trajectory.json>
```

RGB loss와 initial-state prefit은 CUDA Gaussian rasterizer가 필요합니다. 첫 N개 prefit
프레임은 접촉 전 구간이어야 합니다. Mass와 K는 trajectory scale에 따라 서로 보상할
수 있으므로 실제 질량에 가까운 초기값과 `--pairwise_mass_l2_weight`,
`--pairwise_inertia_l2_weight` prior를 사용하는 것이 좋습니다.

학습되는 주요 물성치는 `fit_summary.json`의 `learned_stiffness`,
`learned_damping`, `pairwise_friction_coefficient`, `learned_mass`,
`learned_inertia_diag`에서 확인할 수 있습니다. 전체 config, 입력 manifest, Git commit과
결과 hash는 `experiment_bundle.json`에 함께 저장됩니다.

논문 규약에서는 PLY의 isotropic Gaussian scale을 `s`, collision 반경을 `r=2s`로
해석합니다. 새 Stage-I 결과는 세 `scale_*` 채널이 동일하고 `rot_*`이 identity로
저장되며, PLY 옆의 `point_cloud.spherical.json`에 scale/radius 규약이 기록되므로
`--gaussian_scale_reduction strict` 사용을 권장합니다. 기존 anisotropic
3DGS PLY는 `mean` 또는 보수적인 `max`로 하나의 `s`로 변환할 수 있습니다. 이전 코드의
`r=s` 결과를 그대로 재실행할 때만 다음 호환 옵션을 사용합니다.

```bash
--gaussian_radius_convention legacy_r_equals_s \
--gaussian_scale_reduction mean
```

`--radius_scale`은 위 convention을 적용한 뒤 사용하는 추가 실험 배율입니다. 예를 들어
논문 모드의 실제 collision 반경은 `2 * s * radius_scale`입니다. Stage-II geometry
refinement는 collision radius를 갱신하지만 renderer에는 자동으로 절반인 Gaussian
scale을 전달하므로 두 경로가 동일한 spherical geometry를 유지합니다.

Action 파일은 각 transition에 적용할 world-frame wrench를 저장합니다. 다음 형식을
지원합니다.

```json
{
  "actions": [
    {
      "frame_index": 0,
      "force_world": [1.0, 0.0, 0.0],
      "torque_world": [0.0, 0.0, 0.2]
    },
    {
      "frame_index": 1,
      "generalized_force": [0.0, 1.0, 0.0, 0.0, 0.0, 0.1]
    }
  ]
}
```

`--actions_json`을 생략하면 `<episode_root>/actions/trajectory.json`을 자동으로 사용하고,
파일이 없으면 모든 action을 0으로 둡니다. `--action_force_scale`과
`--action_torque_scale`로 단위 변환을 적용할 수 있습니다. 적용된 wrench는
`predicted_trajectory.json`, action source와 nonzero step 수는 `fit_summary.json`,
원본 action 파일 hash는 `experiment_bundle.json`에 기록됩니다.

움직이는 environment/robot collider의 trajectory는 다음처럼 지정합니다.

```json
{
  "states": [
    {
      "position": [0.0, 0.0, 0.0],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "linear_velocity": [0.0, 0.0, 0.1],
      "angular_velocity": [0.0, 0.2, 0.0]
    }
  ]
}
```

velocity 필드는 생략할 수 있으며 이때 연속 pose의 finite difference로 계산됩니다.
Body B는 prescribed kinematic motion을 그대로 따르고 접촉 반력으로 움직이지 않지만,
그 선속도와 각속도는 각 contact point의 `v + ω×r`에 포함됩니다. 사용된 body-B pose는
`predicted_trajectory.json`, trajectory 정보와 입력 hash는 각각 `fit_summary.json`과
`experiment_bundle.json`에 저장됩니다.

Articulated link는 `ArticulatedLink`와 `forward_kinematics`로 구성합니다.

```python
from stage2.articulated_kinematics import (
    ArticulatedLink,
    forward_kinematics,
    link_velocities_from_poses,
)

links = [
    ArticulatedLink("palm", -1, "fixed", (0, 0, 1),
                    (0.0, 0.0, 0.0), (1, 0, 0, 0)),
    ArticulatedLink("finger", 0, "revolute", (0, 1, 0),
                    (0.05, 0.0, 0.0), (1, 0, 0, 0)),
]
link_p, link_q = forward_kinematics(
    links,
    joint_positions,  # [T, num_links]
    base_position=base_position,
    base_quaternion_wxyz=base_quaternion,
)
link_v, link_w = link_velocities_from_poses(link_p, link_q, dt)
```

각 link의 Gaussian body와 `RigidBodyState`를
`MultiBodyGaussianImpedanceDynamics`에 전달하고, config에는 dynamic object와
kinematic link를 구분해 지정합니다.

```python
MultiBodyImpedanceDynamicsConfig(
    dynamic_flags=(True, False, False),
    kinematic_flags=(False, True, True),
)
```

Kinematic link는 contact impulse로 상태가 바뀌지 않지만 link의 `v+ω×r`은 dual-cone
contact Jacobian에 포함됩니다. FK와 collision/dynamics가 모두 torch 연산이므로 loss에서
joint trajectory까지 gradient가 전달됩니다.

URDF/MJCF 모델과 기록된 joint trajectory는 다음 명령으로 Stage-2 link trajectory로
변환합니다.

```bash
conda run -n gaussian_splatting python \
   tools/prepare_articulated_trajectory.py \
  --model <robot.urdf_or_mjcf.xml> \
  --joint_trajectory <joint_trajectory.json> \
  --output_dir <articulated_trajectory_output>
```

Joint trajectory는 아래 두 형식 중 하나를 사용할 수 있습니다.

```json
{
  "joint_names": ["finger_joint_0", "finger_joint_1"],
  "states": [
    {
      "time": 0.0,
      "joint_positions": [0.0, 0.2],
      "base_position": [0.0, 0.0, 0.0],
      "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  ]
}
```

또는 `joint_positions`, `times`, `base_positions`, `base_quaternions_wxyz` 배열을
최상위 필드로 줄 수 있습니다. 출력 폴더에는 link별 trajectory JSON과 다음 정보를 담은
`articulated_trajectory_manifest.json`이 생성됩니다.

- 원본 URDF/MJCF 및 joint trajectory 경로
- link index/name, joint name/type
- 추론하거나 지정한 `dt`
- 각 link의 Stage-2 trajectory 경로

생성된 link trajectory는 7번의 `--pairwise_body_b_trajectory_json`으로 단일 link
실험에 바로 사용할 수 있고, 여러 link는 8번의 `MultiBodyGaussianImpedanceDynamics`
입력 state로 함께 사용할 수 있습니다.

` ` 폴더명에는 기존 저장소의 오타(`initiailization`)가 그대로 남아 있습니다.

## 실험 결과물 위치

MuJoCo 데이터셋과 Stage 2 결과물은 저장소 루트의 `output/` 아래에 모읍니다 (git 미추적).

```text
output/actual_cola_can_*_data/     MuJoCo GT 에피소드 (rgb/, masks/, state/trajectory.json)
output/actual_cola_can_*_output/   Stage 2 fit 결과, 렌더, 비교 GIF
```

### 캔–바닥 MuJoCo 데이터셋 한 번에 생성

Stage 1 정적 멀티뷰와 Stage 2 낙하/충돌 영상을 같은 물리 크기 규약으로 생성하려면:

```bash
conda run -n cg_wm python tools/generate_mujoco_can_floor_dataset.py \
  --output_root output/can_floor_dataset \
  --train_views 48 --test_views 12 \
  --train_episodes 8 --test_episodes 2 \
  --frames_per_episode 90 --fps 30 \
  --width 640 --height 480 --mujoco_gl egl
```

`stage1/can_floor/images`와 `transforms_*.json`은 NeRF/Blender 형식의 calibrated
multiview 입력이다. `masks`는 캔 binary mask이고 `instance_masks`는
`0=background, 1=cola_can, 2=floor` label map이다. Stage 2의 각 episode에는
`rgb/`, `masks/`, `state/trajectory.json`, `actions/trajectory.json`, `rollout.gif`가
생성된다. 같은 명령과 seed로 재생성할 수 있으며 기존 경로를 덮어쓸 때만
`--overwrite`를 추가한다.

## 현재 캔–바닥 Pipeline: 입력, 출력, 데이터 흐름

아래는 현재 실제 검증에 사용하는 end-to-end 경로이다. Stage 1은 캔과 바닥의
appearance Gaussian을 각각 학습하고, Stage 2는 MuJoCo episode의 상태/RGB를 supervision으로
물리 trajectory를 맞춘다. 마지막 영상은 SIBR Viewer나 MuJoCo renderer를 사용하지 않고
두 Gaussian PLY를 하나의 CUDA Gaussian rasterizer 호출로 렌더링한 뒤 Pillow로 GIF를 만든다.

```text
MuJoCo scene/asset
        |
        +-- static calibrated multiview RGB + masks + transforms_*.json
        |                         |
        |                         +-- Stage 1 can training --> can point_cloud.ply
        |                         +-- Stage 1 floor training -> floor point_cloud.ply
        |                                                        |
        |                                      planar calibration/resampling
        |                                                        |
        |                                      render_floor_grid/point_cloud.ply
        |
        +-- dynamic fall-and-rebound episode
              rgb/ + masks/ + state/trajectory.json + actions/trajectory.json
                                      |
                                      v
                           Stage 2 physics/RGB fitting
                                      |
                         predicted_trajectory.json
                                      |
                 can Gaussian rigid transform per frame
                                      + static floor Gaussian
                                      |
                         one Gaussian rasterizer call
                                      |
                         gaussian_rgb/*.png
                                      |
                              Pillow GIF encoder
                                      |
                    stage2_gaussian_trajectory.gif
```

### 단계별 입출력

| 단계 | 주요 입력 | 주요 출력 |
|---|---|---|
| MuJoCo dataset | can asset, floor, camera/physics 설정 | Stage 1 multiview와 Stage 2 episode |
| Stage 1 can | `images/`, can binary `masks/`, `transforms_{train,test}.json` | can `point_cloud.ply` |
| Stage 1 floor | `images/`, floor binary `masks/`, `transforms_{train,test}.json` | raw floor `point_cloud.ply` |
| Floor render calibration | raw floor PLY, known plane extent/height | regular planar floor Gaussian PLY |
| Stage 2 fit | episode state/RGB/masks, can PLY, initial physics values | `predicted_trajectory.json`, `fit_summary.json`, reproducibility bundle |
| Gaussian-only render | predicted trajectory, can PLY, calibrated floor PLY, camera | per-frame PNG, GIF, renderer manifest |

현재 검증 artifact의 구체적인 경로는 다음과 같다.

```text
# Stage 1 inputs
output/can_stage1_check_data/cola_can/
output/floor_stage1_data/floor/

# Stage 1 outputs
output/can_stage1_check_model_fixed/point_cloud/iteration_3000/point_cloud.ply
output/floor_stage1_model/point_cloud/iteration_3000/point_cloud.ply

# Stage 2 input episode
output/can_floor_dataset/contactwm/stage2/fall_and_rebound/train/cola_can/episode_000/

# Stage 2 outputs
output/can_floor_stage2_end_to_end/predicted_trajectory.json
output/can_floor_stage2_end_to_end/fit_summary.json
output/can_floor_stage2_end_to_end/resolved_config.json
output/can_floor_stage2_end_to_end/experiment_bundle.json

# Gaussian-only rendering outputs
output/floor_stage1_model/render_floor_grid/point_cloud.ply
output/can_floor_stage2_gaussian_only_final/gaussian_rgb/*.png
output/can_floor_stage2_gaussian_only_final/stage2_gaussian_trajectory.gif
output/can_floor_stage2_gaussian_only_final/stage2_gaussian_trajectory_manifest.json
```

### Stage 1 학습

캔과 바닥은 서로 다른 binary alpha mask로 학습한다. `--alpha_subject object`는 캔,
`--alpha_subject floor`는 바닥 mask를 만든다. 현재 검증 모델과 동일한 3,000-iteration
학습의 핵심 명령은 다음과 같다.

```bash
conda run -n gaussian_splatting python stage1/train.py \
  -s output/can_stage1_check_data/cola_can \
  -m output/can_stage1_check_model_fixed \
  --masks_dir output/can_stage1_check_data/cola_can/masks \
  --iterations 3000 --eval --disable_viewer

conda run -n gaussian_splatting python stage1/train.py \
  -s output/floor_stage1_data/floor \
  -m output/floor_stage1_model \
  --masks_dir output/floor_stage1_data/floor/masks \
  --iterations 3000 --eval --disable_viewer
```

Texture가 거의 없는 평면은 RGB multiview만으로 depth가 유일하게 정해지지 않는다. 따라서
raw floor checkpoint는 appearance 검증에는 사용할 수 있지만 동적 scene 렌더/접촉에는
깊이 방향 blob이 생길 수 있다. 현재 pipeline은 알려진 MuJoCo 바닥 평면 prior를 적용해
중심을 규칙적인 격자로 투영하고, 학습된 floor SH 색상 통계는 유지한다.

```bash
conda run -n gaussian_splatting python tools/calibrate_floor_gaussian_ply.py \
  --input_ply output/floor_stage1_model/point_cloud/iteration_3000/point_cloud.ply \
  --output_ply output/floor_stage1_model/render_floor_grid/point_cloud.ply \
  --xy_extent 1.2 --regular_grid_size 49 \
  --max_primitives 4096 --gaussian_scale 0.04 --surface_z 0.04
```

### Stage 2 출력의 의미

`run_stage2_mujoco_stage1_fit.py`는 episode의 `state/trajectory.json`을 target으로 사용하고,
선택적으로 `rgb/`와 `masks/`의 Gaussian image loss도 함께 계산한다. 주요 출력은 다음과 같다.

- `predicted_trajectory.json`: 프레임별 predicted position, quaternion, contact gate
- `fit_summary.json`: trajectory/RGB loss, 접촉 frame, 학습된 물리 파라미터
- `resolved_config.json`: 실제로 적용된 전체 CLI/config
- `experiment_bundle.json`: 입력 hash, Git 상태, 결과 hash를 포함한 재현성 정보
- `stage2_fit_follow_view.gif`: 물리 진단용 영상이며 Gaussian-only 결과를 의미하지 않음

Experimental baseline에서는 analytic plane collider를 계속 사용할 수 있다. 반면
paper-compatible manifest는 캔과 바닥 모두 Stage1 Gaussian을 렌더와 collision에 공유한다.
따라서 paper-compatible 결과에서는 바닥 mesh/plane을 물리용으로 따로 사용하지 않는다.

### SIBR 없이 Gaussian-only PNG/GIF 생성

다음 명령은 trajectory의 pose를 프레임마다 캔 Gaussian 전체에 rigid transform으로 적용한다.
정적인 바닥 Gaussian을 결합한 뒤, 두 자산을 동일한 `gs_render` 호출로 depth-aware rasterize한다.
MuJoCo RGB나 mesh pixel을 합성하지 않는다.

```bash
conda run -n gaussian_splatting python tools/render_stage2_gaussian_trajectory.py \
  --stage1_ply \
    output/can_stage1_check_model_fixed/point_cloud/iteration_3000/point_cloud.ply \
  --floor_stage1_ply \
    output/floor_stage1_model/render_floor_grid/point_cloud.ply \
  --trajectory \
    output/can_floor_stage2_end_to_end/predicted_trajectory.json \
  --output_dir output/can_floor_stage2_gaussian_only_final \
  --frame_stride 2 --image_width 640 --image_height 480 \
  --cam_distance 1.12 --cam_height 0.66 --cam_fovy_deg 40 \
  --white_background --recenter_asset --opacity_threshold 0.05 --fps 15
```

렌더러는 먼저 `gaussian_rgb/NNNNNN.png`를 저장하고 Pillow의 multi-frame GIF encoder로
`stage2_gaussian_trajectory.gif`를 만든다. SIBR Viewer는 필요하지 않지만 현재 rasterizer가
CUDA extension이므로 NVIDIA GPU/CUDA 환경은 필요하다. 출력 manifest의
`renderer=single_call_gaussian_splatting`과 `mujoco_pixels_used=false`로 사용 경로를 확인할
수 있다.

## 구현 범위

### Stage 1: Gaussian Initialization

- multi-view RGB/mask 기반 3D Gaussian Splatting 학습
- spherical Gaussian 제약
- foreground/object-aware Gaussian field
- object mask supervision
- optional SAM feature supervision
- visual hull seed initialization
- object id 저장/복구 및 physics export
- reproducible training preset/schedule runner

주요 파일:

```text
 stage1/train.py
 stage1/render.py
 stage1/metrics.py
 stage1/export_physics_scene.py
 stage1_training_presets.json
 tools/run_stage1_training_schedule.py
```

### Stage 2: Differentiable Collision And Dynamics

- spherical Gaussian union SDF collision detection
- aggregate contact patch selection
- rigid Gaussian body pose transform
- broad phase collision filtering
- pairwise/multibody contact dynamics
- differentiable impedance contact
- Coulomb-style friction cone projection
- learnable initial velocity, physics parameter, and geometry refinement
- MuJoCo rollout comparison
- mask/RGB/Gaussian renderer based fitting hooks
- automated multi-variant evaluation

주요 파일:

```text
 stage2/differentiable_collision_detection.py
 stage2/differentiable_contact_graph.py
 stage2/differentiable_complementarity_free_contact_dynamics.py
 stage2/renderable_gaussian_asset.py
 stage2/differentiable_gaussian_render_loss.py
 tools/generate_mujoco_multi_dice_rollout.py
 tools/run_stage2_multi_dice_rollout_comparison.py
 tools/evaluate_multi_dice_stage2_variants.py
```

## 빠른 실행

### Stage 1 Schedule Dry Run

실제 학습을 돌리기 전에 어떤 명령이 실행되는지 확인합니다.

```bash
python  tools/run_stage1_training_schedule.py \
  --preset dice_smoke \
  --dry_run
```

ContactGaussian-WM 스타일 preset은 SAM feature가 필요합니다.

```bash
python  tools/run_stage1_training_schedule.py \
  --preset contactwm_smoke \
  --dry_run \
  --print_json
```

### Stage 1 Smoke Schedule

```bash
python  tools/run_stage1_training_schedule.py \
  --preset dice_smoke
```

이 runner는 `--python`과 `--mujoco_python`에 shell command가 아니라 Python executable 경로를 받습니다. Conda 환경을 분리해서 쓰는 경우에는 dataset 생성, train, render 단계를 따로 실행하거나 각 환경의 Python 실행 파일 경로를 넘겨야 합니다.

### Multi-Dice MuJoCo Dataset

```bash
conda run -n mujoco python  tools/generate_mujoco_multi_dice_rollout.py \
  --output_root actual_multi_dice_mujoco \
  --scene_name demo_codex \
  --num_dice 3 \
  --frames 120 \
  --width 256 \
  --height 256
```

### Stage 2 Rollout Comparison

```bash
conda run -n gaussian_splatting python  tools/run_stage2_multi_dice_rollout_comparison.py \
  --trajectory actual_multi_dice_mujoco/demo_codex/trajectory.json \
  --stage1_ply  output/<stage1_model>/point_cloud/iteration_30000/point_cloud.ply \
  --output_dir actual_stage2_comparison/demo_codex \
  --dynamics_backend stage2_impedance \
  --fit_iters 40 \
  --fit_physics_iters 40 \
  --fit_geometry_radii \
  --fit_geometry_centers \
  --stage2_static_friction 0.4
```

### Stage 2 Variant Evaluation

```bash
conda run -n gaussian_splatting python  tools/evaluate_multi_dice_stage2_variants.py \
  --trajectory actual_multi_dice_mujoco/demo_codex/trajectory.json \
  --stage1_ply  output/<stage1_model>/point_cloud/iteration_30000/point_cloud.ply \
  --output_root actual_stage2_eval/demo_codex \
  --variants impulse stage2 velocity_fit physics_fit \
  --max_frames 100
```

결과는 variant별 output directory와 함께 아래 파일로 저장됩니다.

```text
multi_dice_stage2_variant_report.json
multi_dice_stage2_variant_report.csv
```

## Gaussian Renderer Loss

Stage 2 pose trajectory를 Gaussian renderer로 렌더링하고 RGB supervision loss를 걸 수 있는 hook이 구현되어 있습니다.

주의: `diff-gaussian-rasterization` 기반 renderer는 CUDA가 필요합니다. CPU-only 환경에서는 renderer backward까지 검증할 수 없습니다.

Single-body pairwise fitting에서도 trajectory, geometry, Gaussian RGB supervision을 하나의
objective로 사용할 수 있습니다.

```bash
conda run -n gaussian_splatting python \
   tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --dynamics pairwise_impedance \
  --geometry_loss_weight 0.5 \
  --gaussian_rgb_loss_weight 0.1 \
  --gaussian_rgb_dir <episode_root/rgb> \
  --device cuda
```

`pairwise_impedance`의 기본 contact model은 논문 III-D-2와 Appendix C의 polyhedral
dual-cone 식입니다. 각 contact Jacobian을 여러 접선 방향에 대해
`J̃ = Jⁿ - μJᵈ`로 확장하고, 각 facet에
`SoftPlus(-K(hJ̃b+φ̃)-D(J̃b))`를 적용합니다. 따라서 normal force와 Coulomb friction이
별도 projection 없이 같은 closed-form 항에서 계산됩니다.

```bash
  --pairwise_contact_model dual_cone \
  --pairwise_dual_cone_directions 4 \
  --pairwise_friction_mode learned \
  --pairwise_friction_coefficient 0.1
```

`--pairwise_friction_mode`은 `off`, `fixed`, `learned`를 지원하며, learned 모드에서는
양수로 제약된 μ가 K·D와 함께 최적화됩니다. 이전 구현과 ablation하려면
`--pairwise_contact_model projected`를 사용합니다.

Geometry loss는 예측 pose와 GT pose로 변환한 동일 Stage-I Gaussian center들의 MSE입니다.
RGB loss는 rollout이 예측한 position/quaternion을 Gaussian rasterizer에 직접 전달하므로
이미지 loss의 gradient가 물성치와 초기 상태까지 이어집니다. 렌더 비용은
`--gaussian_render_stride`, geometry 비용은 `--geometry_loss_stride`로 조절할 수 있습니다.

`--refine_geometry`를 추가하면 Stage-II에서 각 Gaussian의 object-local center offset과
log-radius offset을 물리 파라미터와 함께 학습합니다. 기본
`--geometry_gradient_route collision_only`에서는 collision SDF가 live geometry tensor를
사용하고 renderer와 posed-geometry supervision은 detached tensor를 사용합니다. 따라서
image loss는 predicted pose와 dynamics/collision을 통과한 경로로만 geometry를 갱신합니다.
직접 photometric geometry gradient를 ablation하려면
`--geometry_gradient_route collision_and_render`를 사용합니다. Renderer appearance는
collision filtering 후 남은 Gaussian의 원본 SH/opacity를 source index로 정확히
대응시킵니다. Offset L2와 clamp 범위는
`--geometry_center_l2_weight`, `--geometry_radius_l2_weight`,
`--geometry_max_center_offset`, `--geometry_max_log_radius_offset`으로 제어합니다.
최종 geometry는 `refined_geometry.json`에 저장됩니다.

논문식 이미지 중심 학습에서는 `--image_only_objective`를 사용합니다. 이 모드는
position/orientation/posed-geometry supervision weight를 0으로 만들고 Gaussian RGB
loss와 parameter prior만으로 best state를 선택합니다. `state/trajectory.json`은 필수가
아니며, `--evaluation_trajectory`가 있더라도 학습 graph에는 넣지 않고 종료 후 metric
계산에만 사용합니다. GT가 없을 때는 rollout 시작점을 정하기 위해
`--initial_state_json` 또는 `--prefit_initial_state` 중 하나가 필요합니다.

```bash
conda run -n gaussian_splatting python \
   tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --dynamics pairwise_impedance \
  --refine_geometry \
  --geometry_gradient_route collision_only \
  --image_only_objective \
  --initial_state_json <initial_state_estimate.json> \
  --gaussian_rgb_loss_weight 1.0 \
  --gaussian_render_loss l1_loftr \
  --gaussian_render_loftr_weight 0.1 \
  --loftr_pretrained outdoor \
  --device cuda
```

GT가 없는 실행에서 `fit_summary.json`은
`ground_truth_trajectory_used_for_training=false`, position loss/RMSE는 `null`로 기록한다.
GT를 평가용으로만 추가하려면 `--evaluation_trajectory <trajectory.json>`을 사용한다.
학습 입력 RGB/mask/camera/time과 평가 pose는 각각
`stage2/video_observations.py`의 `VideoObservations`, `EvaluationTrajectory`로 분리되어 있다.

### 객체 이름에 독립적인 scene manifest

`stage2/scene_manifest.py`는 `cola_can`, `floor`, `box` 같은 클래스명이나 shape preset을
사용하지 않는다. 각 항목은 사용자가 정한 ID, rigid-body 역할, render asset, collision
표현으로만 정의된다. 전체 예제는 `configs/scene_manifest.example.json`에 있다.

```json
{
  "version": 1,
  "scene_id": "rigid_contact_000",
  "bodies": [
    {
      "id": "object_00",
      "role": "dynamic",
      "render": {"gaussian_ply": "models/object_00.ply"},
      "collision": {"type": "gaussian_union"}
    }
  ],
  "environment": [
    {
      "id": "surface_00",
      "role": "static",
      "collision": {
        "type": "plane",
        "normal": [0, 0, 1],
        "height": 0.0
      }
    }
  ],
  "observations": {
    "rgb_dir": "observations/rgb",
    "instance_mask_dir": "observations/masks",
    "fps": 30
  },
  "contact_pairs": [
    {"body_a": "object_00", "body_b": "surface_00", "model": "dual_cone"}
  ]
}
```

모든 상대경로는 manifest 파일 위치를 기준으로 해석한다. 지원 역할은 `dynamic`,
`kinematic`, `static`, collision 표현은 `gaussian_union`, `plane`, `query_points`, `none`이다.
Kinematic body에는 trajectory가 필요하고, 관측에는 양의 FPS 또는 strictly-increasing
timestamp 배열이 필요하다.

```bash
conda run -n gaussian_splatting python tools/validate_scene_manifest.py \
  configs/scene_manifest.example.json
```

Validator는 ID 중복, 존재하지 않는 contact body, self/duplicate pair, 잘못된 plane normal,
누락된 Gaussian/trajectory/RGB/camera 경로와 frame time 계약을 검사한다.

검증된 manifest는 adapter를 통해 현재 single-body image-only Stage2에 바로 전달할 수 있다.

```bash
conda run -n gaussian_splatting python tools/run_contactgaussian_pipeline.py \
  --manifest configs/scene_manifest.example.json \
  --output_dir output/manifest_stage2_run
```

실행 전 생성될 명령만 확인하려면 `--dry_run`을 추가한다. 짧은 검증에는
`--fit_iters`, `--max_frames`, `--device`, `--image_loss` override를 사용할 수 있다.
Adapter는 원본 manifest를 검증하고 현재 Stage2가 읽을 compatibility manifest와
`compiled_manifest_run.json`을 생성한다. 원본 manifest 경로와 SHA-256도
`experiment_bundle.json`에 보존한다.

현재 adapter 지원 범위는 다음과 같다.

- 정확히 하나의 `dynamic` spherical-Gaussian body
- 동일 PLY를 공유하는 render/collision representation
- `normal=[0,0,1]`, `height=0`인 하나의 static plane
- `image_only` supervision
- initial-state JSON 또는 image prefit

여러 dynamic body, 움직이는 kinematic contact target, 임의 방향/높이 plane은 잘못된
single-body 변환을 하지 않고 명시적으로 거부한다. Gaussian-union multi-body 장면은
아래 native manifest runner가 `bodies[]`와 `contact_pairs[]`를 직접 실행한다.

Gaussian-union끼리 충돌하는 장면은 native N-body rollout 경로로 직접 실행할 수 있다.

```bash
python tools/run_native_multibody_manifest.py \
  --manifest configs/your_multibody_scene.json \
  --output output/native_multibody/trajectory.json \
  --steps 60 --device cpu
```

이 경로는 선언된 `contact_pairs[]`만 contact graph edge로 사용하며, GT trajectory를 읽지
않고 모든 body의 pose/velocity를 ID별로 출력한다. native dynamics는 dynamic/static
`gaussian_union` body와 static analytic `plane`을 한 장면에서 함께 지원한다. Plane body도
`render.gaussian_ply`를 가지면 화면에는 Gaussian으로 렌더링되지만, 접촉 거리·법선·힘은
manifest의 `normal`과 `height`로 계산된다. 따라서 바닥 Gaussian의 불규칙한 sphere proxy가
물리 안정성을 해치지 않는다. kinematic trajectory playback은 아직 지원하지 않는다.

실제 영상만 사용하여 여러 body의 초기 pose/linear velocity/angular velocity, body별
mass/inertia와 contact stiffness/damping/friction을 학습하려면 `--fit_iters`를 지정한다.
Gaussian rasterizer 때문에
이 모드는 CUDA가 필요하다.

```bash
conda run -n gaussian_splatting python tools/run_native_multibody_manifest.py \
  --manifest configs/your_multibody_scene.json \
  --output output/native_multibody/image_only_fit.json \
  --fit_iters 500 --lr 0.01 --device cuda \
  --render_stride 2 --render_max_frames 30 \
  --image_loss l1_ssim \
  --mass_inertia_lr 0.001 --mass_l2 0.0001 --inertia_l2 0.0001
```

모든 renderable body의 Stage1 PLY를 `MultiBodyStage2GaussianRenderLoss`가 하나의 Gaussian
rasterizer scene으로 합성하고 RGB/mask loss를 dynamics까지 역전파한다. 출력에는 loss history,
학습된 initial state, `learned_contact_pairs[]`, sampled trajectory가 포함되며
`ground_truth_trajectory_used_for_training=false`가 기록된다. stiffness/damping/friction은
manifest에 선언된 contact pair 순서대로 각각 초기화되고 독립적으로 학습된다. 기존 코드에서
scalar contact parameter를 전달하는 경우에는 이전처럼 모든 edge가 그 값을 공유한다.

Body 물성 초기값은 객체 이름이나 shape preset이 아니라 각 manifest body의 `physics`에서 읽는다.

```json
"physics": {
  "mass": {"initial": 0.35},
  "inertia": {"initial_diagonal": [0.0018, 0.0018, 0.0007]}
}
```

`inertia.initial_diagonal`을 생략하면 dynamic Gaussian-union의 inertia는 고정 `[1,1,1]`이
아니라 Stage1 collision Gaussian으로부터 자동 초기화된다. 각 Gaussian을 반지름 세제곱에
비례한 질량을 가진 solid sphere로 보고, Gaussian 전체의 volume-weighted center of mass에
대한 parallel-axis 항까지 합산한다. 따라서 특정 캔/상자 preset 없이도 객체 크기와 manifest
mass에 맞는 `kg·m²` 단위의 초기 관성을 얻는다. Manifest에 관성을 명시하면 그 값이 우선한다.

Mass는 softplus로 양수를 보장한다. Body-frame principal inertia는 세 양수 성분
`(a,b,c)`에서 `(Ix,Iy,Iz)=(a+b,a+c,b+c)`로 만들기 때문에 학습 중에도 양수성과 세
triangle inequality를 만족한다. 이 값은 contact Jacobian의 inverse-mass block, Delassus
행렬, linear/angular velocity update에 공통으로 사용된다. Static body는 optimizer의 학습
대상이 아니며 inverse-mass block도 0이다. 결과 JSON의 `learned_body_physics.<body_id>`에
최종 mass와 `inertia_diagonal`이 기록된다.

초기값에서 지나치게 멀어지는 것을 막는 prior와 별도 learning rate는
`--mass_l2`, `--inertia_l2`, `--mass_inertia_lr`로 조절한다. 비교 실험에서 고정하려면
`--freeze_mass_inertia`를 사용한다.

긴 낙하 영상은 `--render_max_frames 0`으로 끝까지 읽고, GPU rasterization 메모리는
`--temporal_window_frames`로 제한할 수 있다. 학습 iteration마다 선택된 sampled-frame 배열의
연속 window를 cyclic하게 이동하며, `--temporal_window_step`이 window 시작 간격을 정한다.
Dynamics는 각 window의 실제 frame 번호까지 처음부터 적분하므로 후반 충돌 window에서도 초기
상태와 물성치로 gradient가 이어진다. 최종 evaluation과 GIF는 window가 아니라 전체 선택
프레임을 사용한다. 각 iteration의 실제 학습 프레임은 JSON `loss_history[].training_frame_indices`에
기록된다.

```bash
conda run -n cg_wm python tools/run_native_multibody_manifest.py \
  --manifest configs/scene_manifest.sggs_prefit_test.json \
  --output output/sggs_stage2_mass_inertia/full_sequence.json \
  --device cuda --fit_iters 300 --render_stride 3 --render_max_frames 0 \
  --temporal_window_frames 8 --temporal_window_step 4 \
  --render_output_dir output/sggs_stage2_mass_inertia/full_sequence_render
```

물리 파라미터와 초기 상태를 동시에 맞추기 전에 native manifest 경로에서도 image-only
초기 상태 prefit을 실행할 수 있다. 첫 프레임으로 모든 dynamic body의 position/quaternion을
먼저 맞추고, 처음 N개 접촉 전 프레임에는 constant-velocity 모델을 사용해 linear/angular
velocity를 맞춘다. static body는 고정되며 GT trajectory는 읽지 않는다. `--fit_iters 0`이면
prefit만 수행한다.

```bash
conda run -n gaussian_splatting python tools/run_native_multibody_manifest.py \
  --manifest configs/scene_manifest.sggs_prefit_test.json \
  --output output/sggs_stage2_prefit/prefit_only.json --device cuda \
  --fit_iters 0 --render_max_frames 6 --render_width 160 --render_height 120 \
  --prefit_initial_state --prefit_pose_iters 100 \
  --prefit_velocity_iters 100 --prefit_velocity_frames 6 --prefit_lr 0.01
```

실제 SG-GS 캔과 낙하 영상에서 pose loss는 `0.00448 → 0.00208`, velocity loss는
`0.00454 → 0.00269`로 감소했다. 전체 history와 추정 state는 출력 JSON의
`initial_state_prefit`에 저장된다.

### Stage2 pipeline modes

Stage2 실행 경로는 논문 재현과 추가 기능이 섞이지 않도록 세 contract로 분리한다.

- `paper_compatible`: 알려진 manifest initial state와 action을 입력으로 사용하는 논문 기준 경로
- `image_only`: 영상으로 initial state까지 추정하는 확장 경로
- `experimental`: temporal window와 renderer-direct geometry gradient 등의 ablation 경로

```bash
python tools/run_native_multibody_manifest.py \
  --pipeline_mode paper_compatible \
  --manifest configs/your_multibody_scene.json \
  --output output/paper_compatible/result.json --device cuda --fit_iters 100
```

`paper_compatible`에서는 initial-state prefit, temporal window,
`collision_and_render` geometry gradient를 허용하지 않는다. 실행 결과에는
`pipeline_mode`, `pipeline_contract`, `paper_compatibility`가 기록되어 어떤 가정으로 생성된
결과인지 확인할 수 있다. 현재 known-state/action과 fixed-penetration collision 경로까지
구현됐으며, dual-cone contact dynamics와 full-image L1+LoFTR supervision도
Gaussian–Gaussian과 Gaussian–plane에 통일했다.
교체·검증한다. 기존 명령에서 모드를 생략하면 사용 옵션에 따라
`image_only` 또는 `experimental`로 자동 분류되어 이전 실행과 호환된다.

Paper-compatible 경로의 모든 dynamic initial position/quaternion/linear velocity/angular
velocity는 `initialization.state_json`에서 읽은 뒤 optimizer에서 제외된다. Action은 manifest의
`actions`로 선언하며, 자유낙하는 `{"type":"zero_wrench"}`를 사용한다. 외력이 있는 장면은
body ID 기반 world-frame wrench 파일을 지정한다.

RGB를 기록하기 전에 simulator가 먼저 step되는 dataset에서는 episode 생성 전 초기값이 아니라
실제로 렌더된 첫 state를 사용해야 한다. `state_json`이 `{"states": [...]}` trajectory이면
`state_frame`으로 RGB와 같은 frame index를 명시한다.

```json
"initialization": {
  "state_json": "episode_000/state/trajectory.json",
  "state_frame": 0
}
```

Trajectory인데 `state_frame`이 없거나 중복/누락된 frame을 요청하면 실행을 거부한다. 결과의
`initial_state_learning.bodies`에 사용한 파일과 frame 번호가 기록된다.

MuJoCo dataset generator는 object manifest의 `physics_prior.mass_kg`를 collision geom의
명시적 `mass`로 적용한다. 이전의 고정 `density=1000`은 객체 크기에 따라 manifest prior와
다른 질량을 만들 수 있으므로 사용하지 않는다. 모델 생성 직후 요청 mass와
`model.body_mass`를 비교해 불일치하면 중단하며, 각 episode의 `mujoco_body_properties`에 실제
mass, inertia diagonal, free-joint damping, internal timestep을 기록한다. 기존 데이터는 보존하고
질량 수정 데이터는 `output/can_floor_mass_corrected_test`에 별도로 생성했다.

Body `physics.generalized_damping`은 MuJoCo free-joint damping과 같은 generalized coefficient로
해석한다. Translation에는 `f_d=-d*v`, rotation에는 `tau_d=-d*omega`가 적용된다. 작은 inertia에
explicit damping을 적용하면 불안정하므로 MuJoCo처럼 velocity update에서 implicit denominator
`1+h*d/m`과 body-frame `1+h*d/I`를 사용한다. Paper manifest는 mass-corrected episode에 기록된
값 `0.05`를 사용하며, 결과 `contact_dynamics_profile.generalized_damping`에 body 순서대로 저장된다.

```json
"actions": {"type": "wrench_sequence", "path": "actions.json"}
```

```json
{
  "coordinate_frame": "world",
  "frames": [
    {"frame": 0, "bodies": {
      "dynamic_0": {"force": [1, 0, 0], "torque": [0, 0, 0.1]}
    }}
  ]
}
```

`frame=t`의 wrench는 `state[t] → state[t+1]` transition에 적용된다. Force는 mass를 통한
linear acceleration, torque는 현재 quaternion으로 변환한 body-frame inertia와 Euler rigid-body
equation을 통한 angular acceleration에 연결된다. 결과 JSON의 `initial_state_learning`과
`actions`가 고정 상태 및 action 출처를 기록한다.

Paper-compatible collision profile은 Gaussian-union primitive distance에 LSE smooth-min을
적용한 뒤, 내부 거리를 sigmoid로 `-inside_penalty`에 수렴시킨다. 이 변환은
학습된 Gaussian-union SDF의 내부 안정화에만 사용한다. Analytic plane은 정확한
signed distance를 제공하므로 raw distance를 그대로 사용해 바닥 위의 exterior point가
가짜 penetration으로 변환되는 것을 막는다. 기본값은
`smooth_min_temperature=0.01`, `inside_penalty=0.02`, `inside_sharpness=50`이며 다음처럼
manifest에서 장면 단위로 조절할 수 있다.

```json
"training": {
  "paper_collision": {
    "smooth_min_temperature": 0.01,
    "inside_penalty": 0.02,
    "inside_sharpness": 50.0
  }
}
```

결과 JSON의 `collision_profile`은 fixed-penetration이 Gaussian-union에만 적용됨을 기록하고,
plane contact diagnostics의 `raw_signed_distance`와 `signed_distance`는 동일하다.

Paper-compatible contact dynamics는 각 patch의 normal과 tangent basis에서 dual friction-cone
facet `d_k = n - μt_k`를 만들고 해당 facet의 rigid contact Jacobian을 사용한다. Normal force와
friction을 별도로 projection하지 않고 하나의 closed-form facet force로 복원한다.

```text
A = I + h * (h*K + D) * J_dual * M_inv * J_dual^T
lambda = SoftPlus(solve(A, -K*phi - (h*K + D)*J_dual*b))
v_next = b + h * M_inv * J_dual^T * lambda
```

여기서 `b`는 gravity/action을 적용한 free velocity이고 `M_inv`는 학습되는 body mass와
body-frame inertia를 world frame으로 옮긴 generalized inverse mass이다. 결과 JSON의
`contact_dynamics_profile`과 contact diagnostics의 `dual_cone_faces`, `dual_cone_velocity`,
`dual_cone_lambda`로 실제 활성 경로를 확인할 수 있다.

Paper-compatible supervision은 CLI의 일반 `--image_loss`와 GT instance mask crop을 사용하지
않고 논문의 `L = L1 + LLoFTR`를 강제한다. 두 항의 가중치는 1이며 전체 RGB 프레임에서
계산한다. 따라서 예측 객체가 GT mask 밖으로 이동하거나 화면에서 사라지는 것도 L1 벌점을
받는다. LoFTR match 선택은 detach하지만 선택된 rendered RGB patch는 differentiable하므로
renderer, dynamics, physics parameter로 gradient가 이어진다. 결과 JSON의
`image_loss_config.requested_type`과 `type`, `full_image`, `gt_mask_used_for_loss`에서 요청값과
실제 paper 설정을 구분할 수 있다.

### Full multi-contact Jacobian dynamics

Native dynamics는 각 Gaussian contact patch와 dual-cone facet에 대해 rigid contact
Jacobian `J=[d, r_a×d, -d, -r_b×d]`를 명시적으로 구성한다. body–plane 접촉은 뒤쪽 body
block을 생략한다. body-frame diagonal inertia는 현재 quaternion으로 world frame에 옮기고,
정적 body의 inverse mass block은 0으로 둔다. 각 edge에서 다음 contact-space 행렬과
유효질량을 계산해 diagnostics로 제공한다.

```text
W = J M⁻¹ Jᵀ
m_eff = 1 / diag(W)
v_contact = J v_generalized
v_next = b + h M⁻¹ Jᵀ λ
```

Plane 접촉도 더 이상 모든 Gaussian을 하나의 점으로 평균내지 않고 signed distance가 가장
작은 `num_contact_patches`개의 surface point를 유지한다. 따라서 캔 rim처럼 중심에서 벗어난
동시 접촉이 서로 다른 torque를 만든다. `tests/test_contact_jacobian.py`는 off-center angular
Jacobian, 회전 관성을 포함한 Delassus 행렬, static mass block, gradient 전파를 검증한다.
SG-GS 캔의 75-step CPU rollout 결과는
`output/sggs_stage2_jacobian/rollout_k200_d5.json`에 있으며 모든 상태가 finite이고 최저
translation z는 `0.00338 m`였다.

### Observation frames and physics substeps

RGB frame 간격과 contact integration timestep을 분리한다. Manifest의
`observations.fps=30`은 명목상 영상 간격을, `simulation.physics_timestep=0.002`는
데이터를 생성한 fixed physics timestep을 나타낸다. Simulator가 frame당 step 수를
반올림해 녹화하면 명목 FPS와 실제 timestamp가 다를 수 있으므로 Runner는
physics timestep을 임의로 조정하지 않는다. `steps_per_frame`이 있으면 그 값을,
없으면 `round((1/fps)/physics_timestep)`을 사용한다. 현재 GT는 17개의
0.002초 step으로 녹화되어 실제 frame 간격이 0.034초이다.

```json
"observations": {"fps": 30},
"simulation": {"physics_timestep": 0.002, "steps_per_frame": 17}
```

Frame action wrench는 해당 frame의 모든 substep 동안 유지된다. 출력의
`contact_dynamics_profile`에 `nominal_observation_frame_dt`, `observation_frame_dt`,
`requested_physics_timestep`, `integration_dt`, `substeps_per_frame`을 기록해
명목 FPS와 실제 시간축을 둘 다 재현할 수 있다.

Contact pair에 `impedance_prior`를 주면 object preset 없이 body mass와 time constant로
학습 시작 stiffness/damping을 계산한다. Dynamic–static pair은 dynamic mass,
dynamic–dynamic pair은 reduced mass `m_eff=1/(1/m1+1/m2)`를 사용한다.

```text
K = m_eff / time_constant²
D = 2 * damping_ratio * m_eff / time_constant
```

```json
"impedance_prior": {"time_constant": 0.02, "damping_ratio": 1.0}
```

명시적 `stiffness` 또는 `damping`이 있으면 그 값이 prior보다 우선한다.

Collision Gaussian 수를 줄일 때 `primitive_selection="spatial_coverage"`를 선택하면
opacity 상위 점만 가져와 proxy가 한쪽으로 치우치는 문제를 막는다. 이 모드는
정규화한 asset-local 좌표에서 deterministic farthest-point sampling을 적용해
전체 형상을 균일하게 대표한다.

Plane contact와 같이 객체의 최외곽이 torque를 결정하는 실험은
`primitive_selection="support_surface"`를 사용할 수 있다. Fibonacci sphere 방향에서
각각 최대 projection을 가진 Gaussian을 먼저 선택하고 남은 budget은 spatial
coverage로 채운다. 이 방식은 객체 이름이나 cylinder/box preset 없이 끝단, edge,
rim과 같은 support surface를 보존한다.

Visual decoration이 물리 외곽을 부풀리는 asset은 `support_trim_quantile`로
좌표축 별 극단 outlier를 제외하고, `max_radius`로 visual Gaussian scale이
collision surface에 중복 더해지는 것을 제한할 수 있다. 두 값 모두 manifest의
collision metadata이며 object class preset은 사용하지 않는다.

`primitive_selection="geometry_feature_support"`는 experimental mode에서 Stage1 PLY의
`f_geo_*` feature를
Stage2 collision proxy 생성에 직접 사용한다. Primitive budget의 75%는 directional
support surface에 배정하고 나머지는 정규화된 local coordinate와 sigmoid geometry
feature의 joint embedding에서 farthest-point로 선택한다. `geometry_feature_weight`로
형상 coverage 대 feature diversity의 비율을 조절한다.

Paper-compatible mode에서는 `support_surface`, `geometry_feature_support`,
`support_trim_quantile`, `max_radius`를 허용하지 않는다. 논문의 Stage1 geometry
feature는 collision primitive 선택 feature가 아니라 spherical Gaussian의 center와 scale을
학습하는 supervision으로 사용되기 때문이다.

또한 paper-compatible dynamic Gaussian body는 `max_primitives` 및 `primitive_selection`을
허용하지 않는다. Renderer와 collision detector가 object filter 후의 동일한
Stage1 primitive center `c`와 isotropic scale `s`를 사용하며 collision radius는 논문과
같이 `r=2s`로 계산한다. Subsampling 기능은 experimental mode에만 남겨둔다.

Paper-compatible Stage2는 `--refine_geometry` 지정 여부와 관계없이 dynamic
Gaussian body의 center `c`와 isotropic scale `s`를 물리 파라미터
`(M, mu, K, D)`와 공동 최적화한다. 따라서 paper mode에서 `--freeze_mass_inertia`는
허용되지 않는다. 출력의 `geometry_refinement.requested`, `enabled`,
`enabled_by_pipeline_mode`로 CLI 요청과 mode contract에 의한 실제 활성화를 구분한다.

Full-image supervision에서 renderer의 빈 화면은 manifest camera의 `background_rgb`로
설정할 수 있다. 이 값을 GT 생성 환경의 clear color와 맞춰야 정적 배경 오차가
물리 gradient를 압도하지 않는다.

```json
"training": {
  "camera": {"background_rgb": [0.807843, 0.862745, 0.929412]}
}
```

Paper-compatible 테스트는 `render_floor_grid_wide/point_cloud.ply`의 81×81,
6,561 Gaussian grid를 렌더와 collision 양쪽에 사용한다. 각 primitive는 동일한 center와
scale에서 `r=2s` collision sphere가 된다. 캔 119,370개와 바닥 6,561개를 밀집 곱으로
만들지 않도록 `paper_collision.primitive_locality_margin`은 상대 물체의 확장 AABB와
교차하는 primitive만 narrow phase에서 평가한다. 이는 Stage1 asset을 subsampling하는 것이
아니며 전체 primitive와 parameter는 scene 및 renderer에 그대로 남는다.

### Stage1 asset canonical-frame alignment

Stage1 PLY 좌표의 원점과 물리 body의 local origin이 다르면 body의
`initialization.canonical_offset`에 Stage1 asset 좌표계에서 본 body origin을 미터 단위로
적는다. Runner는 이 값을 Gaussian center에서 빼서 renderer, collision primitive,
collision query point에 동일하게 적용한다. 따라서 회전 pivot, contact lever arm,
영상의 객체 자세가 하나의 body-local frame을 사용한다.

```json
"initialization": {
  "state_json": "trajectory.json",
  "state_frame": 0,
  "canonical_offset": [0.0, 0.0, 0.1]
}
```

`canonical_offset`은 객체 이름이나 shape preset이 아닌 asset 별 calibration metadata다.
생략하면 `[0, 0, 0]`이며, 출력 JSON의 `canonical_alignment`에 실제 적용값이
기록된다. 현재 paper-compatible 캔 asset은 Stage1 좌표의 캔 중심이
`z=0.1 m`에 있어 `[0, 0, 0.1]`을 사용한다.

### Native physics–geometry refinement

Manifest 기반 native 학습에서도 dynamic `gaussian_union` body의 object-local center와
log-radius를 contact parameter 및 초기 상태와 함께 학습할 수 있다. Collision proxy의
`source_indices`를 filtered render asset의 원본 PLY index와 대조하므로 객체/opacity filter를
사용해도 동일 Gaussian에만 보정이 적용된다. Center는 `tanh` bounded offset, radius는 bounded
log-scale로 parameterize한다. Paper-compatible mode의 gradient route는
`collision_only`이며 보정된 center/radius는 collision/dynamics에는 live tensor로,
renderer에는 detached tensor로 전달된다. 따라서 image loss는 geometry를 직접적인
photometric shortcut으로 갱신하지 않고 contact BPTT 경로를 통해 갱신한다.
`collision_and_render`는 experimental ablation에서만 사용할 수 있다.

```bash
conda run -n gaussian_splatting python tools/run_native_multibody_manifest.py \
  --manifest configs/scene_manifest.sggs_prefit_test.json \
  --output output/sggs_stage2_geometry_refine/fit.json --device cuda \
  --fit_iters 100 --render_stride 3 --render_max_frames 8 \
  --refine_geometry --geometry_lr 0.001 \
  --geometry_gradient_route collision_only \
  --geometry_center_l2 0.001 --geometry_radius_l2 0.001 \
  --geometry_max_center_delta 0.005 --geometry_max_log_radius_delta 0.1
```

최종 결과의 `geometry_refinement.refined_collision_geometry`에는 body별 원본 PLY
`source_indices`, refined `local_centers`, `radii`가 저장되어 다음 실행에서 재사용할 수 있다.
이전에 수행한 10-iteration `collision_and_render` ablation에서는 image objective가
`0.0045847 → 0.0045305`로 감소했고 256개 공유 Gaussian의 center/radius가 갱신됐다. 결과는
`output/sggs_stage2_geometry_refine/fit.json`에 있다.

Gradient-routing 회귀 테스트는 기본 경로의 renderer center/radius가 gradient를 갖지 않고
collision geometry에는 gradient가 남는지 확인한다. 실제 5-iteration 접촉 구간 smoke test도
`renderer_geometry_detached=true` 상태에서 256개 proxy 중 접촉 관련 geometry가 갱신되고
loss가 `0.0029673 → 0.0029345`로 감소했다. 결과는
`output/sggs_stage2_gradient_route/collision_only.json`에 있다.

### Native L1 + LoFTR supervision

Native multi-body runner의 `--image_loss l1_loftr`는 frozen Kornia LoFTR가 선택한
correspondence에서 differentiable normalized RGB patch loss를 계산한다. Match 선택 자체는
detach하지만 rendered patch는 live tensor이므로 pose, dynamics, contact parameter와 공유
geometry까지 gradient가 전달된다. Mask가 있으면 rendered/target keypoint가 모두 foreground인
match만 남긴다. Match가 `--loftr_min_matches`보다 적으면 LoFTR 항만 graph에 연결된 0이 되고
L1 항은 계속 학습된다.

```bash
conda run -n gaussian_splatting python tools/run_native_multibody_manifest.py \
  --manifest configs/scene_manifest.sggs_prefit_test.json \
  --output output/sggs_stage2_loftr/fit.json --device cuda \
  --fit_iters 100 --image_loss l1_loftr --loftr_pretrained outdoor \
  --loftr_weight 0.1 --loftr_confidence_threshold 0.05 \
  --loftr_max_matches 128 --loftr_min_matches 1 --loftr_patch_radius 2 \
  --refine_geometry
```

결과 JSON은 raw/confidence/final match 수, 평균 confidence, feature loss와 설정 전체를
기록한다. 8개 sampled frame의 3-iteration smoke test에서는 foreground mask 이후 10개
match, 평균 confidence 약 0.94를 사용했고 LoFTR loss가 `0.1503 → 0.1408`로 감소했다.
동일 budget의 L1-only foreground L1은 `0.3897`, L1+LoFTR은 `0.3943`으로 짧은 실행에서는
pixel metric 개선이 없었다. 따라서 현재 결과는 feature supervision 동작 검증이며 충분한
iteration/weight ablation 전에는 품질 우위로 해석하지 않는다. 비교 결과는
`output/sggs_stage2_loftr/fit.json`과 `l1_baseline.json`에 있다.

30-iteration weight ablation (`0`, `0.02`, `0.05`, `0.1`)에서는 L1-only가 foreground
L1 `0.3732`로 가장 좋았고, LoFTR weight 0.1은 feature loss가 가장 낮았지만 foreground
L1은 `0.3923`으로 나빠졌다. 현재 데이터의 권장 기본값은 L1-only이며 LoFTR가 필요하면
`0.02`부터 사용한다. 전체 설정과 결과는 `docs/sggs_loftr_weight_ablation.md`에 기록한다.

실제 접촉 구간을 포함한 end-to-end 테스트 명령, 정량 결과 및 현재 실패 판정은
`docs/native_stage2_test_report.md`에 기록한다. 이 테스트는 GIF 생성 성공만으로 물리 학습
성공을 판정하지 않고, 접촉 후 trajectory 안정성과 contact parameter 갱신 여부도 확인한다.

### 명확한 캔–바닥 충돌 테스트 영상

`tools/generate_mujoco_fall_dataset.py`는 고정 초기 상태 옵션과 `rollout.gif`, 충돌/정지
diagnostics 출력을 지원한다. 현재 채택한 테스트 episode는 다음 위치에 있다.

```text
output/can_floor_contact_test_dataset_v3/
  stage2/fall_and_rebound/train/cola_can/episode_000/
    rgb/                 # 75 frames, 640x480
    masks/               # can foreground masks
    state/trajectory.json
    episode_manifest.json
    rollout.gif
```

카메라는 2.2 m 거리, 1.1 m 높이에서 z=0.82 m를 바라본다. 캔은 z=1.5 m에서 떨어져
frame 15 부근에서 바닥과 충돌하고 frame 30에 정지한다. 전체 75 frames/2.5 s이므로 낙하,
충돌/감쇠, 정지 상태가 모두 포함되며 캔은 모든 구간에서 화면 안에 있다. 재생성 명령은 다음과 같다.

```bash
conda run -n cg_wm python tools/generate_mujoco_fall_dataset.py \
  --dataset_root output/can_floor_contact_test_dataset_v3 \
  --object_name cola_can --split train --fps 30 --seed 2026 \
  --camera_distance 2.2 --camera_height 1.1 --camera_target_z 0.82 \
  --initial_position 0,0,1.5 --initial_euler_deg 8,-6,12 \
  --initial_linear_velocity 0,0,0 --initial_angular_velocity 0.8,0.4,0.3 \
  --cylinder_friction "0.8 0.03 0.003" \
  --cylinder_solref "0.03 1" --floor_solref "0.03 1" \
  --freejoint_damping 0.25
```

이 v3 영상으로 실제 camera calibration을 적용해 Stage2를 다시 학습한 결과와 실패 분석은
`docs/contact_test_v3_stage2_report.md`에 있다. 실행은 완료되고 image loss는 감소했지만,
접촉 후 침투/횡방향 drift 및 새 MuJoCo 영상과 기존 floor Gaussian 사이의 외관 불일치 때문에
현재 물리·렌더 품질 acceptance test는 통과하지 못했다.

현실적인 육안 검사용 영상은 `output/can_floor_realistic_test`에 별도로 생성했다. 이 버전은
하늘 gradient, 작은 타일의 무광 바닥, key/fill light와 그림자, 낮은 실제 촬영 구도를 사용한다.
캔은 기울어진 상태로 떨어져 rim으로 먼저 충돌하고 옆으로 넘어져 짧게 구른 뒤 정지한다.
전체 영상은 episode의 `rollout.gif`에서 확인할 수 있다.

### 현실형 장면과 일치하는 Stage1 multiview

논문 재현 순서의 첫 단계로 현실형 Stage2와 같은 하늘, 타일 바닥, 조명을 사용하는 Stage1
multiview를 `output/realistic_stage1_multiview/can_floor_realistic`에 생성했다. 단일 orbit이
아니라 12–65도 upper-hemisphere를 golden-angle로 샘플링한다.

```bash
conda run -n cg_wm python -m stage1.generate_mujoco_synthetic_dataset \
  --output_root output/realistic_stage1_multiview \
  --scene_name can_floor_realistic --object_type cola_can \
  --alpha_subject scene --train_views 72 --test_views 12 \
  --width 640 --height 480 --fovy_deg 42 --camera_radius 1.1 \
  --camera_sampling hemisphere --min_elevation_deg 12 --max_elevation_deg 65 \
  --appearance_preset realistic_contact --mujoco_gl egl --seed 2026
```

각 view는 RGBA image, binary subject mask, `0=background, 1=can, 2=floor` instance map,
calibrated camera transform을 제공한다. 72개 train view와 12개 test view 모두 캔과 바닥
label을 포함한다.

### SG-GS Stage1 검증 경로

합성 데이터에서는 SAM2 대신 MuJoCo가 제공한 정확한 instance mask로 geometry feature를
만들 수 있다. 객체 이름이나 shape preset은 사용하지 않으며 label 정수만 feature code로
변환한다. 실제 이미지에는 `extract_sam2_features.py`를 사용한다.

```bash
conda run -n cg_wm python -m stage1.build_mask_geometry_features \
  --source_path output/realistic_stage1_multiview/can_realistic_sggs

conda run -n cg_wm python -m stage1.build_visual_hull \
  --source_path output/realistic_stage1_multiview/can_realistic_sggs \
  --masks_dir output/realistic_stage1_multiview/can_realistic_sggs/object_masks \
  --bbox_min=-0.085,-0.085,-0.005 --bbox_max=0.085,0.085,0.215 \
  --grid_resolution 128 --max_points 120000

conda run -n gaussian_splatting python -m stage1.train \
  -s output/realistic_stage1_multiview/can_realistic_sggs \
  -m output/stage1_sggs_can_test --eval --resolution 2 \
  --init_mode visual_hull --sam_features mask_geometry_features \
  --masks_dir object_masks --stage1_preset contactwm \
  --sam_feature_weight 0.1 --object_mask_weight 0.1 --iterations 600 \
  --densify_from_iter 100 --densify_until_iter 400 --disable_viewer
```

학습 뒤 `auto_assign_object_ids.py --propagate_unassigned`를 실행하면 2D mask 투표를 받은
표면 Gaussian의 ownership을 가려진 Visual-Hull 내부점까지 전파한다. 검증 결과는
`output/stage1_sggs_can_test_owned_complete`에 있으며 119,370개 Gaussian 모두 캔 ID 1이다.
저장 PLY의 세 log-scale 축 차이와 identity quaternion 오차는 모두 0이며, SIBR 없이 생성한
12개 test render는 `test/ours_600/renders/`에 있다.

`--gaussian_mask_dir`를 주면 foreground만 비교하고 배경은 설정된 renderer background로
합성합니다. `--gaussian_render_stride`와 `--gaussian_render_max_frames`로 sparse frame
supervision을 구성할 수 있습니다. 여러 calibrated fixed view는
`--gaussian_views_manifest`로 전달합니다.

`l1_loftr`는 `kornia==0.6.12`의 frozen pretrained LoFTR를 사용합니다. 환경을 갱신한
뒤 처음 실행할 때 `indoor` 또는 `outdoor` weight가 다운로드되어 캐시될 수 있습니다.
`--loftr_confidence_threshold`, `--loftr_max_matches`,
`--loftr_min_matches`, `--loftr_patch_radius`로 correspondence selection을 조절합니다.
유효 매칭 수가 최소값보다 작으면 해당 frame batch의 LoFTR 항만 0이 되고 L1 항은
그대로 학습됩니다. Mask를 사용하면 양쪽 keypoint가 모두 foreground인 매칭만 남습니다.
실제 사용된 match 수, 평균 confidence와 feature loss는 iteration diagnostics 및 최종
결과에 기록됩니다. 논문이 LoFTR feature loss의 세부 공식을 공개하지 않았기 때문에,
현재 구현은 공개 LoFTR correspondence에 differentiable normalized RGB patch feature
loss를 적용한 재현 가능한 근사입니다.

```json
{
  "views": [
    {
      "rgb_dir": "camera_0/rgb",
      "mask_dir": "camera_0/mask",
      "image_width": 160,
      "image_height": 120,
      "cam_distance": 1.12,
      "cam_height": 0.66,
      "cam_fovy_deg": 40.0
    }
  ]
}
```

모든 single-body Stage 2 fit은 출력 디렉토리에 재현성 파일도 자동 저장합니다.

```text
resolved_config.json
input_episode_manifest.json
experiment_bundle.json
```

`experiment_bundle.json`에는 Git commit/branch/dirty 상태와 tracked diff hash, 전체 실행
config, episode manifest, Stage-I 및 환경 PLY 해시, `fit_summary.json`,
`predicted_trajectory.json`, diagnostic GIF의 SHA-256이 포함됩니다. 비교·budget
ablation·holdout runner도 내부적으로 같은 fit entrypoint를 사용하므로 모든 하위 run에
동일한 bundle이 생성됩니다.

## Query budget ablation

Query scheme 비교에서 geometry resolution과 query 수의 효과가 섞이지 않도록 두 suite를
별도로 실행할 수 있습니다.

```bash
conda run -n gaussian_splatting python \
   tools/run_query_budget_ablations.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --output_root <ablation_output>
```

- `primitive` suite: scheme마다 moving-body Gaussian primitive 수를 동일하게 유지합니다.
- `point` suite: `primitive 수 × primitive당 direction 수`가 동일하도록 primitive cap을
  역산합니다.
- 두 suite 모두 SDF 평가 뒤 남기는 contact patch 수 `--body_lowest_k`는 독립적으로
  고정됩니다.

각 실행의 요청 budget과 필터링 후 실제 primitive/query 후보 수는
`query_budget_ablation_report.json`에 함께 기록됩니다.

## Multi-episode holdout evaluation

여러 물리 초기값과 여러 episode를 사용한 train/holdout 비교는 다음 runner로 실행합니다.

```bash
conda run -n gaussian_splatting python \
   tools/run_multi_episode_holdout_comparison.py \
  --dataset_manifest <dataset_manifest.json> \
  --output_root <holdout_output>
```

dataset manifest의 각 episode에는 `name`, `split` (`train` 또는 `holdout`),
`episode_root`, `stage1_ply`가 필요합니다. 선택적으로 `pairwise_body_b_ply`를 지정할 수
있습니다. Runner는 query scheme마다 train episode 평균 점수가 가장 좋은 초기값을
선택하고, train에서 학습된 stiffness/damping 중앙값을 holdout에 고정합니다. Holdout은
`--eval_only`로 optimizer step 없이 자립 rollout만 평가합니다. 전체 선택 과정과 개별
metric은 `multi_episode_holdout_report.json`에 저장됩니다.

## 영상 기반 초기 상태 prefit

물리 파라미터와 초기 상태가 서로의 오차를 대신 설명하지 않도록 초기 상태를 별도
단계에서 먼저 추정할 수 있습니다.

```bash
conda run -n gaussian_splatting python \
   tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --prefit_initial_state \
  --prefit_pose_iters 100 \
  --prefit_velocity_iters 100 \
  --prefit_velocity_frames 3 \
  --gaussian_rgb_dir <rgb_dir>
```

첫 프레임의 Gaussian image loss로 위치와 quaternion을 먼저 맞춘 뒤, 처음 N개의
접촉 전 프레임에 constant-velocity 모델을 적용해 선속도와 각속도를 추정합니다.
추정된 pose/velocity는 이후 physics–geometry fit 동안 고정되고
`initial_state_estimate.json`과 experiment bundle에 함께 저장됩니다. 처음 N개 프레임은
접촉이 없어야 하며, RGB prefit은 CUDA Gaussian rasterizer가 필요합니다. 이미 추정한
상태는 `--initial_state_json <initial_state_estimate.json>`으로 재사용할 수 있습니다.

## Git에 올리지 않는 파일

다음 파일/폴더는 실험 결과물 또는 재생성 가능한 캐시라서 `.gitignore`에 포함되어 있습니다.

```text
__pycache__/
*.pyc
actual_*/
 output/
 stage2/_outputs/
 sam_features_sam2/
 **/visual_hull/
 **/physics_export/
 **/build/
```

학습된 모델 checkpoint, 렌더 결과, MuJoCo rollout, evaluation report는 필요한 경우 별도 artifact storage에 보관하는 것을 권장합니다.

## 현재 한계

- ContactGaussian-WM 논문의 전체 공식 pipeline 재현은 아닙니다.
- Stage 1의 SAM feature supervision은 데이터셋에 precomputed feature map이 있어야 합니다.
- Gaussian renderer loss는 CUDA 환경에서만 실제 backward 검증이 가능합니다.
- real-world LEAP Hand 입력과 DreamerV3/PIN-WM baseline은 아직 포함하지 않습니다.

## 15-frame paper protocol benchmark

`tools/run_paper_protocol_benchmark.py`는 하나의 train scene manifest에서 앞 15개
관측 프레임만 사용해 학습하고, 학습된 contact parameter를 여러 unseen holdout
manifest의 전체 open-loop sequence에 고정 적용한다. 같은 조건에서 초기 parameter를
그대로 쓰는 `no_opt`와 seeded log-space CEM baseline도 평가한다. Evaluation trajectory가
있으면 translation/rotation error를, 없으면 full-frame RGB L1/PSNR을 사용한다.

Protocol JSON 형식:

```json
{
  "train_manifest": "configs/train_scene.json",
  "holdout_manifests": [
    "configs/holdout_pose_01.json",
    "configs/holdout_pose_02.json"
  ]
}
```

```bash
conda run -n gaussian_splatting python tools/run_paper_protocol_benchmark.py \
  --protocol configs/paper_protocol.json \
  --output output/paper_protocol/report.json \
  --fit_iters 250 --train_frames 15 --cem_candidates 100 --device cuda
```

Native image-only optimizer는 기본적으로 초기 20% iteration 동안 pose/velocity만
맞춘 뒤 physics/geometry를 여는 curriculum을 사용한다. Contact-parameter prior,
trajectory 폭주 penalty, 전체 parameter gradient clipping과 best-state 복원도 적용된다.

### Contact-identification stability bundle

Experimental/image-only fitting은 positive physics 값을 log-space
`K=exp(log_K)`, `D=exp(log_D)`, `mu=exp(log_mu)`에서 최적화한다. 세 parameter는
`--stiffness_lr`, `--damping_lr`, `--friction_lr`로 서로 다른 learning rate를 사용할 수
있다. 각 iteration은 `physics_gradient_norms`에 parameter별 gradient norm을 기록한다.

초기 model rollout의 contact gate로 접촉 frame을 추정하고, warm-up 이후
`--contact_curriculum_frames`개의 pre/contact/post-contact 관측을 집중 sampling한다.
선택된 frame과 batch 안의 contact frame은 loss history에 저장된다. GT trajectory는
접촉 frame 선택에 사용하지 않는다.

Mask가 있는 experimental fitting에는 `--silhouette_weight`로 asymmetric silhouette
loss를 추가할 수 있다. 기본 false-positive weight는 2, false-negative weight는 1이므로
예측 객체가 GT mask 밖이나 화면 경계로 이탈하는 것을 더 강하게 벌점화한다. Smooth
occupancy를 사용해 foreground가 강한 pixel에서도 gradient가 포화되지 않는다.
Paper-compatible mode는 논문의 full-image L1+LoFTR contract를 보존하기 위해 이 항을
자동으로 비활성화한다.

Silhouette 기본 weight는 `0.005`다. 아래 runner는 동일한 15-frame/50-iteration
조건에서 안정화 적용 전후, K/D/mu learning-rate 4조합, silhouette weight 5조합을
각각 학습한 뒤 protocol의 unseen holdout 전체를 고정된 physics로 평가한다. 개별
실험 결과와 group별 최저 holdout score는 output 디렉터리의 JSON으로 저장된다.

```bash
conda run -n gaussian_splatting python tools/run_contact_stability_ablations.py \
  --protocol configs/paper_benchmark_analytic_protocol.json \
  --output output/contact_stability_ablations --device cuda
```

중단된 sweep은 같은 명령에 `--resume`을 붙여 이어갈 수 있고, 예를 들어 안정화 비교만
실행하려면 `--groups stabilization`을 사용한다.

### Loss별 gradient attribution

`--gradient_attribution`을 켜면 weighted image L1/SSIM/LoFTR, silhouette,
geometry·mass·inertia·contact L2, trajectory stability 항이 각각 initial state,
geometry, mass/inertia, log K, log D, log mu에 만드는 gradient norm을 iteration별
`loss_gradient_attribution`에 기록한다. 측정에는 `autograd.grad`를 사용하므로 optimizer의
실제 `.grad`를 변경하지 않는다. 비용을 줄이려면 예를 들어
`--gradient_attribution_interval 5`로 매 5 iteration만 측정할 수 있다.

```bash
conda run -n gaussian_splatting python tools/run_contact_stability_ablations.py \
  --protocol configs/paper_benchmark_analytic_protocol.json \
  --output output/contact_gradient_attribution --device cuda \
  --groups lr --gradient_attribution --gradient_attribution_interval 1
```

Ablation runner는 개별 iteration 기록을 loss/parameter group별 mean, max,
nonzero-iteration 수로 집계해 `loss_gradient_attribution_summary`에 저장한다.
선정된 LR/silhouette 설정 하나만 50 iteration 진단하려면 전용
`--groups attribution --gradient_attribution` 조합을 사용한다.

Warm-up이 끝나면 experimental/image-only optimizer는 기본적으로 initial position,
orientation, linear/angular velocity를 freeze하고 physics·mass/inertia·geometry만 갱신한다.
각 iteration의 `frozen_parameter_groups`에서 실제 동결 대상을 확인할 수 있다. 기존처럼
initial state를 끝까지 공동 최적화하는 ablation은
`--keep_initial_state_trainable_after_warmup`으로 실행한다.

### Multi-episode open-loop 평가

`tools/run_multi_episode_evaluation.py`는 학습 결과의 contact physics를 고정하고 episode별
초기상태에서 전체 sequence를 open-loop 평가한다. Protocol은 `test_001`과 최소 하나의
추가 episode를 반드시 포함해야 한다. Episode별 score/RGB/trajectory metric 외에 전체
평균, population 표준편차, min/max, worst episode를 저장한다.

```bash
conda run -n gaussian_splatting python tools/run_multi_episode_evaluation.py \
  --protocol configs/multi_episode_test_protocol.json \
  --physics_report output/contact_gradient_attribution/report.json \
  --experiment gradient_attribution_recommended \
  --output output/multi_episode_evaluation --device cuda
```

Episode layout의 embedded initial state, RGB, mask, trajectory를 기존 scene-manifest
template에 자동 연결하며, 생성된 평가 manifest도 결과 디렉터리에 보존한다.

현재 analytic protocol의 최종 100/250-iteration 비교와 선택은
`configs/final_contact_fit_selection.json`에 고정되어 있다. Multi-episode mean score는
100회가 0.37460, 250회가 0.40008이므로 현재 선택은 100회다. Best-state 복원은 서로
다른 objective 구간을 비교하지 않도록 warm-up을 제외한 joint-physics checkpoint에만
적용한다.

## 브랜치 업로드 예시

```bash
git switch -c contactgaussian-wm-stage1-stage2
git add -u
git add \
   stage1_training_presets.json \
   stage2/differentiable_gaussian_render_loss.py \
   stage2/renderable_gaussian_asset.py \
   tools/evaluate_multi_dice_stage2_variants.py \
   tools/render_stage2_gaussian_trajectory.py \
   tools/run_stage1_training_schedule.py
git commit -m "Implement ContactGaussian-WM stage1 stage2 prototype"
git push -u origin contactgaussian-wm-stage1-stage2
```
