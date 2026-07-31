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
  gaussian_initiailization/tools/compare_query_modes.py \
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
from gaussian_initiailization.stage2.articulated_kinematics import (
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
  gaussian_initiailization/tools/prepare_articulated_trajectory.py \
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

`gaussian_initiailization/` 폴더명에는 기존 저장소의 오타(`initiailization`)가 그대로 남아 있습니다.

## 실험 결과물 위치

MuJoCo 데이터셋과 Stage 2 결과물은 저장소 루트의 `output/` 아래에 모읍니다 (git 미추적).

```text
output/actual_cola_can_*_data/     MuJoCo GT 에피소드 (rgb/, masks/, state/trajectory.json)
output/actual_cola_can_*_output/   Stage 2 fit 결과, 렌더, 비교 GIF
```

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
gaussian_initiailization/train.py
gaussian_initiailization/render.py
gaussian_initiailization/metrics.py
gaussian_initiailization/export_physics_scene.py
gaussian_initiailization/stage1_training_presets.json
gaussian_initiailization/tools/run_stage1_training_schedule.py
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
gaussian_initiailization/stage2/differentiable_collision_detection.py
gaussian_initiailization/stage2/differentiable_contact_graph.py
gaussian_initiailization/stage2/differentiable_complementarity_free_contact_dynamics.py
gaussian_initiailization/stage2/renderable_gaussian_asset.py
gaussian_initiailization/stage2/differentiable_gaussian_render_loss.py
gaussian_initiailization/tools/generate_mujoco_multi_dice_rollout.py
gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py
gaussian_initiailization/tools/evaluate_multi_dice_stage2_variants.py
```

## 빠른 실행

### Stage 1 Schedule Dry Run

실제 학습을 돌리기 전에 어떤 명령이 실행되는지 확인합니다.

```bash
python gaussian_initiailization/tools/run_stage1_training_schedule.py \
  --preset dice_smoke \
  --dry_run
```

ContactGaussian-WM 스타일 preset은 SAM feature가 필요합니다.

```bash
python gaussian_initiailization/tools/run_stage1_training_schedule.py \
  --preset contactwm_smoke \
  --dry_run \
  --print_json
```

### Stage 1 Smoke Schedule

```bash
python gaussian_initiailization/tools/run_stage1_training_schedule.py \
  --preset dice_smoke
```

이 runner는 `--python`과 `--mujoco_python`에 shell command가 아니라 Python executable 경로를 받습니다. Conda 환경을 분리해서 쓰는 경우에는 dataset 생성, train, render 단계를 따로 실행하거나 각 환경의 Python 실행 파일 경로를 넘겨야 합니다.

### Multi-Dice MuJoCo Dataset

```bash
conda run -n mujoco python gaussian_initiailization/tools/generate_mujoco_multi_dice_rollout.py \
  --output_root actual_multi_dice_mujoco \
  --scene_name demo_codex \
  --num_dice 3 \
  --frames 120 \
  --width 256 \
  --height 256
```

### Stage 2 Rollout Comparison

```bash
conda run -n gaussian_splatting python gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py \
  --trajectory actual_multi_dice_mujoco/demo_codex/trajectory.json \
  --stage1_ply gaussian_initiailization/output/<stage1_model>/point_cloud/iteration_30000/point_cloud.ply \
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
conda run -n gaussian_splatting python gaussian_initiailization/tools/evaluate_multi_dice_stage2_variants.py \
  --trajectory actual_multi_dice_mujoco/demo_codex/trajectory.json \
  --stage1_ply gaussian_initiailization/output/<stage1_model>/point_cloud/iteration_30000/point_cloud.ply \
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

```bash
conda run -n gaussian_splatting python gaussian_initiailization/tools/render_stage2_gaussian_pose_smoke.py \
  --stage1_ply gaussian_initiailization/output/<stage1_model>/point_cloud/iteration_30000/point_cloud.ply \
  --output_path actual_stage2_render/pose_smoke.png \
  --device cuda
```

주의: `diff-gaussian-rasterization` 기반 renderer는 CUDA가 필요합니다. CPU-only 환경에서는 renderer backward까지 검증할 수 없습니다.

Single-body pairwise fitting에서도 trajectory, geometry, Gaussian RGB supervision을 하나의
objective로 사용할 수 있습니다.

```bash
conda run -n gaussian_splatting python \
  gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py \
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
loss와 parameter prior만으로 best state를 선택합니다.

```bash
conda run -n gaussian_splatting python \
  gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <episode_root> \
  --stage1_ply <point_cloud.ply> \
  --dynamics pairwise_impedance \
  --refine_geometry \
  --geometry_gradient_route collision_only \
  --image_only_objective \
  --gaussian_rgb_loss_weight 1.0 \
  --gaussian_render_loss l1_loftr \
  --gaussian_render_loftr_weight 0.1 \
  --loftr_pretrained outdoor \
  --device cuda
```

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

## 검증한 항목

최근 smoke 기준으로 확인한 항목입니다.

- Python syntax compile
- Gaussian union SDF와 aggregate contact gradient
- pairwise/multibody contact dynamics rollout
- pairwise `axis6`/`fibonacci`/`analytic` body-query generation and gradient
- friction cone projection gradient
- geometry refinement parameter save/load
- Stage 2 evaluator dry/smoke run
- Stage 1 schedule dry run

대표 검증 명령:

```bash
python -m py_compile \
  gaussian_initiailization/stage2/differentiable_collision_detection.py \
  gaussian_initiailization/stage2/differentiable_complementarity_free_contact_dynamics.py \
  gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py \
  gaussian_initiailization/tools/evaluate_multi_dice_stage2_variants.py \
  gaussian_initiailization/tools/run_stage1_training_schedule.py
```

## Query budget ablation

Query scheme 비교에서 geometry resolution과 query 수의 효과가 섞이지 않도록 두 suite를
별도로 실행할 수 있습니다.

```bash
conda run -n gaussian_splatting python \
  gaussian_initiailization/tools/run_query_budget_ablations.py \
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
  gaussian_initiailization/tools/run_multi_episode_holdout_comparison.py \
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
  gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py \
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
gaussian_initiailization/output/
gaussian_initiailization/stage2/_outputs/
gaussian_initiailization/sam_features_sam2/
gaussian_initiailization/**/visual_hull/
gaussian_initiailization/**/physics_export/
gaussian_initiailization/**/build/
```

학습된 모델 checkpoint, 렌더 결과, MuJoCo rollout, evaluation report는 필요한 경우 별도 artifact storage에 보관하는 것을 권장합니다.

## 현재 한계

- ContactGaussian-WM 논문의 전체 공식 pipeline 재현은 아닙니다.
- Stage 1의 SAM feature supervision은 데이터셋에 precomputed feature map이 있어야 합니다.
- Gaussian renderer loss는 CUDA 환경에서만 실제 backward 검증이 가능합니다.
- real-world video 입력과 논문 표 수준의 benchmark 자동화는 아직 정리 중입니다.

## 브랜치 업로드 예시

```bash
git switch -c contactgaussian-wm-stage1-stage2
git add -u
git add \
  gaussian_initiailization/stage1_training_presets.json \
  gaussian_initiailization/stage2/differentiable_gaussian_render_loss.py \
  gaussian_initiailization/stage2/renderable_gaussian_asset.py \
  gaussian_initiailization/tools/evaluate_multi_dice_stage2_variants.py \
  gaussian_initiailization/tools/render_stage2_gaussian_pose_smoke.py \
  gaussian_initiailization/tools/render_stage2_gaussian_trajectory.py \
  gaussian_initiailization/tools/run_stage1_training_schedule.py \
  gaussian_initiailization/tools/smoke_stage2_gaussian_render_loss_backward.py
git commit -m "Implement ContactGaussian-WM stage1 stage2 prototype"
git push -u origin contactgaussian-wm-stage1-stage2
```
