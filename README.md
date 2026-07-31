# CG-WM

ContactGaussian-WM 논문 아이디어를 바탕으로 Stage 1 Gaussian scene initialization과 Stage 2 differentiable contact simulation을 실험하는 연구용 저장소입니다.

현재 코드는 논문 전체를 완전히 재현한 공식 구현체가 아니라, 주사위/박스 같은 rigid object를 대상으로 object-aware Gaussian proxy를 만들고, 그 proxy를 이용해 differentiable collision/contact dynamics를 검증하는 프로토타입입니다.

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

## 검증한 항목

최근 smoke 기준으로 확인한 항목입니다.

- Python syntax compile
- Gaussian union SDF와 aggregate contact gradient
- pairwise/multibody contact dynamics rollout
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
