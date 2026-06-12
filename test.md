# 멀티 주사위 Stage1 → Stage2 테스트 가이드

Stage1 가우시안 에셋 하나를 N개 인스턴스로 복제해 MuJoCo GT와 비교하는
멀티 주사위 파이프라인의 재현 절차. 모든 명령은 저장소 루트에서 실행한다.

요구 패키지: `mujoco`(3.x), `torch`, `imageio`, `Pillow`, `numpy`
(설치된 본인의 파이썬 환경에서 실행하면 된다. conda 환경이라면
`conda activate <env>` 후 아래 명령을 그대로 사용.)

## 1. Stage1 주사위 에셋 생성

권장 방식은 preset 기반 schedule runner를 사용하는 것이다.

```bash
# 빠른 smoke: dataset 생성 → Stage1 학습 → render/metrics/export 명령 순서 실행
python gaussian_initiailization/tools/run_stage1_training_schedule.py \
  --preset dice_smoke \
  --data_root actual_dice_stage1_data \
  --output_root actual_dice_stage1_output \
  --scene_name dice_asset_smoke \
  --model_name dice_asset_smoke_stage1_3000

# 실행하지 않고 전체 명령만 확인
python gaussian_initiailization/tools/run_stage1_training_schedule.py \
  --preset dice_full \
  --scene_name dice_asset_full \
  --model_name dice_asset_full_stage1_30000 \
  --dry_run
```

Preset은 `gaussian_initiailization/stage1_training_presets.json`에 있다.
`dice_smoke`, `dice_full`은 object mask 기반으로 바로 실행 가능하고,
`contactwm_smoke`는 `sam_features_sam2`가 준비된 경우에만 사용한다.

수동 실행은 아래처럼 할 수 있다.

```bash
# 합성 데이터셋 (눈금 있는 주사위, 궤도 뷰 16/4장)
python gaussian_initiailization/generate_mujoco_synthetic_dataset.py \
  --object_type dice --scene_name dice_asset_smoke \
  --train_views 16 --test_views 4 --width 256 --height 256 \
  --output_root actual_dice_stage1_data

# Stage1 학습 (smoke: 3000 iter)
python gaussian_initiailization/train.py \
  --source_path actual_dice_stage1_data/dice_asset_smoke \
  --model_path actual_dice_stage1_output/dice_asset_smoke_stage1_3000 \
  --masks_dir actual_dice_stage1_data/dice_asset_smoke/masks \
  --iterations 3000 --eval --disable_viewer
```

산출물: `actual_dice_stage1_output/dice_asset_smoke_stage1_3000/point_cloud/iteration_3000/point_cloud.ply`
(아래에서 `$PLY`로 표기)

## 2. MuJoCo GT 멀티 주사위 롤아웃 생성

```bash
python gaussian_initiailization/tools/generate_mujoco_multi_dice_rollout.py \
  --output_dir actual_multi_dice_mujoco/episode_000 \
  --dice_count 5 --frames 150 --seed 23
```

산출물: `rgb/`, `masks/`(per-die + union), `trajectory.json`(상태/속도/접촉),
GT GIF와 몽타주.

## 3. 접촉 그래프 평가

```bash
python gaussian_initiailization/tools/evaluate_multi_dice_contact_graph.py \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --stage1_ply $PLY \
  --rgb_dir actual_multi_dice_mujoco/episode_000/rgb \
  --output_dir actual_multi_dice_contact_graph/episode_000 \
  --max_frames 150
```

`contact_graph_summary.json`의 `mujoco_contact_comparison`에 세 가지 기준이 기록된다.

- 순간 라벨 기준 precision/recall: MuJoCo 접촉 기록이 30fps 프레임 순간에만 남아
  공중 접촉이 누락되므로 비관적으로 나온다 (참고용).
- `tolerant_*`: ±`--temporal_tolerance_frames`(기본 2) 프레임 내 접촉이면 TP.
- `gap_*`: 정확한 box-box 거리(교대 투영) 기준, 간격 ≤ `--gap_tolerance`(기본 12mm,
  구 프록시 표면 해상도)이면 GT 양성. **이게 프록시의 실제 검출력을 나타낸다.**

기대값(시드 23, 150프레임): gap precision ≈ 0.82, gap recall ≈ 0.78.

## 4. Stage2 롤아웃 비교 (GT vs 예측)

```bash
python gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --gt_rgb_dir actual_multi_dice_mujoco/episode_000/rgb \
  --stage1_ply $PLY \
  --output_dir actual_multi_dice_stage2_rollout/episode_000_fitted \
  --dynamics_backend stage2_impedance \
  --gt_mask_dir actual_multi_dice_mujoco/episode_000/masks/all \
  --mask_loss_weight 0.1 --mask_loss_resolution 64 \
  --max_frames 100 --max_primitives 256 --substeps 4 \
  --fit_iters 40 --fit_lr 0.03 \
  --fit_physics_iters 40 --fit_physics_lr 0.02 \
  --fit_horizon_frames 36
```

산출물 (RTX 3070 + 5600X 기준 약 15분):

- `gt_vs_stage2_predicted_rollout.gif` — 좌 GT / 우 예측 side-by-side **(메인 결과물)**
- `gt_vs_stage2_predicted_rollout_montage.png` — 발표용 정지 이미지 12장
- `stage2_rollout_summary.json` — position RMSE, 회전 오차(deg), fit 히스토리,
  사용한 dynamics backend

기본 backend `stage2_impedance`는 Stage2 core의 Gaussian N-body impedance dynamics를 사용한다.
`--fit_physics_iters`는 논문 Stage II의 Phys-Geo refinement에 해당하는 경로로,
초기 속도와 함께 `M/K/D/µ/tangential damping/radius multiplier`를 trajectory loss로 최적화한다.
`--fit_geometry_radii`를 추가하면 primitive별 radius multiplier를, `--fit_geometry_centers`를 추가하면
primitive별 local center offset을 함께 학습한다. center offset은 기본 ±15mm 범위로 제한된다.
마찰은 soft Coulomb friction cone projection을 사용한다. `--pair_friction`은 sliding/dynamic friction,
`--stage2_static_friction`은 저속 sticking 영역의 static friction, `--stage2_friction_transition_velocity`는
sticking/sliding 전환 속도를 조절한다.
`--gt_mask_dir`와 `--mask_loss_weight`를 주면 differentiable soft silhouette loss도 함께 사용해
논문식 image-space supervision에 더 가까운 fitting을 수행한다.
CUDA에서 `--device cuda --gaussian_rgb_loss_weight 0.05 --gaussian_render_stride 4`를 추가하면
Stage1 Gaussian renderer를 직접 통과하는 RGB loss까지 사용한다.
이 RGB loss는 Stage2 collision proxy를 만들 때 계산한 `stage1_to_mujoco_scale`을 render asset에도
자동 적용하므로, 물리와 렌더가 같은 canonical object scale을 본다.
`--dynamics_backend impulse`를 주면 이전 데모용 impulse solver와 비교할 수 있다.

fit 결과를 다음 rollout/render에서 재사용하려면 refined params JSON을 저장/로드한다.

```bash
# fit 결과 저장
python gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --stage1_ply $PLY \
  --output_dir actual_multi_dice_stage2_rollout/episode_000_fit_save \
  --dynamics_backend stage2_impedance \
  --fit_physics_iters 40 --fit_horizon_frames 36 \
  --fit_geometry_radii --fit_geometry_centers \
  --pair_friction 0.45 --stage2_static_friction 0.8 \
  --save_refined_params actual_multi_dice_stage2_rollout/episode_000_refined_params.json \
  --skip_render

# 저장된 params로 fit 없이 rollout/render 재실행
python gaussian_initiailization/tools/run_stage2_multi_dice_rollout_comparison.py \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --gt_rgb_dir actual_multi_dice_mujoco/episode_000/rgb \
  --stage1_ply $PLY \
  --output_dir actual_multi_dice_stage2_rollout/episode_000_refined_replay \
  --dynamics_backend stage2_impedance \
  --load_refined_params actual_multi_dice_stage2_rollout/episode_000_refined_params.json
```

## 5. Stage2 variant 자동 평가

```bash
python gaussian_initiailization/tools/evaluate_multi_dice_stage2_variants.py \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --stage1_ply $PLY \
  --gt_rgb_dir actual_multi_dice_mujoco/episode_000/rgb \
  --gt_mask_dir actual_multi_dice_mujoco/episode_000/masks/all \
  --output_root actual_multi_dice_stage2_eval/episode_000 \
  --variants impulse stage2 velocity_fit physics_fit mask_fit \
  --max_frames 100 --max_primitives 256 --substeps 4 \
  --fit_iters 40 --fit_physics_iters 40 --fit_horizon_frames 36
```

산출물:

- `multi_dice_stage2_variant_report.json` — variant별 설정, command, aggregate metric
- `multi_dice_stage2_variant_results.csv` — 표/그래프용 flat metrics

CUDA Gaussian rasterizer가 있는 환경에서는 `--device cuda --variants gaussian_rgb_fit`을 추가해
실제 Stage1 Gaussian RGB render loss variant를 별도로 평가할 수 있다.
mask loss와 Gaussian RGB render loss를 동시에 쓰려면 CUDA 환경에서
`--device cuda --variants full_image_fit`을 사용한다. 이 variant는 trajectory loss,
soft silhouette mask loss, full Gaussian RGB render loss를 함께 최적화하는 가장 논문식에 가까운
평가 preset이다.

## 6. Stage1 Gaussian rigid-pose render smoke

CUDA Gaussian rasterizer가 가능한 환경에서 Stage1 `.ply`를 Stage2 rigid pose로 변환해
한 장 렌더링한다.

```bash
python gaussian_initiailization/tools/render_stage2_gaussian_pose_smoke.py \
  --stage1_ply $PLY \
  --output_png actual_multi_dice_stage2_eval/render_pose_smoke.png \
  --position 0,0,0.08 --quaternion_wxyz 1,0,0,0 \
  --image_width 640 --image_height 480 --white_background
```

CUDA가 없는 서버에서 smoke pipeline만 확인하려면 `--allow_cpu_skip`을 붙인다.

## 7. Gaussian RGB loss backward smoke

CUDA Gaussian rasterizer가 가능한 환경에서 Stage2 pose tensor까지 RGB render loss gradient가
흐르는지 확인한다. 내부적으로 target pose를 한 번 렌더해 GT RGB를 만들고, 살짝 어긋난 pose의
loss를 backward해 position/quaternion gradient를 검사한다.

```bash
python gaussian_initiailization/tools/smoke_stage2_gaussian_render_loss_backward.py \
  --stage1_ply $PLY \
  --output_dir actual_multi_dice_stage2_eval/render_loss_backward_smoke \
  --image_width 96 --image_height 72
```

CUDA가 없는 서버에서 진입 경로만 확인하려면 `--allow_cpu_skip`을 붙인다.

## 8. Stage2 Gaussian trajectory render

CUDA Gaussian rasterizer가 가능한 환경에서 Stage2 pose trajectory 전체를 Stage1 Gaussian
asset 인스턴스들로 렌더링한다.

```bash
python gaussian_initiailization/tools/render_stage2_gaussian_trajectory.py \
  --stage1_ply $PLY \
  --trajectory actual_multi_dice_mujoco/episode_000/trajectory.json \
  --output_dir actual_multi_dice_stage2_eval/episode_000_gaussian_render \
  --max_frames 100 --image_width 640 --image_height 480 \
  --fps 12 --white_background \
  --auto_scale_to_trajectory_half_extent
```

산출물:

- `gaussian_rgb/` — Stage2 pose별 Gaussian render PNG
- `stage2_gaussian_trajectory.gif` — 발표/비교용 trajectory GIF
- `stage2_gaussian_trajectory_manifest.json` — 입력, 출력, frame mapping 기록

CUDA가 없는 서버에서 진입 경로만 확인하려면 `--allow_cpu_skip`을 붙인다.

## 주의사항

- 주사위 굴리기는 카오스 시스템이라 프레임 단위 궤적 일치는 원리적으로 불가능하다.
  평가 포인트는 거동의 그럴듯함(텀블링, 면 정착)과 정착 오차다.
- `--fit_horizon_frames`는 비행~첫 바운스 구간(30~40프레임)으로 짧게 잡아야 한다.
  전체 구간을 fit하면 카오스 노이즈가 gradient를 오염시켜 오히려 나빠진다.
- fit은 CPU에서 도는 것이 빠르다(`--device cpu` 기본). 텐서가 작아 GPU 이득이 없다.
- 빠른 동작 확인만 하려면 `--fit_iters 0 --max_frames 40`으로 2~3분이면 끝난다.
- 렌더링은 `--mujoco_gl` 기본값 `glfw` 기준이다. 디스플레이 없는 환경(서버)에서는
  `--mujoco_gl egl` 또는 `osmesa`로 바꾼다.
