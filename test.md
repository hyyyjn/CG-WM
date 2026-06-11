# 멀티 주사위 Stage1 → Stage2 테스트 가이드

Stage1 가우시안 에셋 하나를 N개 인스턴스로 복제해 MuJoCo GT와 비교하는
멀티 주사위 파이프라인의 재현 절차. 모든 명령은 저장소 루트에서 실행한다.

요구 패키지: `mujoco`(3.x), `torch`, `imageio`, `Pillow`, `numpy`
(설치된 본인의 파이썬 환경에서 실행하면 된다. conda 환경이라면
`conda activate <env>` 후 아래 명령을 그대로 사용.)

## 1. Stage1 주사위 에셋 생성

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
  --max_frames 100 --max_primitives 256 --substeps 4 \
  --fit_iters 40 --fit_lr 0.03 --fit_horizon_frames 36
```

산출물 (RTX 3070 + 5600X 기준 약 15분):

- `gt_vs_stage2_predicted_rollout.gif` — 좌 GT / 우 예측 side-by-side **(메인 결과물)**
- `gt_vs_stage2_predicted_rollout_montage.png` — 발표용 정지 이미지 12장
- `stage2_rollout_summary.json` — position RMSE, 회전 오차(deg), fit 히스토리

기대값(시드 23): position RMSE ≈ 0.19m, 주사위가 튕기고 구른 뒤 면으로 정착.

## 주의사항

- 주사위 굴리기는 카오스 시스템이라 프레임 단위 궤적 일치는 원리적으로 불가능하다.
  평가 포인트는 거동의 그럴듯함(텀블링, 면 정착)과 정착 오차다.
- `--fit_horizon_frames`는 비행~첫 바운스 구간(30~40프레임)으로 짧게 잡아야 한다.
  전체 구간을 fit하면 카오스 노이즈가 gradient를 오염시켜 오히려 나빠진다.
- fit은 CPU에서 도는 것이 빠르다(`--device cpu` 기본). 텐서가 작아 GPU 이득이 없다.
- 빠른 동작 확인만 하려면 `--fit_iters 0 --max_frames 40`으로 2~3분이면 끝난다.
- 렌더링은 `--mujoco_gl` 기본값 `glfw` 기준이다. 디스플레이 없는 환경(서버)에서는
  `--mujoco_gl egl` 또는 `osmesa`로 바꾼다.
