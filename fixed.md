# 수정 내용 정리

## 목적

여러 개의 주사위 trajectory를 하나의 Stage1 Gaussian 주사위 모델로 복제 렌더링할 때, Stage1 PLY에 포함된 배경/잔여 Gaussian까지 같이 복제되어 화면에 이상한 덩어리가 보이는 문제가 있었다. 이를 해결하기 위해 렌더링 전에 주사위 본체 Gaussian만 필터링하고, 필터링된 bbox 기준으로 크기를 보정할 수 있게 수정했다.

## 변경 파일

- `gaussian_initiailization/tools/render_stage2_gaussian_trajectory.py`

## 주요 변경 사항

- `--foreground_threshold` 옵션 추가
  - foreground score가 기준값 이상인 Gaussian만 렌더링한다.
- `--opacity_threshold` 옵션 추가
  - opacity가 너무 낮은 Gaussian을 렌더링에서 제외한다.
- `--recenter_asset` 옵션 추가
  - 필터링된 Gaussian bbox 중심을 기준으로 asset을 다시 정렬한다.
- `filter_asset()` 함수 추가
  - Stage1 Gaussian asset에서 배경/잔여 Gaussian을 제거하고, 실제 렌더링에 사용할 Gaussian만 남긴다.
- scale 계산 방식 수정
  - 기존에는 원본 PLY 전체 bbox 기준으로 scale을 계산했다.
  - 수정 후에는 필터링된 Gaussian bbox 기준으로 scale을 계산한다.

## 사용 예시

```powershell
python gaussian_initiailization/tools/render_stage2_gaussian_trajectory.py `
  --stage1_ply actual_dice_stage1_output/professor_dice_asset_stage1_30000/point_cloud/iteration_30000/point_cloud.ply `
  --trajectory actual_multi_dice_stage2_rollout/impulse_baseline_60f/stage2_predicted_trajectory.json `
  --output_dir actual_multi_dice_gaussian_render/impulse_baseline_60f_3dgs_gtcam_scale135 `
  --image_width 512 `
  --image_height 512 `
  --cam_distance 1.12 `
  --cam_height 0.66 `
  --cam_fovy_deg 40 `
  --white_background `
  --foreground_threshold 0.6 `
  --opacity_threshold 0.02 `
  --recenter_asset `
  --auto_scale_to_trajectory_half_extent `
  --scale_multiplier 1.35 `
  --fps 20
```

## 확인 결과

여러 개의 주사위 predicted trajectory를 Stage1 Gaussian 주사위 하나로 복제 렌더링하는 것이 가능함을 확인했다. 테스트셋 GT와 비교하기 위해 GT 카메라값에 맞추고, 화면상 크기 보정을 적용한 비교 GIF도 생성했다.

최종 비교 결과:

`actual_multi_dice_gaussian_render/impulse_baseline_60f_3dgs_gtcam_scale135_compare_testset/gt_vs_gaussian_multi_dice_size_matched.gif`

## 주의 사항

- 오른쪽 Gaussian 렌더는 Stage2 predicted trajectory를 렌더링한 것이므로 GT와 위치/회전이 완전히 같지는 않다.
- GT는 MuJoCo 렌더러 기반이라 바닥, 그림자, 반사가 포함되어 있고, Gaussian 렌더는 흰 배경으로 렌더링된다.
- 현재 색상은 주사위별로 따로 학습된 것이 아니라 하나의 Stage1 Gaussian 주사위 모델을 여러 개 복제한 결과다.
