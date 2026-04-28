# Dataset / Stage 1 / Stage 2 정리

이 문서는 이번 작업에서 정리한 세 가지를 설명한다.

- Stage 1 입력 asset 구성
- Stage 1 blur 문제와 수정 내용
- Stage 2 MuJoCo dataset 생성 흐름

기존 문서인 `README.md`, `EXPLAIN.md`, `EXPLAIN2.md`, `REPORT.md`는 그대로 두고, 이번 작업 내용만 따로 정리했다.

## 1. blur의 원인

Stage 1 결과가 뿌옇게 보이던 이유는 입력 asset과 학습 후반 구조 둘 다에 있었다.

입력 asset 쪽에서는 카메라가 너무 가깝고, cube 면 구분이 약하고, mask가 불안정한 경우가 있었다. 이 상태에서는 학습이 처음부터 흐릿한 기준을 받아들일 수밖에 없다.

학습 쪽에서는 densification이 후반까지 오래 이어질 수 있었다. 그러면 geometry가 늦게까지 흔들리고, appearance를 또렷하게 정리하는 시간이 줄어든다.

이번 수정은 이 두 부분을 같이 정리하는 방향으로 들어갔다.

- Stage 1 asset에서 프레이밍, 면 색, mask 생성을 정리했다.
- Stage 1 학습에서는 densification 종료 시점과 appearance refine 구간을 분리했다.

## 2. 경로 표기

- `<REPO_ROOT>`: 저장소 루트
- `<BLENDER_EXE>`: Blender 실행 파일
- `<SAM2_CHECKPOINT>`: SAM2 체크포인트 파일

예시:

```text
<REPO_ROOT>/gaussian_initiailization/output/stage1_cube_clean_asset
```

## 3. 수정한 파일

### `gaussian_initiailization/tools/render_object_views_blender.py`

Stage 1 입력 asset을 만드는 스크립트다.

이번에는 아래 내용을 추가하거나 정리했다.

- `face_palette` 재질 모드 추가
- box-like mesh에서 6면에 다른 색을 주는 처리 추가
- alpha 기반 mask가 맞지 않을 때 RGB 기반 fallback mask 추가
- 지금 쓰는 Stage 1 cube asset 세팅에 맞는 카메라 옵션 정리

이 파일은 실행하면 아래를 자동으로 만든다.

- `images/train`
- `images/test`
- `masks/train`
- `masks/test`
- `transforms_train.json`
- `transforms_test.json`
- `points3d.ply`
- `object_asset.json`
- `stage1_dataset_summary.json`

### `gaussian_initiailization/train.py`

Stage 1 학습 실행 파일이다.

이번에는 아래 내용을 추가하거나 정리했다.

- `--sg_gs_stage1` strict 모드 동작 정리
- geometry / appearance 분리 최적화 반영
- object mask prior 사용 경로 반영
- densification 종료 시점 제어 반영
- 후반 appearance refine 반영

핵심은 후반부를 계속 퍼뜨리는 방향이 아니라 정리하는 방향으로 바꾼 것이다.

### `gaussian_initiailization/arguments/__init__.py`

Stage 1 관련 실행 옵션 정의 파일이다.

이번에는 아래 옵션이 들어갔다.

- `--sg_gs_stage1`
- `--stage1_densify_ratio`
- `--stage1_appearance_refine`
- `--geometry_rgb_weight`
- `--require_sam_features`

### `gaussian_initiailization/render.py`

학습 결과 렌더링 파일이다.

이번에는 `foreground_threshold`가 없는 예전 설정 파일도 render가 되도록 fallback 처리를 넣었다.

## 4. 추가한 파일

### `gaussian_initiailization/tools/create_contactwm_stage2_layout.py`

Stage 2 dataset 폴더 구조를 만드는 스크립트다.

역할:

- `object_asset.json`을 받아 Stage 2 dataset 뼈대 생성

기능:

- `dataset_manifest.json` 생성
- `objects/...` 생성
- `stage2/.../episode_*` 폴더 생성

### `gaussian_initiailization/tools/generate_mujoco_fall_dataset.py`

Stage 2 MuJoCo trajectory dataset 생성 스크립트다.

역할:

- Stage 2 낙하 / 회전 / settle trajectory 생성

기능:

- 낙하 높이 설정
- 수평 속도 설정
- 회전 속도 설정
- damping 설정
- cube 면 색 설정
- segmentation 기반 mask 저장
- RGB / state / actions 저장

### `gaussian_initiailization/tools/preview_mujoco_fall.py`

생성된 trajectory를 MuJoCo viewer로 재생하는 스크립트다.

역할:

- dataset episode를 직접 확인하는 용도

기능:

- episode 선택
- 카메라 거리 / 높이 조절
- `--loop` 반복 재생
- `--hold_after` 종료 전 유지

### `gaussian_initiailization/tools/export_episode_gif.py`

episode 프레임을 GIF로 묶는 스크립트다.

역할:

- 공유용 미리보기 생성

기능:

- RGB 프레임을 GIF로 저장

### `gaussian_initiailization/tools/export_sibr_viewer_ply.py`

Stage 1 결과를 SIBR viewer용 PLY로 바꾸는 스크립트다.

역할:

- viewer용 변환

기능:

- Stage 1 결과 PLY를 SIBR에서 읽을 수 있는 형태로 저장

### `gaussian_initiailization/tools/test_cube.obj`

Stage 1 / Stage 2 테스트용 cube mesh다.

### `gaussian_initiailization/tools/test_sphere.obj`

Stage 2 bounce 실험용 sphere mesh다.

## 5. 필요한 항목

Python 환경에는 아래가 있어야 한다.

```bat
pip install mujoco
pip install imageio
```

기존 환경에는 아래가 이미 있어야 한다.

- `torch`
- 기존 3D Gaussian Splatting 의존성
- SAM2 실행 의존성

외부 코드와 체크포인트는 아래를 기준으로 한다.

```text
<REPO_ROOT>/external/sam2
<REPO_ROOT>/external/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

## 6. 실행 전 확인

```bat
conda activate gs
cd /d <REPO_ROOT>
```

필수 import 확인:

```bat
python -c "import torch; print('torch ok')"
python -c "import mujoco; print('mujoco ok')"
python -c "import imageio; print('imageio ok')"
```

파일 확인:

```bat
"<BLENDER_EXE>" --version
dir external\sam2
dir external\sam2\checkpoints
dir gaussian_initiailization\tools\test_cube.obj
```

## 7. 실행 순서

### 7.1 Stage 1 asset 생성

명령:

```bat
"%BLENDER_EXE%" --background --python gaussian_initiailization\tools\render_object_views_blender.py -- --mesh_path gaussian_initiailization\tools\test_cube.obj --output_path gaussian_initiailization\output\stage1_cube_clean_asset --object_name cube --physics_shape box --num_views 48 --test_hold 8 --resolution 448 --radius_scale 5.4 --lens 30 --elevation_min -18 --elevation_max 38 --point_count 30000 --material_mode face_palette
```

### 7.2 SAM2 feature 추출

```bat
set SAM2_CHECKPOINT=<SAM2_CHECKPOINT>
set PYTHONPATH=%CD%\external\sam2
python gaussian_initiailization\extract_sam2_features.py --source_path gaussian_initiailization\output\stage1_cube_clean_asset --output_dir sam_features --checkpoint "%SAM2_CHECKPOINT%" --config configs/sam2.1/sam2.1_hiera_t.yaml --feature_source high_res0 --output_channels 9 --splits train test
```

### 7.3 Stage 1 학습

```bat
python gaussian_initiailization\train.py --source_path gaussian_initiailization\output\stage1_cube_clean_asset --model_path gaussian_initiailization\output\stage1_cube_clean_train_mask_8k --images images --masks_dir masks --sam_features sam_features --geometry_feature_dim 9 --sam_feature_weight 0.05 --sg_gs_stage1 --object_mask_weight 0.5 --object_mask_bce_weight 3.0 --iterations 8000 --test_iterations 4000 8000 --save_iterations 4000 8000 --checkpoint_iterations 4000 8000 --eval --disable_viewer
```

### 7.4 결과 렌더링

4000 iteration:

```bat
python gaussian_initiailization\render.py --model_path gaussian_initiailization\output\stage1_cube_clean_train_mask_8k --source_path gaussian_initiailization\output\stage1_cube_clean_asset --images images --masks_dir masks --sam_features sam_features --geometry_feature_dim 9 --iteration 4000
```

8000 iteration:

```bat
python gaussian_initiailization\render.py --model_path gaussian_initiailization\output\stage1_cube_clean_train_mask_8k --source_path gaussian_initiailization\output\stage1_cube_clean_asset --images images --masks_dir masks --sam_features sam_features --geometry_feature_dim 9 --iteration 8000
```

### 7.5 Stage 2 dataset 뼈대 생성

```bat
python gaussian_initiailization\tools\create_contactwm_stage2_layout.py --dataset_root gaussian_initiailization\output\stage2_cube_dynamic_dataset --object_asset gaussian_initiailization\output\stage1_cube_clean_asset\object_asset.json --scenario fall_and_rebound --train_episodes 1 --test_episodes 2
```

### 7.6 MuJoCo dataset 생성

```bat
python gaussian_initiailization\tools\generate_mujoco_fall_dataset.py --dataset_root gaussian_initiailization\output\stage2_cube_dynamic_dataset --object_name cube --split all --camera_distance 8.5 --camera_height 4.6 --train_drop_height 2.8 --test_drop_height 3.4 --planar_speed_train 0.55 --planar_speed_test 0.75 --spin_speed_train 4.5 --spin_speed_test 6.0 --freejoint_damping 0.12 --box_face_colors "#f15a4a,#f4a261,#e9c46a,#2a9d8f,#4f86f7,#9b5de5"
```

### 7.7 GIF 생성

```bat
python gaussian_initiailization\tools\export_episode_gif.py --episode_dir gaussian_initiailization\output\stage2_cube_dynamic_dataset\stage2\fall_and_rebound\train\cube\episode_000 --output_name cube_dynamic_episode_000.gif --fps 20
```

### 7.8 MuJoCo viewer 반복 재생

```bat
python gaussian_initiailization\tools\preview_mujoco_fall.py --dataset_root gaussian_initiailization\output\stage2_cube_dynamic_dataset --object_name cube --split train --episode_index 0 --camera_distance 8.5 --camera_height 4.6 --freejoint_damping 0.12 --loop
```

### 7.9 SIBR viewer용 변환

선택 단계다.

```bat
python gaussian_initiailization\tools\export_sibr_viewer_ply.py --input gaussian_initiailization\output\stage1_cube_clean_train_mask_8k\point_cloud\iteration_8000\point_cloud.ply --output gaussian_initiailization\output\stage1_cube_clean_train_mask_8k_sibr\point_cloud\iteration_8000\point_cloud.ply
```

SIBR viewer 실행:

```bat
"gaussian_initiailization\SIBR_viewers\install\bin\SIBR_gaussianViewer_app_rwdi.exe" -m "gaussian_initiailization\output\stage1_cube_clean_train_mask_8k_sibr" --iteration 8000
```

## 8. 현재 저장 구조

```text
gaussian_initiailization/output/
  stage1_cube_clean_asset/
  stage1_cube_clean_train_mask_8k/
  stage2_cube_dynamic_dataset/
```

### `stage1_cube_clean_asset`

```text
stage1_cube_clean_asset/
  images/
    train/
    test/
  masks/
    train/
    test/
  sparse/
    0/
  transforms_train.json
  transforms_test.json
  points3d.ply
  object_asset.json
  stage1_dataset_summary.json
  sam_features/
```

### `stage1_cube_clean_train_mask_8k`

```text
stage1_cube_clean_train_mask_8k/
  cfg_args
  training_args.json
  cameras.json
  exposure.json
  input.ply
  chkpnt4000.pth
  chkpnt8000.pth
  point_cloud/
    iteration_4000/
    iteration_8000/
  train/
  test/
```

### `stage2_cube_dynamic_dataset`

```text
stage2_cube_dynamic_dataset/
  dataset_manifest.json
  objects/
    cube/
      object_manifest.json
  stage2/
    fall_and_rebound/
      scenario_manifest.json
      train/
        cube/
          episode_000/
            episode_manifest.json
            rgb/
            masks/
            state/
              trajectory.json
            actions/
              trajectory.json
            cube_dynamic_episode_000.gif
      test/
        cube/
          episode_000/
          episode_001/
```

## 9. 정리

- Stage 1 입력 asset은 cube 형체와 면 방향이 보이도록 정리되어 있다.
- Stage 1 학습은 후반 blur를 줄이는 방향으로 설정되어 있다.
- Stage 2 dataset은 Stage 1 `object_asset.json`을 받아 MuJoCo trajectory를 만든다.
- 결과는 GIF, viewer, SIBR 변환 경로까지 포함해 바로 확인할 수 있다.
