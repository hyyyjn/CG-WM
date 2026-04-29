# Gaussian Initialization in CG-WM

이 디렉터리는 CG-WM의 scene initialization 실험 코드를 담고 있습니다.
기반은 Inria의 3D Gaussian Splatting이며, 현재 코드는 ContactGaussian-WM 방향의 초기화 실험을 위해 아래 기능들이 추가된 상태입니다.

- isotropic spherical Gaussian 제약
- geometry / appearance decoupled optimization
- geometry / appearance loss 분리
- image-wise exposure optimization
- SAM2 feature supervision 경로
- richer SAM feature supervision
- object mask prior supervision
- per-Gaussian learned foreground score
- foreground-aware debug / thresholded render

폴더 이름은 현재 저장소 기준으로 `gaussian_initiailization`입니다.

## 현재 구현 상태

현재 코드에서 실제로 동작하는 핵심 기능은 다음과 같습니다.

- 기본 3DGS 학습, checkpoint 저장, render, metric 계산
- isotropic Gaussian 제약
  - scale 3축 평균 사용
  - rotation은 identity quaternion으로 고정
- geometry / appearance optimizer 분리
  - alternating optimization
  - joint optimization
- geometry / appearance loss 분리
  - appearance: RGB reconstruction 중심
  - geometry: reconstruction + regularization + optional depth + optional SAM2 feature loss + optional object mask loss
- exposure optimizer 분리 및 checkpoint 복구
- 구 decoupled checkpoint 호환 resume
- SAM2 feature map `.npy` 로딩 및 geometry step supervision
- configurable geometry feature dimension (`--geometry_feature_dim`)
- configurable SAM feature extraction channel count (`--output_channels`)
- `--disable_viewer` 사용 시 viewer import 실패 fallback
- Gaussian별 `object_id` 저장 / 복구 / PLY round-trip
- Gaussian별 `foreground_logit` 저장 / 복구 / PLY round-trip
- object-only render (`--object_id`)
- foreground-threshold render (`--foreground_threshold`)
- foreground score / overlay / object mask prior debug render
- 수동 / 자동 object grouping 스크립트
- foreground object mask 자동 추출 스크립트
- physics stage 연결용 intermediate export
- rigid-friendly body / collision / mass metadata export
- 외부 instance segmentation 결과 normalize helper
- densification statistics logging (`densification_stats.jsonl`)
- variant / densification 비교용 `compare_variants.py`

현재 저장소 기준으로는 이 디렉터리가 "scene initialization + object-aware separation 실험 + object grouping + rigid-friendly export"까지 담당합니다.
즉 rigid-body dynamics 자체는 아직 없지만, 그 직전 단계까지는 코드와 입출력 경로가 연결된 상태입니다.
최근 구현의 중심은 `grouping`보다는 `train.py` 내부의 object-aware supervision과 `gaussian_model.py`의 foreground score입니다.

## 코드 구조

- `train.py`
  - 학습 진입점
  - decoupled optimization, geometry/apppearance loss 계산, densification/pruning 수행
- `render.py`
  - 저장된 Gaussian으로 train/test 뷰 렌더링
- `metrics.py`
  - PSNR / SSIM / LPIPS 평가
- `extract_sam2_features.py`
  - SAM2 image encoder feature를 configurable channel count `.npy`로 저장하는 전처리 스크립트
- `assign_object_ids.py`
  - 외부 object id 배열을 Gaussian에 붙여 grouped model로 저장
- `auto_assign_object_ids.py`
  - 2D instance mask를 multi-view voting으로 Gaussian `object_id`에 자동 할당
- `extract_object_masks.py`
  - 이미지에서 foreground object mask를 자동 추출
- `prepare_instance_masks.py`
  - 외부 segmentation 결과를 `auto_assign_object_ids.py` 입력 형식으로 normalize
- `estimate_masked_colmap.py`
  - foreground mask를 반영한 입력 이미지로 COLMAP pose를 다시 추정
- `build_visual_hull.py`
  - multi-view mask와 camera로 visual hull seed point cloud를 생성
- `run_scene_initialization_pipeline.py`
  - mask 추출, masked COLMAP, visual hull, SAM2, train 단계를 순차 실행
- `export_physics_scene.py`
  - grouped Gaussian scene을 physics stage용 JSON / NPZ와 rigid metadata로 export
- `arguments/`
  - CLI 인자 정의
- `scene/`
  - dataset loading, camera 구성, GaussianModel 정의
- `gaussian_renderer/`
  - CUDA rasterizer 호출

## 핵심 구현 포인트

### 0. Visual Hull Seed Initialization

초기 Gaussian seed를 외부 PLY로 주입할 수 있습니다.

- `build_visual_hull.py`가 `images + masks + cameras`에서 `visual_hull/visual_hull.ply`를 생성합니다.
- 학습 시 `--init_mode visual_hull`를 주면 이 seed를 우선 사용합니다.
- 다른 PLY를 직접 쓰고 싶으면 `--init_ply_path`를 줄 수 있습니다.

예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/build_visual_hull.py \
  --source_path /path/to/scene \
  --masks_dir /path/to/masks \
  --grid_resolution 128
```

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/scene_vh \
  --init_mode visual_hull \
  --eval \
  --disable_viewer
```

### 0.1 Mask-Aware COLMAP Pose Estimation

foreground mask를 이용해 background를 제거한 이미지로 COLMAP pose를 다시 추정할 수 있습니다.

- 입력 이미지는 `masked_colmap/input` 아래에 생성됩니다.
- COLMAP 결과는 최종적으로 `source_path/sparse/0`에 반영됩니다.
- 기존 `source_path/sparse`가 있으면 덮어쓰게 되므로 필요하면 미리 백업하세요.

예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/estimate_masked_colmap.py \
  --source_path /path/to/scene \
  --masks_dir /path/to/masks \
  --background_mode white \
  --mask_dilate 5 \
  --overwrite
```

### 0.2 End-to-End Orchestration

한 번에 전체 초기화 파이프라인을 돌리고 싶으면 orchestration 스크립트를 사용할 수 있습니다.

기본 흐름:

- mask 추출
- masked COLMAP
- visual hull seed 생성
- SAM2 feature 추출
- visual hull seed로 SG-GS 학습

예시:

```bash
python gaussian_initiailization/run_scene_initialization_pipeline.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/scene_pipeline \
  --iterations 10000 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --alternating_optimization
```

이미 준비된 단계를 건너뛰고 싶으면:

- `--skip_mask_extraction`
- `--skip_masked_colmap`
- `--skip_visual_hull`
- `--skip_sam2`
- `--skip_train`

### 0.3 MuJoCo Synthetic Dataset Quickstart

간단한 box / sphere / cylinder 장면을 MuJoCo로 렌더해서
현재 scene initialization 코드가 바로 읽을 수 있는 synthetic 데이터셋을 만들 수 있습니다.

생성 스크립트 출력 형식:

- `images/train/*.png`
- `images/test/*.png`
- `masks/train/*.png`
- `masks/test/*.png`
- `transforms_train.json`
- `transforms_test.json`

예시:

```bash
MUJOCO_GL=egl conda run -n mujoco python gaussian_initiailization/generate_mujoco_synthetic_dataset.py \
  --output_root gaussian_initiailization/output/mujoco_data \
  --scene_name box_demo \
  --object_type box \
  --train_views 24 \
  --test_views 8
```

headless 환경에서 `egl`이 안 되면 `MUJOCO_GL=osmesa`도 시도해볼 수 있습니다.

그 다음 visual hull 기반 초기화 테스트:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/build_visual_hull.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_demo \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_demo/masks \
  --grid_resolution 96
```

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_demo \
  --model_path gaussian_initiailization/output/mujoco_box_demo_init \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_demo/masks \
  --init_mode visual_hull \
  --iterations 3000 \
  --object_mask_weight 0.1 \
  --eval \
  --disable_viewer
```

synthetic 데이터셋에는 COLMAP이 없으므로 `run_scene_initialization_pipeline.py`를 그대로 쓸 때는
최소한 아래 옵션으로 COLMAP 관련 단계를 건너뛰는 편이 안전합니다.

```bash
python gaussian_initiailization/run_scene_initialization_pipeline.py \
  --source_path gaussian_initiailization/output/mujoco_data/box_demo \
  --model_path gaussian_initiailization/output/mujoco_box_demo_pipeline \
  --masks_dir gaussian_initiailization/output/mujoco_data/box_demo/masks \
  --skip_mask_extraction \
  --skip_masked_colmap \
  --skip_sam2 \
  --iterations 3000
```

### 1. Isotropic Gaussian

`scene/gaussian_model.py`에서 anisotropic Gaussian을 그대로 쓰지 않습니다.

- `get_scaling()`은 3축 scale 평균을 사용합니다.
- `get_rotation()`은 identity quaternion을 반환합니다.
- densification 이후 새 Gaussian도 같은 제약을 유지합니다.

즉 현재 scene initialization은 회전이 없는 spherical Gaussian 표현 위에서 학습됩니다.

### 2. Decoupled optimization

현재는 geometry와 appearance를 optimizer 수준에서 분리해서 학습할 수 있습니다.

- geometry parameters
  - `xyz`
  - `scaling`
  - `rotation`
  - `features_geo`
- appearance parameters
  - `features_dc`
  - `features_rest`
  - `opacity`
- exposure parameters
  - per-image exposure matrix

지원 모드:

- `--alternating_optimization`
  - geometry step과 appearance step을 번갈아 수행
- `--joint_optimization`
  - 한 iteration에서 geometry step과 appearance step을 모두 수행

관련 옵션:

- `--geometry_iters`
- `--appearance_iters`

### 3. Loss 분리

현재 `train.py`는 geometry step과 appearance step에서 별도 forward/backward를 수행합니다.

- appearance loss
  - L1
  - DSSIM
- geometry loss
  - reconstruction term
  - optional depth loss
  - scale regularization
  - optional SAM2 feature loss
  - optional object mask BCE + L1 loss
- appearance regularization
  - opacity regularization

로그에는 아래 항목이 분리되어 기록됩니다.

- `geometry_loss`
- `appearance_loss`
- `geometry_feature_loss`
- `object_mask_loss`

### 3.1 Object-Aware Geometry Supervision

최근 구현에서는 object separation을 post-hoc grouping에만 맡기지 않고,
geometry step 안으로 일부 끌어왔습니다.

- `--masks_dir`를 주면 per-view object mask prior를 camera에 실어 학습합니다.
- mask가 없고 RGBA 입력이면 alpha 채널을 object mask prior로 fallback 사용합니다.
- `train.py`의 `compute_object_mask_loss`가 Gaussian foreground score를 렌더해 target mask와 BCE + L1로 맞춥니다.
- 이 foreground score는 `scene/gaussian_model.py`의 per-Gaussian `foreground_logit`에서 나옵니다.

즉 현재 object separation은 아직 완전한 instance-level stage는 아니지만,
foreground/background-aware scene initialization으로는 이미 학습 내부에 들어와 있습니다.

### 4. SAM2 feature supervision

SAM2 feature supervision은 현재 구현되어 있습니다.

- dataset 아래 `sam_features/.../*.npy`를 읽습니다.
- Blender synthetic 데이터셋에서는 split-aware 경로를 지원합니다.
  - `sam_features/train/*.npy`
  - `sam_features/test/*.npy`
  - `sam_features/val/*.npy`
- geometry step에서 Gaussian의 geometry feature를 렌더한 뒤 target feature와 L1로 비교합니다.
- geometry feature는 현재 multi-chunk supervision을 지원합니다.
  - 즉 3채널만 쓰는 대신, 더 높은 feature 차원을 3채널 chunk로 나눠 여러 번 render/loss를 계산할 수 있습니다.
- 현재 구현에서는 geometry RGB render와 geometry feature render가 같은 screen-space gradient를 공유하도록 연결해,
  SAM feature loss가 densification 통계에도 간접적으로 반영되도록 수정했습니다.
- `--sam_feature_normalization`으로 feature preprocessing을 제어할 수 있습니다.
  - `none`
    - 기본값
    - per-view min-max normalization 없이 원본 feature scale을 유지
  - `per_view_minmax`
    - 예전 동작 재현용
  - `clip_0_1`
    - feature 값을 0~1 범위로만 제한

현재 loader는 예전 flat 경로도 fallback으로 지원합니다.

## 환경 구성

이 프로젝트는 현재 두 환경으로 나눠 쓰는 것이 가장 안정적입니다.

### 1. `gaussian_splatting`

학습과 렌더용 환경입니다.

사용:

- `train.py`
- `render.py`
- `metrics.py`

이 환경에는 기존 3DGS CUDA extension들이 설치되어 있어야 합니다.

### 2. `sam2cpu`

SAM2 feature extraction용 환경입니다.

사용:

- `extract_sam2_features.py`
- official SAM2 repo

이 환경은 Python 3.10 + SAM2 + PyTorch 2.5.1 CPU 조합으로 구성했습니다.

중요:

- 이 둘이 다른 conda 환경인 것은 GitHub 업로드 문제와는 별개입니다.
- 중요한 것은 코드를 커밋하는 것이고, 환경은 README와 `environment.yml` 또는 설치 절차로 문서화하면 됩니다.
- 생성된 `sam_features_sam2` 같은 산출물은 보통 Git에 올리지 않습니다.

## 환경 설정 방법

아래 절차는 이 저장소를 새로 받은 뒤 재현하는 기준입니다.

### 1. 저장소 준비

프로젝트 루트로 이동합니다.

```bash
cd /path/to/CG-WM
```

이 README는 `gaussian_initiailization` 디렉터리 기준으로 설명합니다.

### 2. `gaussian_splatting` 환경 만들기

학습과 렌더는 이 환경에서 수행합니다.

기본 방법:

```bash
conda env create -f gaussian_initiailization/environment.yml
```

활성화:

```bash
conda activate gaussian_splatting
```

주의:

- 이 환경은 Python 3.7 / PyTorch 1.12.1 / CUDA extension 기준입니다.
- `diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`이 함께 설치됩니다.
- CUDA가 잡히지 않거나 extension build에 실패하면, CUDA toolkit / compiler 환경을 먼저 확인해야 합니다.

환경이 이미 만들어졌는데 extension만 다시 설치해야 하면:

```bash
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/diff-gaussian-rasterization
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/simple-knn
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/fused-ssim
```

### 3. `sam2cpu` 환경 만들기

SAM2 feature extraction은 별도 Python 3.10 환경을 권장합니다.

환경 생성:

```bash
conda create -n sam2cpu python=3.10 -y
```

PyTorch CPU 설치:

```bash
conda run -n sam2cpu pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
```

### 4. SAM2 설치

official SAM2 repo를 별도로 받습니다.

```bash
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2_repo
conda run -n sam2cpu pip install -e /path/to/sam2_repo
```

체크포인트도 필요합니다. 예시는 SAM2.1 tiny입니다.

```bash
wget -O /path/to/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

현재 [extract_sam2_features.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_sam2_features.py)는 기본값으로 아래 경로를 사용합니다.

- `/home/cgr-ugrad-2026/work/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt`

다른 경로를 쓰면 `--checkpoint`로 직접 넘기면 됩니다.

### 5. 환경 확인

`gaussian_splatting` 확인:

```bash
conda run -n gaussian_splatting python -c "import torch; print(torch.__version__)"
```

`sam2cpu` 확인:

```bash
conda run -n sam2cpu python -c "import torch, sam2; print(torch.__version__); print('sam2 ok')"
```

### 6. 왜 환경을 나눴는가

이 프로젝트는 현재 다음 이유로 환경을 분리하는 편이 안전합니다.

- 3DGS 학습 코드는 오래된 PyTorch / CUDA extension 조합에 맞춰져 있습니다.
- SAM2는 Python 3.10 이상, PyTorch 2.5.1 이상 조합을 요구합니다.
- 두 스택을 한 conda 환경에 강제로 합치면 충돌 가능성이 큽니다.

따라서 현재 권장 방식은:

- feature extraction: `sam2cpu`
- training / render: `gaussian_splatting`

입니다.

## 실제 실행 절차

### 0. 현재 기준 object-aware 흐름 요약

지금 `main` 브랜치 기준으로 가장 대표적인 흐름은 아래입니다.

1. foreground mask 준비
   - 이미 있으면 `--masks_dir`로 사용
   - 없으면 `extract_object_masks.py`로 생성
2. SAM2 feature 준비
   - `extract_sam2_features.py`로 `sam_features_sam2/` 생성
3. object-aware Stage 1 학습
   - `--sg_gs_stage1`
   - `--sam_feature_weight`
   - `--object_mask_weight`
   - `--masks_dir`
4. debug render 확인
   - `renders`
   - `foreground_scores`
   - `foreground_overlay`
   - `object_mask_prior`
5. 필요하면 `--foreground_threshold`로 tighter render 생성

현재 구현에서 object-aware 성분은 `mask prior -> object mask loss -> foreground score -> thresholded render`
경로를 통해 가장 직접적으로 반영됩니다.

즉 지금 단계에서는 `mask 품질`의 영향이 꽤 큰 편입니다.

### 1. SAM2 feature 추출

먼저 SAM2 feature map을 만듭니다.

```bash
conda run -n sam2cpu python gaussian_initiailization/extract_sam2_features.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --output_dir sam_features_sam2 \
  --output_channels 9 \
  --splits train test val
```

출력:

- `source_path/sam_features_sam2/train/*.npy`
- `source_path/sam_features_sam2/test/*.npy`
- `source_path/sam_features_sam2/val/*.npy`

### 2. 학습

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/lego_sam2_10k \
  --iterations 10000 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --alternating_optimization \
  --geometry_iters 1 \
  --appearance_iters 1
```

object-aware 설정까지 같이 켜는 권장 예시는 아래와 같습니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --masks_dir /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego/auto_masks \
  --object_mask_weight 0.1 \
  --model_path gaussian_initiailization/output/lego_objaware_mask_10k \
  --iterations 10000 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --sg_gs_stage1
```

현재 `lego` 기준으로 실제 검증에 쓴 최신 명령은 아래입니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/main_objaware_lego_10k \
  --iterations 10000 \
  --resolution 8 \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --masks_dir /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego/auto_masks \
  --object_mask_weight 0.1 \
  --eval \
  --disable_viewer \
  --quiet \
  --sg_gs_stage1 \
  --test_iterations -1 \
  --save_iterations 10000
```

### 3. 렌더

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/lego_sam2_10k \
  --iteration 10000 \
  --skip_train \
  --resolution 4 \
  --eval \
  --quiet
```

특정 grouped object만 따로 보고 싶으면 `--object_id`를 사용할 수 있습니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/your_model_grouped \
  --iteration 10000 \
  --skip_train \
  --eval \
  --object_id 1
```

이 경우 출력은 `test_object_1/ours_<iter>/renders` 아래에 저장됩니다.

foreground score를 더 엄격히 잘라서 halo를 줄이고 싶으면 `--foreground_threshold`를 사용할 수 있습니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/lego_objaware_mask_10k \
  --iteration 10000 \
  --skip_train \
  --resolution 8 \
  --eval \
  --quiet \
  --geometry_feature_dim 9 \
  --masks_dir /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego/auto_masks \
  --foreground_threshold 0.5
```

이 경우 출력은 `test_fgthr_0p5/ours_<iter>/...` 아래에 저장됩니다.

현재 코드에서는 `cfg_args` 병합 과정 때문에 `foreground_threshold`를 명시하지 않으면
일부 환경에서 기본값이 빠지는 경우가 있습니다.
그래서 전체 Gaussian을 포함한 기본 object-aware render도 아래처럼 `0.0`을 명시하는 것을 권장합니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/main_objaware_lego_10k \
  --iteration 10000 \
  --skip_train \
  --resolution 8 \
  --eval \
  --quiet \
  --geometry_feature_dim 9 \
  --masks_dir /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego/auto_masks \
  --foreground_threshold 0.0
```

즉 현재 권장 비교는 아래 두 결과입니다.

- `test_fgthr_0p0`
  - object-aware 학습 결과 전체
- `test_fgthr_0p5`
  - foreground score가 낮은 Gaussian을 제거한 tighter render

## 최근 검증 결과

실제로 아래가 확인되었습니다.

- `lego` 데이터셋에서 10,000 iteration 학습 성공
- SAM2 feature supervision을 켠 2-iteration smoke test 성공
- `baseline`, `joint`, `alternating` 짧은 CUDA smoke test 성공
- `lego_sam2_10k` 모델 render 성공
- test view 200장 저장 확인
- richer feature supervision smoke test 성공
  - `geometry_feature_dim=3`, `20 iter` 학습 + render
  - `geometry_feature_dim=9`, `20 iter` 학습 + render
  - `geometry_feature_dim=9`, `650 iter` 학습 + densification log + render
- richer feature supervision 5k 비교 실험 성공
  - `geometry_feature_dim=3`, `5000 iter` 학습 + render
  - `geometry_feature_dim=9`, `5000 iter` 학습 + render
  - 두 실험 모두 train 100장 / test 200장 render 저장 확인
- 현재 코드가 직접 만든 decoupled checkpoint self-resume 성공
- 예전 decoupled checkpoint에서 빠져 있던 `f_geo` optimizer group을 이름 기반 fallback으로 복구 가능
- `--disable_viewer`를 줬을 때 socket permission 문제 없이 학습 시작 가능
- `lego_objaware_mask_10k` 10,000 iteration object-aware 학습 성공
- `foreground_scores`, `object_mask_prior`, `foreground_overlay` debug render 저장 확인
- `--foreground_threshold 0.5` 기반 thresholded render 생성 확인
- `main_objaware_lego_10k` 10,000 iteration object-aware 학습 성공
- `main_objaware_lego_10k/test_fgthr_0p0` 전체 object-aware render 확인
- `main_objaware_lego_10k/test_fgthr_0p5` tighter render 확인

예시 결과:

- 모델 출력: `gaussian_initiailization/output/lego_sam2_10k`
- render 출력: `gaussian_initiailization/output/lego_sam2_10k/test/ours_10000/renders`
- 3채널 5k 출력: `gaussian_initiailization/output/test_geo3_5k`
- 9채널 5k 출력: `gaussian_initiailization/output/test_geo9_5k`
- variant 비교 smoke JSON: `gaussian_initiailization/output/test_geo_compare_smoke.json`
- object-aware 10k 모델: `gaussian_initiailization/output/lego_objaware_mask_10k`
- object-aware render: `gaussian_initiailization/output/lego_objaware_mask_10k/test/ours_10000`
- thresholded render: `gaussian_initiailization/output/lego_objaware_mask_10k/test_fgthr_0p5/ours_10000`
- 최신 main 기준 object-aware 10k 모델: `gaussian_initiailization/output/main_objaware_lego_10k`
- 최신 main 기준 전체 render: `gaussian_initiailization/output/main_objaware_lego_10k/test_fgthr_0p0/ours_10000`
- 최신 main 기준 thresholded render: `gaussian_initiailization/output/main_objaware_lego_10k/test_fgthr_0p5/ours_10000`

## 현재 코드에 있는 주요 옵션

### 데이터 / 입출력

- `--source_path`
- `--model_path`
- `--images`
- `--depths`
- `--sam_features`
- `--masks_dir`
- `--geometry_feature_dim`
- `--sam_feature_normalization`
- `--resolution`
- `--white_background`
- `--eval`
- `--disable_viewer`
- `--object_id`
- `--foreground_threshold`

### decoupled optimization

- `--alternating_optimization`
- `--joint_optimization`
- `--geometry_iters`
- `--appearance_iters`

### geometry supervision

- `--sam_feature_weight`
- `--object_mask_weight`
- `--object_mask_bce_weight`
- `--depth_l1_weight_init`
- `--depth_l1_weight_final`

### optimization / render

- `--iterations`
- `--checkpoint_iterations`
- `--save_iterations`
- `--start_checkpoint`
- `--convert_SHs_python`
- `--compute_cov3D_python`
- `--antialiasing`

## 체크포인트

현재 checkpoint는 decoupled optimization 상태를 포함해 저장/복구합니다.

포함 항목:

- Gaussian parameter tensors
- geometry optimizer state
- appearance optimizer state
- exposure optimizer state
- geometry feature tensor
- foreground logit tensor
- densification 관련 통계

이전 tuple 형식 checkpoint에 대한 backward compatibility도 일부 유지합니다.

추가로 현재는 예전 decoupled checkpoint에 geometry optimizer의 `f_geo` group이 없는 경우에도,
parameter group 이름 기준으로 가능한 state만 복구하도록 fallback을 넣었습니다.
즉, 예전 checkpoint는 완전 동일 복원이 아니라 "호환 가능한 optimizer state 복구 + 없는 group은 현재 초기화값 사용" 방식으로 resume됩니다.

## Viewer 관련 주의

기존 3DGS viewer는 import 시점에 socket을 여는 구조라 제한된 환경에서 바로 실패할 수 있습니다.

현재 `train.py`는 다음처럼 동작합니다.

- `--disable_viewer`를 주면 viewer를 사용하지 않고 바로 학습합니다.
- viewer import 자체가 실패해도 no-op fallback으로 처리합니다.
- 따라서 원격/샌드박스/권한 제한 환경에서도 training smoke test를 돌릴 수 있습니다.

## Object Grouping과 Physics Export

현재는 initialization 결과를 object-level representation으로 넘기기 위한 최소 파이프라인이 구현되어 있습니다.
다만 논문 정렬 관점에서는 `grouping`보다 `object-aware initialization`이 더 핵심이고,
최근 구현도 그 방향으로 이동했습니다.

- Gaussian별 `object_id`가 checkpoint와 PLY에 함께 저장됩니다.
- `assign_object_ids.py`로 외부에서 만든 object id 배열을 붙일 수 있습니다.
- `auto_assign_object_ids.py`로 2D instance mask에서 자동 grouping할 수 있습니다.
- `extract_object_masks.py`로 이미지에서 foreground object mask를 자동 추출할 수 있습니다.
- `export_physics_scene.py`로 object별 Gaussian subset과 요약 통계를 export할 수 있습니다.

`auto_assign_object_ids.py`의 현재 기능:

- multi-view voting
- occlusion-aware frontmost filtering
- ignore label 지원
- confidence-weighted voting
- boundary downweighting
- multi-view consistency refinement

physics export 산출물:

- `physics_scene.json`
- `physics_scene_arrays.npz`

각 object summary에는 현재 다음 정보가 포함됩니다.

- `object_id`
- `source_track_id`
- `is_background`
- center / radius / bbox / mean scale / mean opacity
- body frame
- sphere + AABB collision proxy
- default mass / density / friction / restitution / inertia

array bundle에도 object별 `source_track_ids`가 함께 저장됩니다.

즉 지금은 initialization 결과를 바로 rigid grouping 실험과 physics stage 입구까지 넘길 수 있는 상태입니다.

## 자동 Object Mask 추출

외부 segmentation 모델이 없더라도, 현재는 이미지에서 foreground object mask를 자동으로 뽑는 보조 경로가 있습니다.

- RGBA 이미지면 alpha 채널 기반 foreground mask 추출
- 일반 RGB 이미지면 background subtraction 기반 foreground mask 추출

예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/extract_object_masks.py \
  --source_path /path/to/scene \
  --output_masks_dir /path/to/auto_masks \
  --output_confidence_dir /path/to/auto_confidence \
  --method auto
```

배경색이 거의 고정된 장면이라면 `bg_subtract`를 명시할 수도 있습니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/extract_object_masks.py \
  --source_path /path/to/scene \
  --output_masks_dir /path/to/auto_masks \
  --output_confidence_dir /path/to/auto_confidence \
  --method bg_subtract \
  --background_hint white \
  --diff_threshold 0.12
```

이 스크립트의 출력은 바로 `auto_assign_object_ids.py` 입력으로 사용할 수 있습니다.

## 외부 Instance Mask 연동

외부 segmentation 모델의 출력 형식이 바로 맞지 않아도, 먼저 `prepare_instance_masks.py`로 normalize한 뒤
`auto_assign_object_ids.py`에 연결하면 됩니다.

권장 순서는 아래와 같습니다.

1. object mask 준비

방법 A. 외부 segmentation 결과 normalize

```bash
conda run -n gaussian_splatting python gaussian_initiailization/prepare_instance_masks.py \
  --source_path /path/to/scene \
  --input_masks_dir /path/to/external_masks \
  --output_masks_dir /path/to/prepared_masks \
  --input_confidence_dir /path/to/external_confidence \
  --output_confidence_dir /path/to/prepared_confidence \
  --overwrite
```

방법 B. scene 이미지에서 foreground mask 자동 추출

```bash
conda run -n gaussian_splatting python gaussian_initiailization/extract_object_masks.py \
  --source_path /path/to/scene \
  --output_masks_dir /path/to/prepared_masks \
  --output_confidence_dir /path/to/prepared_confidence \
  --method auto
```

2. normalized mask로 Gaussian object id 자동 할당

```bash
conda run -n gaussian_splatting python gaussian_initiailization/auto_assign_object_ids.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/your_model \
  --iteration 10000 \
  --masks_dir /path/to/prepared_masks \
  --confidence_maps_dir /path/to/prepared_confidence \
  --confidence_threshold 0.5 \
  --output_model_path gaussian_initiailization/output/your_model_grouped
```

3. grouped scene export

```bash
conda run -n gaussian_splatting python gaussian_initiailization/export_physics_scene.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/your_model_grouped \
  --iteration 10000 \
  --density 1000 \
  --friction 0.5 \
  --restitution 0.1 \
  --output_dir gaussian_initiailization/output/your_model_grouped/physics_export
```

`export_physics_scene.py`의 rigid metadata는 현재 "physics stage 입구용 기본값"입니다.
즉 이후 differentiable rigid-body stage에서 더 정확한 mass / contact parameter로 refinement 하는 것을 전제로 합니다.

`prepare_instance_masks.py`는 현재 다음 입력을 지원합니다.

- flat 구조
  - `/path/to/external_masks/r_1.npy`
  - `/path/to/external_masks/r_1.png`
- split-aware 구조
  - `/path/to/external_masks/test/r_1.npy`
  - `/path/to/external_masks/train/r_10.png`

confidence map도 같은 규칙으로 찾습니다.

추가로 `--input_format`으로 아래 adapter를 바로 사용할 수 있습니다.

- `--input_format generic`
  - 기존 label map / indexed png / rgb-colored instance map
- `--input_format sam2`
  - frame별 디렉터리 안에 여러 binary mask가 쌓인 구조
  - 예: `test/r_1/mask_0.npy`, `test/r_1/mask_1.png`
  - sidecar JSON에 `predicted_iou`, `score`, `confidence`가 있으면 confidence weight로 반영
- `--input_format mask2former`
  - per-pixel label map
  - 또는 `(C,H,W)` / `(H,W,C)` logits, probability tensor
  - 이 경우 argmax label map과 max probability confidence map을 자동 생성
- `--input_format deva`
  - `Annotations/`, `masks/`, `masklets/` 같은 추적 결과 디렉터리 검색
  - frame별 stacked binary mask 디렉터리나 indexed label map 둘 다 지원
  - sidecar JSON의 `track_id` / `object_id` / `instance_id` / `id`를 읽어 track label을 그대로 유지
  - JSON이 없으면 파일명 숫자를 fallback track id로 사용

예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/prepare_instance_masks.py \
  --source_path /path/to/scene \
  --input_masks_dir /path/to/sam2_outputs \
  --output_masks_dir /path/to/prepared_masks \
  --output_confidence_dir /path/to/prepared_confidence \
  --input_format sam2 \
  --overwrite
```

```bash
conda run -n gaussian_splatting python gaussian_initiailization/prepare_instance_masks.py \
  --source_path /path/to/scene \
  --input_masks_dir /path/to/mask2former_outputs \
  --output_masks_dir /path/to/prepared_masks \
  --output_confidence_dir /path/to/prepared_confidence \
  --input_format mask2former \
  --overwrite
```

최근 smoke 검증에서는 flat external mask 2장을 normalize한 뒤,
그 결과를 `auto_assign_object_ids.py`에 넣어 grouped model 생성까지 확인했습니다.
추가로 adapter smoke로는 `generic`, `sam2` stacked-mask, `mask2former` logits tensor, `deva` track-preserving stacked-mask 경로를 각각 확인했습니다.

## 한계와 주의점

- geometry supervision은 현재 SAM2 feature, optional depth, optional object mask까지 연결된 1차 구현입니다.
- SAM2 feature는 더 이상 3채널 고정은 아니지만, 아직 object consistency 자체를 직접 강하게 밀어주는 단계는 아닙니다.
- 현재 object-aware 분리는 `foreground/background-aware separation`에 더 가깝고, multi-instance separation은 아직 약합니다.
- 따라서 현재 object-aware 품질은 `mask prior` 품질에 크게 영향을 받습니다.
- automatic grouping은 2D mask 품질에 크게 의존합니다.
- 현재 physics export는 intermediate representation이며, differentiable rigid-body solver와 직접 결합되지는 않습니다.
- object-only rendering, foreground score render, object mask prior render, overlay debug render는 구현되어 있습니다.
- halo를 줄이기 위한 `--foreground_threshold`는 들어갔지만, 학습 내부 pruning/regularization은 더 보강할 여지가 있습니다.

## 논문 정렬을 위한 다음 구현

ContactGaussian-WM의 scene initialization 다음 단계까지 맞추려면, 현재 코드에서 가장 중요한 후속 구현은 아래 순서입니다.

1. foreground halo 억제와 pruning 강화
- 낮은 foreground score Gaussian opacity 억제
- foreground/background-aware pruning
- boundary sharpening regularization

2. object-aware initialization 강화
- foreground/background seed 차별화
- mask/visual hull prior를 초기 Gaussian 속성에 더 직접 반영
- SAM2 feature와 object mask를 결합한 더 강한 separation loss

3. multi-object separation과 physics 연결
- post-hoc grouping 의존 축소
- multi-instance aware representation 검토
- collision geometry export와 이후 rigid-body stage 연결

4. scene initialization과 dynamics refinement의 2-stage 학습 분리
- 현재 README의 initialization stage 이후에 physics-aware refinement stage 추가
- geometry와 physical parameters를 joint refinement 하는 학습 루프 구성

즉 현재 구현은 논문 파이프라인 중 "stage 1: unified spherical Gaussian scene initialization"에 가깝고,
논문과 같은 효과를 내려면 다음부터는 rigid object abstraction과 differentiable physics stage가 핵심입니다.

## GitHub 업로드 시 권장

올리는 것:

- 소스 코드
- README
- TODO
- 필요 시 environment 파일
- `.gitignore`

보통 올리지 않는 것:

- `__pycache__/`
- `*.pyc`
- `.codex/`
- `.cuda-11.8/`
- `output/`
- `sam_features_sam2/`
- 대용량 checkpoint
- CUDA extension build 산출물 (`*.so`, `build/`)
- `densification_stats.jsonl`가 들어 있는 실험 output 폴더 전체

즉 GitHub에는 "코드와 실행 방법"을 올리고, 학습 결과물과 feature cache는 제외하는 구성이 적절합니다.

현재 저장소에서는 실험 결과를 로컬에서 보존할 수 있도록 `output/`은 Git 추적 대상에서 제외하는 것을 권장합니다.

업로드 전 작업 트리 정리 예시:

```bash
find gaussian_initiailization -type d -name __pycache__ -prune -exec rm -rf {} +
find gaussian_initiailization -type d -path '*/build/*' -prune -exec rm -rf {} +
find gaussian_initiailization -type f \( -name '*.pyc' -o -name '*.so' \) -delete
git status --short
```

이미 한 번 Git이 추적한 캐시 / 빌드 파일이 있다면 index에서도 빼는 것이 좋습니다.

```bash
git rm -r --cached gaussian_initiailization/**/__pycache__
git rm -r --cached gaussian_initiailization/submodules/simple-knn/build
git rm -r --cached gaussian_initiailization/submodules/diff-gaussian-rasterization/build
git rm -r --cached gaussian_initiailization/submodules/fused-ssim/build
```
