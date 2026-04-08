# Gaussian Initialization in CG-WM

이 디렉터리는 CG-WM의 scene initialization 실험 코드를 담고 있습니다.
기반은 Inria의 3D Gaussian Splatting이며, 현재 코드는 ContactGaussian-WM 방향의 초기화 실험을 위해 아래 기능들이 추가된 상태입니다.

- isotropic spherical Gaussian 제약
- geometry / appearance decoupled optimization
- geometry / appearance loss 분리
- image-wise exposure optimization
- SAM2 feature supervision 경로

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
  - geometry: reconstruction + regularization + optional depth + optional SAM2 feature loss
- exposure optimizer 분리 및 checkpoint 복구
- 구 decoupled checkpoint 호환 resume
- SAM2 feature map `.npy` 로딩 및 geometry step supervision
- `--disable_viewer` 사용 시 viewer import 실패 fallback
- Gaussian별 `object_id` 저장 / 복구 / PLY round-trip
- 수동 / 자동 object grouping 스크립트
- physics stage 연결용 intermediate export
- 외부 instance segmentation 결과 normalize helper

현재 저장소 기준으로는 이 디렉터리가 "scene initialization + object grouping/export 입구"까지 담당합니다.
즉 rigid-body dynamics 자체는 아직 없지만, 그 직전 단계까지는 코드와 입출력 경로가 연결된 상태입니다.

## 코드 구조

- `train.py`
  - 학습 진입점
  - decoupled optimization, geometry/apppearance loss 계산, densification/pruning 수행
- `render.py`
  - 저장된 Gaussian으로 train/test 뷰 렌더링
- `metrics.py`
  - PSNR / SSIM / LPIPS 평가
- `extract_sam2_features.py`
  - SAM2 image encoder feature를 `.npy`로 저장하는 전처리 스크립트
- `assign_object_ids.py`
  - 외부 object id 배열을 Gaussian에 붙여 grouped model로 저장
- `auto_assign_object_ids.py`
  - 2D instance mask를 multi-view voting으로 Gaussian `object_id`에 자동 할당
- `prepare_instance_masks.py`
  - 외부 segmentation 결과를 `auto_assign_object_ids.py` 입력 형식으로 normalize
- `export_physics_scene.py`
  - grouped Gaussian scene을 physics stage용 JSON / NPZ로 export
- `arguments/`
  - CLI 인자 정의
- `scene/`
  - dataset loading, camera 구성, GaussianModel 정의
- `gaussian_renderer/`
  - CUDA rasterizer 호출

## 핵심 구현 포인트

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
  - scale / opacity regularization
  - optional SAM2 feature loss

로그에는 아래 항목이 분리되어 기록됩니다.

- `geometry_loss`
- `appearance_loss`
- `geometry_feature_loss`

### 4. SAM2 feature supervision

SAM2 feature supervision은 현재 구현되어 있습니다.

- dataset 아래 `sam_features/.../*.npy`를 읽습니다.
- Blender synthetic 데이터셋에서는 split-aware 경로를 지원합니다.
  - `sam_features/train/*.npy`
  - `sam_features/test/*.npy`
  - `sam_features/val/*.npy`
- geometry step에서 Gaussian의 geometry feature를 렌더한 뒤 target feature와 L1로 비교합니다.
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

### 1. SAM2 feature 추출

먼저 SAM2 feature map을 만듭니다.

```bash
conda run -n sam2cpu python gaussian_initiailization/extract_sam2_features.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --output_dir sam_features_sam2 \
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

## 최근 검증 결과

실제로 아래가 확인되었습니다.

- `lego` 데이터셋에서 10,000 iteration 학습 성공
- SAM2 feature supervision을 켠 2-iteration smoke test 성공
- `baseline`, `joint`, `alternating` 짧은 CUDA smoke test 성공
- `lego_sam2_10k` 모델 render 성공
- test view 200장 저장 확인
- 현재 코드가 직접 만든 decoupled checkpoint self-resume 성공
- 예전 decoupled checkpoint에서 빠져 있던 `f_geo` optimizer group을 이름 기반 fallback으로 복구 가능
- `--disable_viewer`를 줬을 때 socket permission 문제 없이 학습 시작 가능

예시 결과:

- 모델 출력: `gaussian_initiailization/output/lego_sam2_10k`
- render 출력: `gaussian_initiailization/output/lego_sam2_10k/test/ours_10000/renders`

## 현재 코드에 있는 주요 옵션

### 데이터 / 입출력

- `--source_path`
- `--model_path`
- `--images`
- `--depths`
- `--sam_features`
- `--sam_feature_normalization`
- `--resolution`
- `--white_background`
- `--eval`
- `--disable_viewer`

### decoupled optimization

- `--alternating_optimization`
- `--joint_optimization`
- `--geometry_iters`
- `--appearance_iters`

### geometry supervision

- `--sam_feature_weight`
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

- Gaussian별 `object_id`가 checkpoint와 PLY에 함께 저장됩니다.
- `assign_object_ids.py`로 외부에서 만든 object id 배열을 붙일 수 있습니다.
- `auto_assign_object_ids.py`로 2D instance mask에서 자동 grouping할 수 있습니다.
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

array bundle에도 object별 `source_track_ids`가 함께 저장됩니다.

즉 지금은 initialization 결과를 바로 rigid grouping 실험과 physics stage 입구까지 넘길 수 있는 상태입니다.

## 외부 Instance Mask 연동

외부 segmentation 모델의 출력 형식이 바로 맞지 않아도, 먼저 `prepare_instance_masks.py`로 normalize한 뒤
`auto_assign_object_ids.py`에 연결하면 됩니다.

권장 순서는 아래와 같습니다.

1. 외부 segmentation 결과 normalize

```bash
conda run -n gaussian_splatting python gaussian_initiailization/prepare_instance_masks.py \
  --source_path /path/to/scene \
  --input_masks_dir /path/to/external_masks \
  --output_masks_dir /path/to/prepared_masks \
  --input_confidence_dir /path/to/external_confidence \
  --output_confidence_dir /path/to/prepared_confidence \
  --overwrite
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
  --output_dir gaussian_initiailization/output/your_model_grouped/physics_export
```

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

- geometry supervision은 현재 SAM2 feature와 optional depth까지 연결된 1차 구현입니다.
- SAM2 feature는 현재 3채널 proxy supervision에 가깝습니다.
- automatic grouping은 2D mask 품질에 크게 의존합니다.
- 현재 physics export는 intermediate representation이며, differentiable rigid-body solver와 직접 결합되지는 않습니다.
- object-only rendering이나 instance-level visualization은 아직 구현되지 않았습니다.

## 논문 정렬을 위한 다음 구현

ContactGaussian-WM의 scene initialization 다음 단계까지 맞추려면, 현재 코드에서 가장 중요한 후속 구현은 아래 순서입니다.

1. object-level rigid grouping
- Gaussian별 object id 저장
- foreground / background 분리
- instance mask 기반 grouping

2. collision geometry export
- object별 center / radius / spherical Gaussian 집합 export
- physics stage가 바로 읽을 수 있는 intermediate format 정의

3. differentiable collision and dynamics 연결
- Gaussian geometry에서 contact point를 계산하는 경로
- rigid-body state update와 rendering loss를 end-to-end로 연결

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
