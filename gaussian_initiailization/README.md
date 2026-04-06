# Scene Initialization in CG-WM

이 디렉터리는 CG-WM 프로젝트의 scene initialization 실험 코드를 담고 있습니다.
기반 코드는 Inria의 3D Gaussian Splatting 구현이며, 현재 저장소에서는 원본 구조를 유지하면서 일부 동작을 실험 목적에 맞게 수정한 상태입니다.

현재 문서는 "지금 이 코드가 실제로 무엇을 하는지"를 기준으로 작성되어 있습니다.

## 한눈에 보기

- 입력 데이터: COLMAP 장면 또는 Blender/NeRF synthetic 형식
- 핵심 학습 대상: 3D Gaussian의 위치, 색상 SH 계수, opacity, scale, exposure
- 핵심 변경점:
  - Gaussian rotation을 사실상 고정
  - scale을 등방성(isotropic)으로 강제
  - exposure optimizer 사용
  - sparse Adam 사용 가능
- 현재 상태:
  - 기본 학습/저장/렌더/평가 파이프라인은 동작
  - README에 예전에 적혀 있던 `alternating_optimization`, `joint_optimization` 관련 옵션은 현재 코드에 구현되어 있지 않음

## 디렉터리 구조

- `train.py`
  - 학습 루프 진입점
  - 랜덤 카메라 샘플링, 렌더링, loss 계산, densification/pruning, 저장 수행
- `render.py`
  - 저장된 모델을 불러 train/test view 렌더링
- `convert.py`
  - COLMAP 기반 데이터 전처리 스크립트
- `metrics.py`
  - 렌더 결과에 대해 SSIM / PSNR / LPIPS 계산
- `full_eval.py`
  - 여러 데이터셋에 대해 학습/렌더/평가를 한 번에 돌리는 스크립트
- `scene/`
  - 데이터 로딩, 카메라 구성, GaussianModel 정의
- `gaussian_renderer/`
  - CUDA rasterizer를 감싸는 렌더 함수
- `arguments/`
  - CLI 인자 정의
- `utils/`
  - 카메라, 이미지, SH, loss, 일반 유틸리티
- `submodules/`
  - `diff-gaussian-rasterization`
  - `simple-knn`
  - `fused-ssim`

## 실행 흐름

### 1. 학습 시작

`train.py`는 다음 순서로 동작합니다.

1. CLI 인자를 파싱합니다.
2. `GaussianModel`과 `Scene`를 생성합니다.
3. `Scene`가 입력 데이터를 읽고 초기 point cloud와 카메라를 준비합니다.
4. 매 iteration마다 train camera 하나를 랜덤 선택합니다.
5. 현재 Gaussian들을 렌더링합니다.
6. L1 + SSIM 기반 loss를 계산합니다.
7. 필요 시 depth regularization을 추가합니다.
8. backward 후 optimizer step을 수행합니다.
9. 주기적으로 densification / pruning을 수행합니다.
10. 지정 iteration에서 point cloud와 checkpoint를 저장합니다.

### 2. 장면 로딩

`scene/__init__.py`의 `Scene` 클래스가 데이터를 읽습니다.

- `source_path/sparse`가 있으면 COLMAP 장면으로 처리합니다.
- `transforms_train.json`이 있으면 Blender 장면으로 처리합니다.
- 최초 학습 시:
  - 입력 PLY를 `model_path/input.ply`로 복사
  - 카메라 정보를 `cameras.json`으로 저장
  - sparse point cloud에서 Gaussian 초기화
- 렌더링 또는 재로딩 시:
  - `point_cloud/iteration_x/point_cloud.ply`를 읽어 모델 복원

### 3. 렌더링

`gaussian_renderer.render()`는 다음 정보를 rasterizer에 넘깁니다.

- Gaussian 중심 좌표
- opacity
- scale / rotation 또는 precomputed covariance
- SH feature 또는 Python에서 미리 계산한 RGB

렌더 후에는 필요 시 exposure를 적용하고 결과를 `[0, 1]` 범위로 clamp합니다.

## 현재 코드의 핵심 구현 포인트

### 1. Isotropic spherical Gaussian 제약

현재 저장소에서 가장 큰 커스텀 변경은 `scene/gaussian_model.py`에 있습니다.

- `get_scaling`
  - 학습된 3축 scale을 그대로 쓰지 않고 평균값으로 묶어 `(s, s, s)` 형태로 사용합니다.
- `get_rotation`
  - 학습된 rotation 파라미터 대신 identity quaternion을 반환합니다.
- `get_covariance`
  - isotropic scaling + identity rotation 기준으로 covariance를 계산합니다.
- `densify_and_split`
  - 새로 생성되는 Gaussian도 같은 제약을 유지합니다.

즉, 현재 코드는 원본 3DGS의 anisotropic Gaussian을 그대로 쓰지 않고, 회전이 고정된 isotropic Gaussian 형태로 장면을 학습합니다.

### 2. Exposure 학습

각 이미지별 exposure 파라미터를 따로 들고 있습니다.

- `create_from_pcd()`에서 이미지 수만큼 `3x4` exposure 행렬을 초기화합니다.
- 학습 중에는 메인 optimizer와 별개로 `exposure_optimizer`가 step을 수행합니다.
- 저장 시 `exposure.json`으로 내보냅니다.
- 렌더 시 `use_trained_exp`가 켜져 있으면 결과 이미지에 exposure를 적용합니다.

### 3. Densification / pruning

원본 3DGS 흐름을 유지합니다.

- 화면 공간 gradient를 누적
- 작은 Gaussian은 clone
- 큰 Gaussian은 split
- opacity가 너무 작거나 화면/월드 공간에서 너무 큰 Gaussian은 prune

다만 split 이후에도 isotropic 제약이 유지되도록 수정되어 있습니다.

### 4. SH 기반 색 표현

색은 spherical harmonics 계수로 관리합니다.

- 초기 point color는 `RGB2SH`로 DC 성분에 들어갑니다.
- 렌더 시:
  - Python에서 SH -> RGB 변환을 하거나
  - rasterizer 내부에서 처리합니다.

## 현재 사용 가능한 주요 옵션

실제로 코드에 정의되어 있는 옵션만 정리합니다.

### 공통/입력 관련

- `--source_path`, `-s`
- `--model_path`, `-m`
- `--images`, `-i`
- `--depths`, `-d`
- `--resolution`, `-r`
- `--white_background`
- `--eval`
- `--train_test_exp`
- `--data_device`

### 학습 관련

- `--iterations`
- `--position_lr_init`
- `--position_lr_final`
- `--feature_lr`
- `--opacity_lr`
- `--scaling_lr`
- `--rotation_lr`
- `--exposure_lr_init`
- `--exposure_lr_final`
- `--densification_interval`
- `--densify_from_iter`
- `--densify_until_iter`
- `--densify_grad_threshold`
- `--depth_l1_weight_init`
- `--depth_l1_weight_final`
- `--random_background`
- `--optimizer_type`
- `--checkpoint_iterations`
- `--start_checkpoint`
- `--disable_viewer`

### 렌더 파이프라인 관련

- `--convert_SHs_python`
- `--compute_cov3D_python`
- `--antialiasing`
- `--debug`

## 현재 코드에 없는 기능

예전 문서에 있었더라도, 현재 코드 기준으로는 아래 기능이 구현되어 있지 않습니다.

- `--alternating_optimization`
- `--joint_optimization`
- `--geometry_iters`
- `--appearance_iters`
- geometry optimizer / appearance optimizer 분리 학습 로직

즉, 지금 학습 코드는 하나의 메인 Gaussian optimizer와 하나의 exposure optimizer를 사용하는 구조입니다.

## 체크포인트와 저장 형식

### 저장되는 것

- `point_cloud/iteration_x/point_cloud.ply`
- `exposure.json`
- `chkpnt<iter>.pth`
- `cfg_args`
- TensorBoard 로그 가능 시 이벤트 파일

### 체크포인트에 포함되는 것

현재 `capture()` 기준으로 다음이 저장됩니다.

- SH degree
- xyz
- feature DC / rest
- scaling
- rotation 파라미터
- opacity
- densification 관련 텐서
- 메인 optimizer state
- spatial lr scale

### 주의할 점

현재 checkpoint는 메인 optimizer state는 복원하지만, exposure tensor 자체와 exposure optimizer state를 함께 저장/복원하는 구조는 아닙니다.

따라서 "중간 checkpoint에서 완전히 같은 학습 상태로 exposure까지 이어서 재개"를 보장하는 구현은 아직 아닙니다.

## 기본 사용 예시

### 학습

```bash
python train.py \
  --source_path /path/to/dataset \
  --model_path output/example \
  --iterations 30000 \
  --eval \
  --disable_viewer \
  --quiet
```

### 렌더링

```bash
python render.py \
  --source_path /path/to/dataset \
  --model_path output/example \
  --iteration 30000 \
  --skip_train \
  --resolution 4 \
  --quiet
```

### 체크포인트 저장

```bash
python train.py \
  --source_path /path/to/dataset \
  --model_path output/example_ckpt \
  --iterations 30000 \
  --checkpoint_iterations 5000 10000 20000 \
  --eval \
  --disable_viewer \
  --quiet
```

### 체크포인트 재개

```bash
python train.py \
  --source_path /path/to/dataset \
  --model_path output/example_ckpt \
  --iterations 30000 \
  --start_checkpoint output/example_ckpt/chkpnt10000.pth \
  --eval \
  --disable_viewer \
  --quiet
```

## 데이터 형식

### COLMAP

아래 구조를 기대합니다.

```text
<scene>/
  images/
  sparse/0/
    cameras.bin or cameras.txt
    images.bin or images.txt
    points3D.bin or points3D.txt
```

필요하면 `convert.py`로 COLMAP 파이프라인을 돌려 변환할 수 있습니다.

### Blender / NeRF synthetic

아래 파일이 있으면 Blender 데이터셋으로 인식합니다.

```text
transforms_train.json
```

## 평가

렌더 결과가 저장된 뒤 `metrics.py`로 평가할 수 있습니다.

```bash
python metrics.py -m output/example
```

출력:

- `results.json`
- `per_view.json`

## 알려진 상태와 해석

- 이 코드는 원본 3DGS의 완전한 대체 구현이 아니라, scene initialization 실험용 변형 버전입니다.
- 문서보다 코드가 더 신뢰할 만한 기준이며, 현재는 isotropic spherical Gaussian 제약이 가장 명확한 변경점입니다.
- 품질 튜닝이나 ContactGaussian-WM 논문 수준의 decoupled optimization 재현은 아직 이 디렉터리에서 완성된 상태로 보이지 않습니다.

## 추천 확인 포인트

코드를 더 볼 때는 아래 순서가 가장 이해하기 쉽습니다.

1. `train.py`
2. `scene/__init__.py`
3. `scene/gaussian_model.py`
4. `gaussian_renderer/__init__.py`
5. `arguments/__init__.py`

이 순서대로 보면 입력에서 학습, 저장, 렌더까지의 전체 흐름이 거의 다 연결됩니다.
