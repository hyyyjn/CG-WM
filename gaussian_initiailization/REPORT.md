# Gaussian Initialization Progress Report

## Goal

ContactGaussian-WM에서 `scene initialization` 단계를 구현하는 것이 현재 목표입니다.

현재 구현은 ContactGaussian-WM 전체 파이프라인이 아니라,
그중 `stage 1: unified spherical Gaussian scene initialization`에 가까운 범위를 대상으로 합니다.

## 개요

ContactGaussian-WM의 scene initialization에서 중요한 부분은 다음과 같습니다.

- spherical Gaussian 제약
- decoupled optimization

이 scene initialization 부분은 기존 3DGS 논문의 파이프라인과 구조적으로 유사하기 때문에,
기존 Inria 3D Gaussian Splatting 코드를 기반으로 수정하는 방식으로 구현을 시작했습니다.

그 위에 geometry 중심 supervision, checkpoint 확장, object grouping 연결, physics export 경로를 추가해
scene initialization 결과를 후속 stage로 넘길 수 있도록 확장했습니다.

## 현재까지 구현한 것

현재까지 구현된 핵심 기능은 다음과 같습니다.

- isotropic spherical Gaussian 제약
- geometry / appearance decoupled optimization
- geometry / appearance loss 분리
- image-wise exposure optimization
- depth supervision 경로
- SAM2 feature supervision 경로
- richer SAM feature supervision
- geometry feature 차원 확장
- checkpoint / resume 확장
- viewer 비활성화 fallback
- Gaussian별 `object_id` 저장 / 복구
- object-only render
- 수동 / 자동 object grouping
- foreground object mask 자동 추출
- 외부 instance segmentation 결과 normalize helper
- physics stage 연결용 intermediate export
- densification statistics logging

즉 현재 코드는 단순 3DGS reconstruction 코드가 아니라,

- scene initialization 학습
- 결과 render 확인
- Gaussian object grouping
- physics stage 직전 intermediate export

까지 이어지는 파이프라인을 갖추고 있습니다.

## 구현 상세

### 2.1 Spherical Gaussian 제약

기존 3DGS의 anisotropic Gaussian을 그대로 쓰지 않고,
scene initialization 단계에서는 spherical Gaussian 표현을 사용하도록 바꾸었습니다.

구현 방식:

- scaling은 내부적으로 3축 평균값을 사용해 isotropic하게 읽음
- rotation은 identity quaternion으로 고정
- densification 이후 새 Gaussian도 같은 제약 유지

효과:

- initialization 단계의 Gaussian 표현을 unified spherical Gaussian 방향으로 맞춤

관련 파일:

- `scene/gaussian_model.py`

### 2.2 Geometry / Appearance Decoupled Optimization

scene initialization에서 geometry와 appearance의 역할을 분리하기 위해
optimizer 수준에서 branch를 나누었습니다.

구현 방식:

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
- 아무 옵션도 주지 않으면 baseline 방식으로 동작

효과:

- geometry 학습 책임과 appearance 학습 책임을 분리해 initialization 실험을 더 명확하게 함

관련 파일:

- `train.py`
- `scene/gaussian_model.py`
- `arguments/__init__.py`

정리하면, decoupled optimization을 지원하기 위해

- geometry Gaussian과 visual Gaussian에 해당하는 parameter를 구조적으로 분리했고
- optimization 시 branch별로 update가 따로 일어나도록 수정했으며
- 각 branch에서 사용하는 loss를 분리해 구성했습니다.

### 2.3 Loss 분리

geometry와 appearance를 구조적으로만 분리한 것이 아니라,
loss 계산도 branch별로 나누어 구성했습니다.

appearance step:

- RGB reconstruction
- L1
- DSSIM
- opacity regularization

geometry step:

- reconstruction term
- scale regularization
- optional depth loss
- optional SAM feature loss

효과:

- initialization에서 geometry branch가 appearance reconstruction에만 끌려가지 않도록 분리

관련 파일:

- `train.py`

### 2.4 Depth Supervision

geometry initialization 품질을 높이기 위해 optional monocular inverse-depth supervision을 연결했습니다.

구현 방식:

- camera가 invdepth map과 reliable mask를 가질 수 있도록 확장
- geometry step에서 inverse-depth 차이에 대한 L1 loss 추가

효과:

- RGB reconstruction만 쓰는 경우보다 geometry prior를 더 줄 수 있음

관련 파일:

- `scene/cameras.py`
- `utils/camera_utils.py`
- `train.py`

### 2.5 SAM2 Feature Supervision

scene initialization에서 더 풍부한 geometry signal을 쓰기 위해
SAM2 feature map supervision 경로를 추가했습니다.

구현 방식:

- `extract_sam2_features.py`로 이미지별 `.npy` feature map 저장
- dataset 로더에서 `sam_features/.../*.npy`를 읽음
- geometry step에서 Gaussian geometry feature를 render한 뒤 target feature와 L1 비교
- `--geometry_feature_dim`을 통해 geometry feature 차원 조절 가능
- 3채널 초과 feature는 multi-chunk 방식으로 supervision
- geometry render와 feature render가 같은 screen-space gradient를 공유하도록 연결
- `--sam_feature_normalization`으로 feature preprocessing 제어 가능

효과:

- RGB만으로는 약한 geometry supervision을 richer feature로 보강
- densification에도 간접적으로 feature supervision 영향 반영

관련 파일:

- `extract_sam2_features.py`
- `scene/cameras.py`
- `utils/camera_utils.py`
- `train.py`

### 2.6 Checkpoint / Resume 확장

decoupled optimization과 geometry feature를 쓰면서,
기존 checkpoint 구조만으로는 상태 복원이 충분하지 않아 저장 포맷을 확장했습니다.

구현 방식:

- Gaussian parameter tensor 저장
- geometry optimizer state 저장
- appearance optimizer state 저장
- exposure optimizer state 저장
- geometry feature tensor 저장
- `object_id` 저장
- densification 관련 상태 저장
- 예전 tuple 형식 checkpoint에 대한 fallback 유지
- 예전 decoupled checkpoint에서 optimizer group 누락 시 이름 기반 fallback restore 지원

효과:

- scene initialization 실험을 중단했다가 이어서 재개 가능
- 구조가 바뀐 이전 checkpoint와도 어느 정도 호환

관련 파일:

- `scene/gaussian_model.py`
- `train.py`

### 2.7 Object Grouping 연결

scene initialization 결과를 이후 object-level stage로 넘기기 위해
Gaussian별 object id와 grouping 경로를 추가했습니다.

구현 방식:

- Gaussian별 `object_id`를 PLY / checkpoint에 저장
- 외부 object id 배열을 수동으로 붙일 수 있는 `assign_object_ids.py` 추가
- 2D instance mask로부터 multi-view voting 기반 자동 grouping을 수행하는 `auto_assign_object_ids.py` 추가
- occlusion-aware frontmost filtering 지원
- confidence-weighted voting 지원
- boundary downweighting 지원
- multi-view consistency refinement 지원

효과:

- initialization 결과를 object abstraction과 physics stage 입구로 넘길 수 있음

관련 파일:

- `assign_object_ids.py`
- `auto_assign_object_ids.py`
- `scene/gaussian_model.py`
- `render.py`

### 2.8 Object Mask 준비 및 Physics Export

외부 segmentation 결과가 바로 맞지 않는 경우를 대비해,
mask normalization과 자동 mask 추출 경로도 추가했습니다.

구현 방식:

- `extract_object_masks.py`
  - alpha 기반 foreground mask
  - background subtraction 기반 foreground mask
- `prepare_instance_masks.py`
  - generic
  - sam2
  - mask2former
  - deva
  형식 지원
- `export_physics_scene.py`
  - grouped Gaussian scene을 JSON / NPZ intermediate format으로 export
  - body frame, sphere + AABB collision proxy, 기본 mass metadata 포함

효과:

- scene initialization 결과를 바로 object grouping 실험과 physics stage 직전까지 넘길 수 있음

관련 파일:

- `extract_object_masks.py`
- `prepare_instance_masks.py`
- `export_physics_scene.py`

## 3. 기존 대비 추가 구현한 것

기존 Inria 3D Gaussian Splatting 대비 추가되었거나 확장된 기능은 다음과 같습니다.

### 3.1 핵심 학습 구조 확장

- isotropic spherical Gaussian 제약
- geometry / appearance decoupled optimization
- geometry / appearance loss 분리
- depth supervision
- SAM2 feature supervision
- richer feature supervision
- geometry feature dimension 확장

### 3.2 실험 편의성 확장

- viewer import 실패 시 no-op fallback
- `--disable_viewer` 옵션
- decoupled checkpoint 저장 / 복구
- old checkpoint compatibility fallback
- densification statistics logging
- variant 비교용 스크립트

### 3.3 후속 stage 연결 확장

- Gaussian `object_id` 저장 / 복구
- object-only render
- manual / automatic object grouping
- automatic foreground mask extraction
- external mask normalization helper
- physics-friendly export

즉, 기존 3DGS가 주로 `scene reconstruction` 자체에 집중했다면,
현재 구현은 `scene initialization -> object grouping -> physics stage 입구`까지 실험 가능하도록 확장된 상태입니다.

## 4. 현재 상태 평가

scene initialization 관점에서 보면 현재 구현은 다음처럼 정리할 수 있습니다.

- 구현이 잘 된 부분
  - spherical Gaussian 표현
  - decoupled optimization 구조
  - geometry supervision 경로
  - checkpoint / render / object grouping 연결

- 아직 부족한 부분
  - object-aware initial Gaussian seeding
  - silhouette / normal / boundary consistency 같은 stronger geometry supervision
  - scene initialization 품질의 정량 지표 체계
  - differentiable contact / rigid-body refinement
  - 2-stage pipeline의 dynamics refinement stage

즉 현재 단계는
`scene initialization의 핵심 구조는 구현되었고 실험 가능한 상태`
라고 평가할 수 있습니다.

### 4.1 ContactGaussian-WM 기준 아직 빠진 모듈 체크리스트

ContactGaussian-WM 본문에서 설명하는 전체 파이프라인과 비교하면,
현재 구현은 scene initialization 학습기 쪽은 상당 부분 반영되어 있지만
데이터 생성 / dense initialization / 후속 refinement 쪽은 아직 비어 있는 부분이 있습니다.

아직 빠진 모듈은 다음과 같습니다.

- Blender multi-view data generation
  - synthetic object와 LEAP Hand component용 OBJ를 입력으로 받아
    72개 균일 구면 시점에서 render
  - image / mask / camera parameter 자동 생성
- mesh-based dense point cloud initialization
  - Blender mesh에서 dense surface point cloud를 직접 샘플링해 초기화
  - 현재 synthetic loader는 해당 기능이 없으면 random point cloud로 시작
- Visual Hull 기반 dense initialization
  - multi-view mask로 visual hull을 구성하고
    그 결과를 dense point cloud로 변환해 초기 Gaussian seeding에 사용
  - 현재 real-world 쪽은 COLMAP sparse point cloud를 그대로 읽는 구조
- mask-driven initialization pipeline
  - mask를 object grouping 보조 정보가 아니라
    초기 Gaussian 생성 자체에 직접 반영하는 단계
- LEAP Hand component preprocessing
  - hand component별 OBJ 정리
  - component identity를 유지한 dataset 생성
  - object / hand part 단위의 unified preprocessing
- unified representation builder
  - 논문 Fig.7 수준으로 object와 hand component를
    같은 representation으로 묶는 입력 전처리
- object-aware initial Gaussian seeding
  - object / component 경계를 반영해 초기 Gaussian을 배치하는 로직
- silhouette-based geometry prior
  - mask reprojection
  - silhouette consistency
  - occupancy 류 supervision
- normal / boundary consistency supervision
  - stronger geometry prior에 해당하는 추가 supervision
- contact / dynamics refinement stage
  - scene initialization 이후의 rigid-body / contact / dynamics refinement
- initialization quality evaluation protocol
  - scene initialization 품질을 정량적으로 비교할 metric 체계
- end-to-end pipeline orchestration
  - OBJ 또는 real capture 입력부터
    preprocessing -> dense initialization -> SG-GS training -> export까지
    이어지는 전체 실행 스크립트

정리하면 현재 구현은
`ContactGaussian-WM 전체 파이프라인 중 SG-GS 기반 scene initialization 엔진에 더 가깝고,`
`논문에서 강조하는 데이터 생성 / dense initialization / 후속 dynamics refinement는 아직 남아 있다`
고 볼 수 있습니다.

## 5. 테스트 가능한 방법

현재 구현은 아래 순서로 직접 테스트할 수 있습니다.

### 5.1 환경 준비

프로젝트 루트:

```bash
cd /home/cgr-ugrad-2026/work/CG-WM
```

학습 / render 환경:

```bash
conda env create -f gaussian_initiailization/environment.yml
```

필요 시 extension 재설치:

```bash
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/diff-gaussian-rasterization
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/simple-knn
conda run -n gaussian_splatting pip install -e gaussian_initiailization/submodules/fused-ssim
```

SAM2용 CPU 환경:

```bash
conda create -n sam2cpu python=3.10 -y
conda run -n sam2cpu pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
```

### 5.2 SAM2 Feature 추출

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

### 5.3 Scene Initialization Smoke Test

가장 짧게 구현이 동작하는지 확인하는 테스트입니다.

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/test_smoke \
  --iterations 20 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --alternating_optimization \
  --geometry_iters 1 \
  --appearance_iters 1
```

확인 포인트:

- 학습이 에러 없이 시작 / 종료되는지
- `point_cloud/iteration_20/point_cloud.ply`가 생기는지
- `training_args.json`이 저장되는지

### 5.4 장기 학습 예시

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

### 5.5 Render 확인

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/test_smoke \
  --iteration 20 \
  --skip_train \
  --resolution 4 \
  --eval \
  --quiet
```

확인 포인트:

- render PNG가 정상 저장되는지
- test view가 복원되는지

예시 출력:

- `gaussian_initiailization/output/test_smoke/test/ours_20/renders`

### 5.6 Baseline / Joint / Alternating 비교 테스트

baseline:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/test_baseline \
  --iterations 20 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet
```

joint:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/test_joint \
  --iterations 20 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --joint_optimization
```

alternating:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/test_alternating \
  --iterations 20 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --alternating_optimization \
  --geometry_iters 1 \
  --appearance_iters 1
```

### 5.7 Geometry Feature 차원 비교 테스트

3채널:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 3 \
  --model_path gaussian_initiailization/output/test_geo3_5k \
  --iterations 5000 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --alternating_optimization \
  --geometry_iters 1 \
  --appearance_iters 1
```

9채널:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --model_path gaussian_initiailization/output/test_geo9_5k \
  --iterations 5000 \
  --resolution 8 \
  --eval \
  --disable_viewer \
  --quiet \
  --alternating_optimization \
  --geometry_iters 1 \
  --appearance_iters 1
```

### 5.8 Object Grouping 테스트

자동 foreground mask 추출:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/extract_object_masks.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --output_masks_dir /tmp/lego_masks \
  --output_confidence_dir /tmp/lego_conf \
  --method auto
```

grouping:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/auto_assign_object_ids.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/test_smoke \
  --iteration 20 \
  --masks_dir /tmp/lego_masks \
  --confidence_maps_dir /tmp/lego_conf \
  --confidence_threshold 0.5 \
  --output_model_path gaussian_initiailization/output/test_smoke_grouped
```

grouped object render:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/test_smoke_grouped \
  --iteration 20 \
  --skip_train \
  --eval \
  --object_id 1
```

### 5.9 Physics Export 테스트

```bash
conda run -n gaussian_splatting python gaussian_initiailization/export_physics_scene.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/test_smoke_grouped \
  --iteration 20 \
  --density 1000 \
  --friction 0.5 \
  --restitution 0.1 \
  --output_dir gaussian_initiailization/output/test_smoke_grouped/physics_export
```

확인 포인트:

- `physics_scene.json`
- `physics_scene_arrays.npz`

## 6. 결론

현재까지의 구현은 ContactGaussian-WM의 scene initialization 목표를 위해
기존 3DGS를 다음 방향으로 확장한 상태입니다.

- spherical Gaussian 기반 표현으로 변경
- geometry / appearance decoupled optimization 도입
- depth / SAM feature 기반 geometry supervision 추가
- checkpoint / logging / 실험 편의성 보강
- object grouping과 physics export 경로 연결

정리하면 현재 코드는
`scene initialization의 핵심 구조를 구현했고, 직접 학습 / render / grouping / export까지 테스트 가능한 상태`
입니다.

다만 아직은 initialization 품질을 더 높이기 위한

- object-aware initial seeding
- stronger geometry prior
- differentiable contact / dynamics refinement

이 남아 있으므로, 현재 단계는 `scene initialization 실험 기반이 갖춰진 중간 이상 단계`로 보는 것이 적절합니다.
