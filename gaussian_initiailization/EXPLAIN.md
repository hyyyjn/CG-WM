# Gaussian Initialization Code Explain

이 문서는 `gaussian_initiailization` 디렉터리의 핵심 코드 흐름을 빠르게 파악하기 위한 안내서입니다.
특히 지금 구현이 ContactGaussian-WM의 scene initialization 관점에서 어디까지 와 있는지,
그리고 최근 추가한 object-aware 분리 경로가 어디에 들어갔는지를 한 번에 볼 수 있도록 정리했습니다.

## 현재 코드의 큰 그림

현재 구현은 크게 두 층으로 보면 이해가 쉽습니다.

1. 논문에 더 가까운 stage 1
   - masked COLMAP
   - visual hull seed
   - SAM2 feature supervision
   - spherical Gaussian initialization
   - geometry / appearance decoupled optimization
   - object mask prior를 이용한 object-aware geometry supervision
   - Gaussian별 learned foreground score

2. repo 확장 경로
   - `assign_object_ids.py`
   - `auto_assign_object_ids.py`
   - `export_physics_scene.py`

즉 지금 핵심 학습은 점점 object-aware initialization 쪽으로 이동했고,
`grouping`은 여전히 physics export를 위한 보조 경로로 남아 있습니다.

## 전체 실행 흐름

가장 자주 쓰는 흐름은 아래 순서입니다.

1. `extract_sam2_features.py`
   - 이미지에서 SAM2 feature를 뽑아 `.npy`로 저장
2. `estimate_masked_colmap.py`
   - foreground mask를 반영해 pose를 다시 추정
3. `build_visual_hull.py`
   - multi-view mask로 visual hull seed 생성
4. `train.py`
   - scene initialization 학습
   - 최근에는 object mask prior와 foreground score 학습까지 포함
5. `render.py`
   - RGB render 저장
   - foreground score / overlay / object mask prior 디버그 이미지 저장
   - 필요하면 foreground threshold를 걸어 tighter object render 생성
6. `assign_object_ids.py` / `auto_assign_object_ids.py`
   - 필요할 때만 post-hoc object id 할당
7. `export_physics_scene.py`
   - grouped result를 physics stage용 포맷으로 export

## 핵심 데이터 객체

- `scene/cameras.py`의 `Camera`
  - 이미지, alpha, depth, SAM feature, object mask prior, camera pose를 한 묶음으로 관리
- `scene/gaussian_model.py`의 `GaussianModel`
  - Gaussian 파라미터와 optimizer, save/load, densification, foreground score를 담당

## 1. [train.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)

학습 진입점이자 현재 object-aware initialization이 실제로 구현되는 중심 파일입니다.

### 중요한 함수

- `_DisabledNetworkGUI`
  - viewer를 끄거나 import 실패했을 때 no-op 대체 객체로 사용합니다.

- `append_jsonl(path, payload)`
  - JSONL 형식 로그를 저장합니다.
  - 현재 `densification_stats.jsonl` 기록에 사용됩니다.

- `zero_active_optimizers(gaussians, use_decoupled_optimization)`
  - 현재 활성화된 optimizer gradient를 초기화합니다.

- `compute_losses(render_pkg, viewpoint_cam, opt, depth_l1_weight_value)`
  - 기본 RGB reconstruction loss와 optional depth loss를 계산합니다.

- `compute_sam_feature_loss(...)`
  - geometry feature supervision을 계산합니다.
  - `geometry_feature_dim > 3`이면 3채널 chunk로 나눠 여러 번 render 후 평균 loss를 만듭니다.

- `compute_object_mask_loss(...)`
  - 최근 추가된 object-aware supervision 핵심 함수입니다.
  - `viewpoint_cam.object_mask`를 target으로 사용합니다.
  - Gaussian의 `foreground score`를 렌더해 BCE + L1로 mask prior와 맞춥니다.
  - 지금 구현에서 논문 방향 object separation에 가장 가까운 학습 항입니다.

- `training(...)`
  - 전체 학습 루프입니다.
  - 아래를 모두 담당합니다.
  - `Scene`, `GaussianModel` 생성
  - checkpoint 복구
  - geometry / appearance step 분기
  - SAM feature loss
  - object mask loss
  - densification / pruning
  - checkpoint / evaluation 저장

- `prepare_output_and_logger(...)`
  - output 폴더와 `cfg_args`, `training_args.json` 등을 저장합니다.

- `training_report(...)`
  - 중간 evaluation과 tensorboard logging을 담당합니다.

### 이 파일에서 특히 중요한 포인트

- decoupled optimization이 실제로 분기되는 곳
- geometry / appearance loss가 분리되는 곳
- SAM feature loss가 geometry step에 연결되는 곳
- object mask prior가 geometry step에 직접 들어가는 곳
- foreground score supervision과 densification 로그가 연결되는 곳

## 2. [scene/gaussian_model.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/gaussian_model.py)

Gaussian 파라미터와 학습 상태를 관리하는 핵심 모델 파일입니다.

### 현재 모델이 들고 있는 중요한 상태

- position
- SH appearance feature
- geometry feature
- opacity
- isotropic scaling
- fixed rotation
- `object_id`
- `foreground_logit`

`foreground_logit`은 최근 추가된 per-Gaussian object-aware 상태입니다.
`sigmoid(foreground_logit)`으로 얻는 `foreground score`가 mask supervision과 thresholded render의 기반이 됩니다.

### 중요한 함수

- `create_from_pcd(...)`
  - point cloud seed에서 Gaussian들을 초기화합니다.
  - foreground logit도 같이 초기화됩니다.

- `capture()` / `restore(...)`
  - checkpoint 저장 / 복구
  - foreground logit을 포함합니다.
  - 예전 checkpoint와의 호환 분기도 들어 있습니다.

- `save_ply(...)` / `load_ply(...)`
  - Gaussian scene을 PLY로 round-trip 합니다.
  - `object_id`, `foreground_logit`, `f_geo_*` 같은 커스텀 속성도 저장합니다.

- `training_setup(...)`
  - optimizer를 구성합니다.
  - geometry optimizer에 foreground 관련 parameter group도 포함됩니다.

- `geometry_parameters()`
  - geometry branch 파라미터 목록을 반환합니다.
  - foreground logit도 이 branch에 포함됩니다.

- `get_foreground_scores`
  - `sigmoid(self._foreground_logit)` 형태의 score getter입니다.

- densification / prune 관련 함수들
  - clone / split / prune 시 foreground 상태도 같이 전파되도록 확장돼 있습니다.

### 이 파일의 의미

이제 object-aware separation은 더 이상 pure post-hoc 라벨링만이 아니라,
Gaussian representation 내부에 foreground 성향이 직접 들어간 상태입니다.

## 3. [gaussian_renderer/__init__.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/gaussian_renderer/__init__.py)

CUDA rasterizer를 호출하는 렌더링 핵심 파일입니다.

### 중요한 함수

- `render(...)`
  - Gaussian scene을 렌더합니다.

### 중요한 옵션

- `override_color`
  - geometry feature나 foreground score를 RGB처럼 직접 렌더할 때 사용합니다.
  - SAM feature supervision과 foreground debug render 모두 여기 의존합니다.

- `screenspace_points`
  - 여러 render pass가 같은 projection gradient를 공유할 수 있게 합니다.

- `gaussian_mask`
  - 특정 Gaussian subset만 렌더합니다.
  - object-only render와 foreground-threshold render에 사용됩니다.

## 4. [scene/cameras.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/cameras.py)

카메라와 per-view 입력 데이터를 관리합니다.

### `Camera`가 현재 들고 있는 입력

- RGB image
- alpha mask
- depth map
- SAM feature map
- object mask prior
- pose / projection matrix

### 최근 변경에서 중요한 부분

- `object_mask`
  - 외부 `--masks_dir`가 있으면 그 mask를 사용합니다.
  - 없고 RGBA 입력이면 alpha를 object mask prior로 fallback 사용합니다.

- `has_object_mask_prior`
  - 현재 뷰에 object-aware supervision이 가능한지 표시합니다.

즉 object mask supervision은 별도 후처리가 아니라 camera 입력으로 학습 루프까지 직접 전달됩니다.

## 5. [utils/camera_utils.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/utils/camera_utils.py)

dataset reader와 `Camera` 사이를 잇는 helper입니다.

### 중요한 함수

- `load_object_mask_prior(masks_dir, cam_info)`
  - split-aware 또는 flat 구조에서 object mask를 찾습니다.
  - `.npy`, `.png`를 모두 지원합니다.

- `loadCam(...)`
  - camera를 만들 때 SAM feature, depth와 함께 object mask prior도 같이 실어 보냅니다.

### 의미

이 파일 덕분에 `--masks_dir`만 주면 object mask prior가 자동으로 학습 입력으로 들어갑니다.

## 6. [render.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/render.py)

학습된 Gaussian을 이미지로 저장하는 스크립트입니다.

### 중요한 함수

- `build_gaussian_mask(...)`
  - `--object_id`, `--foreground_threshold`를 이용해 렌더에 사용할 Gaussian subset을 만듭니다.

- `build_foreground_debug_render(...)`
  - Gaussian foreground score를 grayscale render로 만듭니다.

- `render_set(...)`
  - 특정 split의 여러 뷰를 저장합니다.
  - RGB render 외에 debug 이미지도 같이 저장합니다.

- `render_sets(...)`
  - train/test split 전체 렌더를 수행합니다.

### 현재 지원하는 출력

- `renders`
  - 일반 RGB render
- `gt`
  - ground truth image
- `foreground_scores`
  - learned foreground score render
- `foreground_overlay`
  - RGB 위에 foreground score를 겹친 디버그 이미지
- `object_mask_prior`
  - camera가 들고 있는 mask prior

### 현재 중요한 옵션

- `--object_id`
  - grouped model에서 특정 object만 렌더
- `--foreground_threshold`
  - foreground score가 일정 값보다 낮은 Gaussian을 렌더에서 제외

`--foreground_threshold`는 object mask prior와 foreground score는 잘 맞는데
RGB render 가장자리에 halo가 남는 경우를 줄이는 데 유용합니다.

## 7. [assign_object_ids.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/assign_object_ids.py)

외부에서 만든 object id 배열을 Gaussian scene에 붙일 때 쓰는 스크립트입니다.

### 용도

- 수동 clustering 결과 적용
- 외부 grouping 결과 적용
- physics export용 grouped model 생성

논문 기준 핵심 stage라기보다 보조 도구에 가깝습니다.

## 8. [auto_assign_object_ids.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/auto_assign_object_ids.py)

2D mask를 여러 뷰에 투영해 Gaussian별 object id를 자동 할당합니다.

### 현재 역할

- multi-view voting 기반 post-hoc grouping
- occlusion-aware frontmost filtering
- confidence-weighted voting
- boundary downweighting
- multi-view consistency refinement

### 해석 주의

이 파일은 여전히 유용하지만, 논문 정렬 관점에서 보면 주 학습 경로라기보다 fallback 또는 bridge에 더 가깝습니다.
최근 구현의 중심은 `train.py` 안의 object-aware loss와 `gaussian_model.py`의 foreground score입니다.

## 9. [export_physics_scene.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/export_physics_scene.py)

grouped Gaussian scene을 physics stage용으로 export합니다.

### 출력

- `physics_scene.json`
- `physics_scene_arrays.npz`

### 현재 의미

scene initialization 결과를 rigid-friendly intermediate representation으로 넘기는 입구입니다.
아직 differentiable rigid-body dynamics 자체는 없습니다.

## 10. [extract_sam2_features.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_sam2_features.py)

이미지에서 SAM2 feature를 뽑아 `.npy`로 저장합니다.

### 중요한 점

- `--output_channels`를 지원합니다.
- richer feature supervision을 위해 3채널 고정이 아닙니다.
- `train.py`의 `compute_sam_feature_loss`와 직접 연결됩니다.

## 11. [build_visual_hull.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/build_visual_hull.py)

multi-view mask와 camera를 이용해 visual hull seed point cloud를 생성합니다.

### 현재 의미

object-aware initialization 이전 단계의 strong prior 역할을 합니다.
지금 구현에서 object separation을 학습 안으로 더 넣더라도, visual hull는 여전히 중요한 seed source입니다.

## 12. [estimate_masked_colmap.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/estimate_masked_colmap.py)

foreground mask를 반영한 이미지로 COLMAP pose를 다시 추정합니다.

### 현재 의미

mask를 단순 시각화가 아니라 camera estimation 단계에도 연결하는 전처리 경로입니다.

## 13. [extract_object_masks.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_object_masks.py)

segmentation 모델이 없을 때 foreground mask를 자동으로 뽑는 보조 스크립트입니다.

### 현재 역할

- RGBA alpha 기반 foreground mask 추출
- background subtraction 기반 foreground mask 추출

이 출력은 `--masks_dir`로 학습이나 grouping에 바로 연결할 수 있습니다.

## 14. [prepare_instance_masks.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/prepare_instance_masks.py)

외부 segmentation 출력을 `auto_assign_object_ids.py`가 읽을 수 있는 형식으로 정리합니다.

### 현재 역할

- `generic`, `sam2`, `mask2former`, `deva` 입력 normalize
- label map / confidence map 통일

## 지금 코드를 읽는 추천 순서

- 학습 구조를 먼저 보고 싶으면
  - [`train.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)
  - [`scene/gaussian_model.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/gaussian_model.py)
  - [`scene/cameras.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/cameras.py)
  - [`utils/camera_utils.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/utils/camera_utils.py)

- object-aware separation이 어떻게 들어갔는지 보고 싶으면
  - [`train.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)
  - [`scene/gaussian_model.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/gaussian_model.py)
  - [`render.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/render.py)

- post-hoc grouping이 필요할 때만 보면 되는 파일
  - [`auto_assign_object_ids.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/auto_assign_object_ids.py)
  - [`assign_object_ids.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/assign_object_ids.py)

- physics export를 보고 싶으면
  - [`export_physics_scene.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/export_physics_scene.py)
