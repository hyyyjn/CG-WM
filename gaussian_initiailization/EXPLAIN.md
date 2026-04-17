# Gaussian Initialization Code Explain

이 문서는 `gaussian_initiailization` 디렉터리의 핵심 코드 파일과 함수 역할을 빠르게 파악하기 위한 안내서입니다.
특히 ContactGaussian-WM의 scene initialization 관점에서

- 어디서 학습이 시작되는지
- geometry / appearance loss가 어디서 계산되는지
- object grouping / render / export가 어디서 이어지는지

를 한 번에 볼 수 있도록 정리했습니다.

## 전체 흐름

대략적인 실행 흐름은 아래와 같습니다.

1. `extract_sam2_features.py`
   - 이미지에서 SAM2 feature를 뽑아 `.npy`로 저장
2. `train.py`
   - Gaussian scene initialization 학습
3. `render.py`
   - 학습된 Gaussian을 train/test 뷰로 렌더
4. `assign_object_ids.py` / `auto_assign_object_ids.py`
   - Gaussian별 `object_id` 부여
5. `export_physics_scene.py`
   - object-level physics export 생성

핵심 데이터 객체는 두 가지입니다.

- `scene/cameras.py`의 `Camera`
  - 이미지, alpha, depth, SAM feature, camera pose를 한 묶음으로 보관
- `scene/gaussian_model.py`의 `GaussianModel`
  - Gaussian 파라미터와 optimizer, save/load, densification을 담당

## 1. [train.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)

학습 진입점입니다. scene initialization에서 가장 중요한 파일입니다.

### 주요 클래스 / 함수

- `_DisabledNetworkGUI`
  - viewer를 끄거나 import 실패했을 때 no-op 대체 객체로 사용합니다.
  - `--disable_viewer` 환경에서 socket 오류 없이 학습이 돌게 해줍니다.

- `append_jsonl(path, payload)`
  - JSONL 형식 로그를 파일에 한 줄씩 추가합니다.
  - 현재는 `densification_stats.jsonl` 저장에 사용됩니다.

- `zero_active_optimizers(gaussians, use_decoupled_optimization)`
  - 현재 사용 중인 optimizer들의 gradient를 초기화합니다.
  - baseline / decoupled optimization 둘 다 처리합니다.

- `compute_losses(render_pkg, viewpoint_cam, opt, depth_l1_weight_value)`
  - RGB reconstruction loss를 계산합니다.
  - 내부에서 L1, DSSIM, optional depth loss를 계산하고 반환합니다.
  - appearance / geometry step 모두 기본 reconstruction term으로 사용합니다.

- `compute_sam_feature_loss(viewpoint_cam, gaussians, pipe, bg, separate_sh, weight, shared_screenspace_points=None)`
  - geometry feature supervision을 계산합니다.
  - 현재 richer feature supervision 핵심 함수입니다.
  - `geometry_feature_dim > 3`인 경우 feature를 3채널 chunk로 나눠 여러 번 render 후 평균 loss를 만듭니다.
  - `shared_screenspace_points`를 받아 RGB render와 screen-space gradient를 공유합니다.

- `training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, full_args=None)`
  - 전체 학습 루프입니다.
  - 아래 일을 모두 담당합니다.
  - `Scene`, `GaussianModel` 생성
  - checkpoint 복구
  - camera sampling
  - geometry / appearance step 분기
  - densification / pruning
  - checkpoint 저장
  - 평가 타이밍 처리

- `prepare_output_and_logger(output_args, snapshot_args)`
  - output 폴더를 만들고 `cfg_args`, `training_args.json` 등을 저장합니다.

- `training_report(...)`
  - 중간 evaluation과 tensorboard logging을 담당합니다.

### 이 파일에서 특히 중요한 포인트

- decoupled optimization이 실제로 구현되는 곳
- geometry / appearance loss 분리가 적용되는 곳
- SAM feature loss가 geometry step과 densification에 연결되는 곳
- `densification_stat.jsonl`가 생성되는 곳

## 2. [scene/gaussian_model.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/gaussian_model.py)

Gaussian 파라미터와 학습 상태를 관리하는 핵심 모델 파일입니다.

### 주요 클래스

- `GaussianModel`
  - position, SH feature, geometry feature, opacity, scaling, rotation, object id, optimizer state를 모두 보관합니다.

### 생성 / 초기화 관련 함수

- `_expand_isotropic_scaling(scaling)`
  - scalar-like scale을 3축 isotropic scale로 확장합니다.

- `_identity_rotation(num_points, device)`
  - 모든 Gaussian의 rotation을 identity quaternion으로 만듭니다.

- `_get_isotropic_scaling(self)`
  - 내부 scaling 파라미터를 isotropic 형태로 읽습니다.

- `setup_functions(self)`
  - activation / inverse activation 등 Gaussian 파라미터 변환 함수를 설정합니다.

- `__init__(self, sh_degree, optimizer_type="default", geometry_feature_dim=3)`
  - 모델 생성자입니다.
  - 최근에 `geometry_feature_dim`을 받아 richer feature supervision과 연결되도록 확장됐습니다.

- `create_from_pcd(self, pcd, cam_infos, spatial_lr_scale)`
  - point cloud로부터 초기 Gaussian들을 생성합니다.
  - 초기 position, SH, geometry feature, scaling, opacity 등을 세팅합니다.

### save / load / checkpoint 관련 함수

- `capture(self)`
  - 현재 모델 상태를 checkpoint용 tuple로 묶습니다.

- `restore(self, model_args, training_args)`
  - checkpoint에서 모델과 optimizer 상태를 복구합니다.
  - old checkpoint compatibility도 여기서 처리합니다.

- `save_ply(self, path)`
  - Gaussian scene을 `.ply`로 저장합니다.
  - `f_geo_*`, `object_id` 등 커스텀 속성도 저장합니다.

- `load_ply(self, path, use_train_test_exp=False)`
  - `.ply`에서 Gaussian scene을 복구합니다.

### optimizer / parameter 관련 함수

- `training_setup(self, training_args)`
  - optimizer를 만들고 learning rate schedule을 설정합니다.

- `update_learning_rate(self, iteration)`
  - iteration에 따라 learning rate를 갱신합니다.

- `_get_gaussian_optimizers(self)`
  - 현재 optimizer 목록을 반환합니다.

- `_load_optimizer_state(self, optimizer, state_dict)`
  - optimizer state를 shape / group name 기준으로 안전하게 복구합니다.

- `geometry_parameters(self)`
  - geometry branch 파라미터 목록을 반환합니다.

- `appearance_parameters(self)`
  - appearance branch 파라미터 목록을 반환합니다.

- `exposure_parameters(self)`
  - exposure branch 파라미터 목록을 반환합니다.

- `set_parameter_requires_grad(self, geometry_enabled, appearance_enabled, exposure_enabled)`
  - branch별로 grad on/off를 제어합니다.

### getter 함수

- `get_scaling`, `get_rotation`, `get_xyz`
- `get_features`, `get_features_dc`, `get_features_rest`
- `get_geometry_features`
- `get_object_ids`
- `get_opacity`
- `get_exposure`, `get_exposure_from_name`
- `get_covariance`

이 함수들은 renderer나 학습 루프가 모델 상태를 읽을 때 사용합니다.

### densification / pruning 관련 함수

- `replace_tensor_to_optimizer(self, tensor, name)`
  - optimizer가 추적 중인 특정 파라미터 텐서를 새 텐서로 교체합니다.

- `_prune_optimizer(self, mask)`
  - pruning 후 optimizer state를 남은 Gaussian에 맞게 줄입니다.

- `prune_points(self, mask)`
  - 특정 Gaussian들을 실제로 제거합니다.

- `cat_tensors_to_optimizer(self, tensors_dict)`
  - 새 Gaussian을 추가할 때 optimizer state도 같이 확장합니다.

- `densification_postfix(...)`
  - clone / split 후 새 Gaussian들을 모델에 붙입니다.

- `densify_and_split(self, grads, grad_threshold, scene_extent, N=2)`
  - 큰 Gaussian을 쪼개는 split 단계입니다.

- `densify_and_clone(self, grads, grad_threshold, scene_extent)`
  - 작은 Gaussian을 복제하는 clone 단계입니다.

- `densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii)`
  - split / clone / prune를 한 iteration 묶음으로 수행합니다.
  - 현재는 통계도 같이 반환해서 `densification_stats.jsonl`에 기록됩니다.

- `add_densification_stats(self, viewspace_point_tensor, update_filter)`
  - screen-space gradient 통계를 누적합니다.

### object 관련 함수

- `set_object_ids(self, object_ids)`
  - Gaussian별 object id를 설정합니다.

## 3. [gaussian_renderer/__init__.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/gaussian_renderer/__init__.py)

렌더링 핵심 함수가 있는 파일입니다.

### 주요 함수

- `render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False, screenspace_points=None, gaussian_mask=None)`
  - CUDA rasterizer를 호출해 Gaussian scene을 렌더합니다.
  - 반환값에는 다음이 들어 있습니다.
  - `render`
  - `viewspace_points`
  - `visibility_filter`
  - `radii`
  - `depth`

### 이 함수에서 중요한 옵션

- `override_color`
  - geometry feature를 직접 RGB처럼 렌더할 때 사용합니다.
  - SAM feature supervision에서 핵심입니다.

- `screenspace_points`
  - 외부에서 넘겨주면 여러 render pass가 같은 2D projection gradient를 공유할 수 있습니다.

- `gaussian_mask`
  - 특정 Gaussian subset만 렌더합니다.
  - object-only render에서 사용됩니다.

## 4. [scene/cameras.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/cameras.py)

카메라와 per-view 입력 데이터를 관리합니다.

### 주요 클래스

- `Camera`
  - image, alpha, depth, SAM feature, pose, projection matrix를 한 객체에 담습니다.

- `MiniCam`
  - viewer나 custom render용 간단한 카메라 표현입니다.

### `Camera.__init__`

가장 중요한 초기화 함수입니다. 아래 일을 합니다.

- 이미지 resize와 tensor 변환
- alpha mask 생성
- train/test exposure split 처리
- SAM feature map resize / normalization
- depth map resize / 신뢰도 처리
- world-view / projection matrix 계산

### 최근 변경과 연결된 부분

- `sam_feature_normalization`
  - `none`
  - `per_view_minmax`
  - `clip_0_1`

- SAM feature map을 더 이상 3채널로 강제로 자르지 않음
- richer feature supervision을 위해 arbitrary channel count를 그대로 유지

## 5. [render.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/render.py)

학습된 Gaussian을 실제 이미지로 저장하는 스크립트입니다.

### 주요 함수

- `render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, object_id=None)`
  - 특정 split의 여러 뷰를 순회하며 render / gt 이미지를 저장합니다.

- `render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh, object_id=None)`
  - train/test split 전체를 렌더합니다.

### 특징

- `--object_id`를 지원합니다.
- grouped model에서 특정 object만 따로 렌더할 수 있습니다.

## 6. [assign_object_ids.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/assign_object_ids.py)

외부에서 만든 object id 배열을 Gaussian scene에 붙일 때 쓰는 스크립트입니다.

### 주요 함수

- `load_object_ids(path)`
  - `.npy` object id 배열을 읽고 shape를 확인합니다.

### 용도

- 수동 clustering 결과
- 외부 grouping 결과
- 디버깅용 object id 배열

을 grouped Gaussian model로 저장할 때 사용합니다.

## 7. [auto_assign_object_ids.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/auto_assign_object_ids.py)

2D mask를 여러 뷰에 투영해 Gaussian별 object id를 자동 할당하는 파일입니다.

### 주요 함수

- `encode_mask(mask)`
  - mask array를 compact JSON-friendly 형태로 바꿀 때 사용합니다.

- `load_mask(mask_path, width, height)`
  - mask를 읽고 카메라 해상도에 맞게 resize합니다.

- `load_confidence_map(confidence_path, width, height)`
  - confidence map을 읽고 resize / normalize합니다.

- `compute_boundary_weights(mask, ignored_labels, boundary_band_width, boundary_min_weight)`
  - object 경계 근처 픽셀 vote를 약하게 만드는 weight map을 계산합니다.

- `resolve_mask_path(...)`
  - 현재 뷰 이미지에 대응하는 mask 파일 경로를 찾습니다.

- `resolve_aux_path(...)`
  - confidence 같은 보조 파일 경로를 찾습니다.

- `parse_ignore_ids(ignore_ids_arg)`
  - 무시할 label id 목록을 파싱합니다.

- `project_gaussians(camera, xyz)`
  - Gaussian 중심을 이미지 평면으로 투영합니다.

- `compute_frontmost_mask(visible_idx, px, py, depth, width, height, depth_tolerance)`
  - 같은 픽셀의 후보들 중 가장 앞에 있는 Gaussian만 남깁니다.
  - occlusion-aware voting 핵심 함수입니다.

- `save_grouped_model(...)`
  - 새 object id를 반영한 grouped Gaussian model을 저장합니다.

- `accumulate_view_votes(...)`
  - 각 뷰의 mask label을 Gaussian별 vote로 누적합니다.
  - confidence, boundary downweighting, ignore label, occlusion filter가 이 안에서 적용됩니다.

- `assign_labels_from_votes(vote_counts, num_gaussians, background_id, min_votes)`
  - 누적 vote로 최종 object id를 결정합니다.

- `refine_object_ids(object_ids, vote_counts, per_view_observations, background_id, consistency_threshold, max_iterations)`
  - 초기 assignment 이후 multi-view consistency refinement를 수행합니다.

### 이 파일의 역할

- 2D segmentation 결과를 3D Gaussian object grouping으로 바꾸는 중심 파일

## 8. [export_physics_scene.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/export_physics_scene.py)

grouped Gaussian scene을 physics stage로 넘길 때 쓰는 export 스크립트입니다.

### 주요 함수

- `infer_source_track_id(object_id, background_id)`
  - export object의 source track id를 정합니다.

- `compute_rigid_metadata(center, bbox_min, bbox_max, radius, density, friction, restitution, dynamic)`
  - body frame, collision proxy, mass, inertia 같은 기본 rigid metadata를 계산합니다.

- `build_object_export(gaussians, object_id, background_id, density, friction, restitution)`
  - 하나의 object에 대한 JSON summary와 array bundle 항목을 만듭니다.

### 출력

- `physics_scene.json`
- `physics_scene_arrays.npz`

## 9. [extract_sam2_features.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_sam2_features.py)

이미지에서 SAM2 feature를 뽑아 `.npy`로 저장합니다.

### 주요 함수

- `parse_args()`
  - CLI 인자를 파싱합니다.

- `collect_images(split_dir)`
  - split 디렉터리 안의 이미지를 모읍니다.

- `reduce_feature_channels(feature_map, output_channels)`
  - raw SAM feature를 원하는 채널 수로 줄이거나 늘립니다.
  - 최근 richer feature supervision을 위해 `--output_channels`를 지원하게 확장됐습니다.

- `select_feature_map(predictor, feature_source)`
  - predictor 내부에서 어떤 feature를 쓸지 고릅니다.

- `resolve_output_dir(source_path, output_dir_arg)`
  - feature 저장 디렉터리를 결정합니다.

- `main()`
  - 실제 추출 루프입니다.

## 10. [extract_object_masks.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_object_masks.py)

segmentation 모델이 없을 때 foreground mask를 자동으로 뽑는 보조 스크립트입니다.

### 주요 함수

- `collect_frames(source_path, splits)`
  - 데이터셋에서 frame 목록을 수집합니다.

- `estimate_background_color(rgb)`
  - 배경색을 추정합니다.

- `normalize_confidence(confidence)`
  - confidence map을 0~1 범위로 맞춥니다.

- `extract_alpha_mask(image_rgba, alpha_threshold)`
  - RGBA alpha 기반 foreground mask를 만듭니다.

- `extract_bg_subtract_mask(...)`
  - background subtraction 기반 foreground mask를 만듭니다.

- `extract_mask(image_bgra, args)`
  - 실제 추출 방법을 선택하는 wrapper입니다.

## 11. [prepare_instance_masks.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/prepare_instance_masks.py)

외부 segmentation 결과를 `auto_assign_object_ids.py`가 읽을 수 있는 형식으로 정리하는 파일입니다.

### 주요 함수

- `encode_mask(mask)`
  - mask 저장용 보조 함수
- `load_array(path)`
  - `.npy` 또는 이미지 로드
- `normalize_confidence_map(array)`
  - confidence normalize
- `collect_dataset_frames(source_path, splits)`
  - 데이터셋 frame 목록 수집
- `build_search_roots(root_dir, split_name, input_format)`
  - 포맷별 탐색 경로 구성
- `is_valid_mask_candidate(path)`
  - 실제 mask 후보인지 판단
- `find_matching_path(...)`
  - frame에 대응하는 mask 파일 찾기
- `to_label_map(mask_array)`
  - 입력을 label map으로 통일
- `infer_channel_axis(array)`
  - channel axis 추정
- `logits_to_labels_and_confidence(array)`
  - logits / prob tensor를 label + confidence로 변환
- `load_json_score(path)`, `load_json_dict(path)`
  - sidecar JSON 로드
- `extract_int_from_name(path)`
  - 파일명에서 숫자 추출
- `infer_track_id(mask_file, metadata, preserve_track_ids)`
  - DEVA 같은 포맷에서 track id를 결정
- `combine_binary_masks(mask_dir, preserve_track_ids=False)`
  - 여러 binary mask를 하나의 label map으로 결합
- `load_mask_and_confidence(mask_path, input_format)`
  - 포맷별 입력을 통일된 mask/confidence로 읽기
- `maybe_resize(array, width, height, interpolation)`
  - 해상도 보정

### 역할

- `generic`, `sam2`, `mask2former`, `deva` 결과를 모두 normalization하는 adapter

## 12. [compare_variants.py](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/compare_variants.py)

실험 결과를 빠르게 비교하는 요약 스크립트입니다.

### 주요 함수

- `parse_cfg_args(cfg_path)`
  - 기존 `cfg_args` 파일을 읽습니다.

- `parse_training_args(training_args_path)`
  - 새 `training_args.json`을 읽습니다.

- `compute_psnr(render_path, gt_path)`
  - 간단한 PSNR을 직접 계산합니다.

- `summarize_split(model_dir, split_name)`
  - train/test split의 렌더 결과를 요약합니다.

- `load_results_json(model_dir)`
  - 기존 results 파일이 있으면 읽습니다.

- `load_densification_stats(model_dir)`
  - `densification_stats.jsonl`를 읽어 마지막 이벤트와 평균 통계를 요약합니다.

- `latest_iteration(model_dir)`
  - 가장 최근 checkpoint iteration을 찾습니다.

- `summarize_model(model_path)`
  - 한 실험 폴더의 전체 요약을 만듭니다.

- `print_summary_table(summaries)`
  - CLI 표 형태로 결과를 출력합니다.

## 파일별로 어디부터 읽으면 좋은가

- 학습 로직이 궁금하면:
  - [`train.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)
  - [`scene/gaussian_model.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/gaussian_model.py)
  - [`gaussian_renderer/__init__.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/gaussian_renderer/__init__.py)

- SAM feature supervision이 궁금하면:
  - [`extract_sam2_features.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/extract_sam2_features.py)
  - [`scene/cameras.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/scene/cameras.py)
  - [`train.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/train.py)

- object grouping이 궁금하면:
  - [`auto_assign_object_ids.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/auto_assign_object_ids.py)
  - [`assign_object_ids.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/assign_object_ids.py)
  - [`render.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/render.py)

- physics export가 궁금하면:
  - [`export_physics_scene.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/export_physics_scene.py)

- 실험 비교가 궁금하면:
  - [`compare_variants.py`](/home/cgr-ugrad-2026/work/CG-WM/gaussian_initiailization/compare_variants.py)

## 한 줄 요약

- `train.py`: 학습 오케스트레이션
- `gaussian_model.py`: Gaussian 상태와 densification
- `gaussian_renderer/__init__.py`: 실제 렌더
- `cameras.py`: view 입력 묶음
- `render.py`: 저장된 모델 이미지화
- `auto_assign_object_ids.py`: 2D mask -> 3D Gaussian grouping
- `export_physics_scene.py`: grouped Gaussian -> physics export
