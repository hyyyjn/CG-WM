# SG-GS Stage 1 Implementation Explain

이 문서는 우리가 이번에 `gaussian_initiailization` 안에서 Stage 1을 논문 방향에 맞춰 구현/테스트하면서 바꾼 내용을 정리한 문서입니다.

기존 `EXPLAIN.md`가 파일별 코드 구조를 훑는 문서라면, 이 문서는 다음 내용을 중심으로 봅니다.

- 논문 Stage 1 관점에서 왜 수정했는지
- 어떤 파일을 어떻게 바꿨는지
- SAM2 feature와 Visual Hull의 역할이 무엇인지
- synthetic / mesh object 데이터셋을 어떻게 만들었는지
- 10k bunny color 실험을 어떻게 돌렸는지
- 결과 파일이 어디에 어떤 구조로 저장되는지
- SIBR viewer에서 왜 검정 화면이 나왔고 어떻게 해결했는지

## 0. 현재 기준 흐름

이 문서는 원래 `stage1 bunny synthetic` 실험 메모로 시작했지만,
현재 기준으로는 아래 흐름까지 포함해서 읽는 것이 맞습니다.

### 0.1 지금 메인 브랜치에서 실제로 쓰는 루프

현재 object-aware scene initialization의 대표 흐름은 아래입니다.

1. foreground mask 준비
   - 이미 있으면 `--masks_dir`로 사용
   - 없으면 `extract_object_masks.py`
2. 필요하면 masked COLMAP 재추정
   - `estimate_masked_colmap.py`
3. 필요하면 visual hull seed 생성
   - `build_visual_hull.py`
4. SAM2 feature 추출
   - `extract_sam2_features.py`
5. object-aware Stage 1 학습
   - `train.py`
   - `--sg_gs_stage1`
   - `--sam_feature_weight`
   - `--object_mask_weight`
   - `--masks_dir`
6. debug render 확인
   - `render.py`
   - `renders`
   - `foreground_scores`
   - `foreground_overlay`
   - `object_mask_prior`
7. 필요하면 `--foreground_threshold`로 tighter render 생성

즉 지금 구현의 중심은 더 이상 `post-hoc grouping` 하나가 아니라,
학습 안에 들어간 `object mask prior + foreground score` 경로입니다.

### 0.2 현재 object-aware 학습이 실제로 어떻게 동작하는가

현재 학습에서 object-aware 성분은 대략 아래 경로로 연결됩니다.

```text
mask prior
  -> Camera.object_mask
  -> compute_object_mask_loss
  -> Gaussian foreground_logit
  -> foreground score render
  -> foreground_threshold render
```

여기서 중요한 점은 두 가지입니다.

- 지금 구현은 `instance-level object decomposition`보다는 `foreground/background-aware separation`에 더 가깝습니다.
- 따라서 현재 단계에서는 `mask 품질`이 결과에 꽤 큰 영향을 줍니다.

### 0.3 현재 대표 실험

현재 메인 브랜치 기준 대표 실험은 `lego` object-aware 10k 학습입니다.

모델 경로:

```text
gaussian_initiailization/output/main_objaware_lego_10k
```

대표 출력:

```text
point_cloud/iteration_10000/point_cloud.ply
test_fgthr_0p0/ours_10000/renders
test_fgthr_0p0/ours_10000/foreground_scores
test_fgthr_0p0/ours_10000/foreground_overlay
test_fgthr_0p0/ours_10000/object_mask_prior
test_fgthr_0p5/ours_10000/renders
```

이때 보통은 아래 둘을 같이 봅니다.

- `test_fgthr_0p0`
  - 학습된 전체 object-aware 결과
- `test_fgthr_0p5`
  - foreground score가 낮은 Gaussian을 잘라낸 tighter 결과

### 0.4 현재 흐름에서 바로 보이는 한계

- foreground/background separation은 가능하지만, multi-instance separation은 아직 약함
- object-aware 결과는 mask 품질에 크게 영향받음
- halo를 줄이기 위해 render 시 threshold를 쓰는 경우가 많음
- 논문식 rigid-body dynamics stage는 아직 없음

아래 섹션들은 이 현재 흐름의 배경이 된 `stage1 strict`, synthetic bunny, viewer 대응 작업을 기록한 히스토리 메모로 읽으면 됩니다.

## 1. 현재 목표

현재 목표는 논문 전체 pipeline을 전부 구현하는 것이 아니라, 먼저 Stage 1을 안정적으로 구현하는 것입니다.
현재는 이 Stage 1 위에 object-aware separation 실험까지 일부 얹힌 상태입니다.

지금 구현/테스트한 범위는 다음과 같습니다.

- object-centric synthetic / mesh object dataset 생성
- mesh 표면에서 초기 `points3d.ply` 생성
- 이미지별 mask 생성
- 이미지별 SAM2 feature map 생성
- SG-GS Stage 1 strict 학습 모드 추가
- geometry / appearance alternating optimization 확인
- color가 보이는 bunny synthetic dataset으로 10k 학습 확인
- Python `render.py` 결과 확인
- SIBR viewer용 PLY 변환본 생성

아직 구현하지 않은 범위는 다음과 같습니다.

- real-world object용 Visual Hull 생성
- 실제 장면에서 object segmentation 후 object별 Gaussian 분리
- contact graph 학습
- object dynamics 학습
- robot / free-fall trajectory 기반 world model 학습
- rollout / downstream control

## 2. Visual Hull과 SAM2의 역할

헷갈렸던 부분을 먼저 정리합니다.

### Visual Hull

Visual Hull은 여러 카메라 view의 object mask를 3D 공간으로 back-project해서, 여러 silhouette가 공통으로 허용하는 3D 부피를 만드는 방식입니다.

논문 Stage 1에서 real-world object는 mesh가 없기 때문에, 초기 3D point cloud를 그냥 mesh에서 샘플링할 수 없습니다. 그래서 real-world object 쪽에서는 대략 다음 흐름이 필요합니다.

1. 여러 view의 RGB 이미지 준비
2. camera pose 준비, 보통 COLMAP 또는 calibrated camera
3. 각 view에서 object mask 준비
4. mask들을 이용해 Visual Hull 생성
5. Visual Hull 내부/표면에서 dense point cloud 생성
6. 이 point cloud를 `points3d.ply`로 사용
7. Stage 1 Gaussian initialization 시작

즉 Visual Hull은 real-world object에서 `points3d.ply`를 만들기 위한 초기 geometry 수단입니다.

현재 우리는 Stanford Bunny mesh를 사용했기 때문에 Visual Hull을 만들 필요가 없었습니다. mesh가 있으므로 mesh surface에서 바로 `points3d.ply`를 샘플링했습니다.

### SAM2

SAM2는 Visual Hull 자체를 만드는 도구로 구현한 것이 아닙니다.

현재 코드에서 SAM2는 이미지별 feature map을 추출해서 Stage 1 geometry supervision에 넣는 역할입니다.

현재 흐름은 다음과 같습니다.

1. `extract_sam2_features.py`가 각 이미지에서 SAM2 feature를 추출
2. 결과를 `sam_features/train/*.npy`, `sam_features/test/*.npy`에 저장
3. `train.py`가 각 camera view를 로드할 때 해당 feature map도 같이 로드
4. geometry step에서 Gaussian geometry feature rendering과 SAM2 feature를 비교
5. `--sam_feature_weight`로 loss 가중치를 줌

정리하면 다음과 같습니다.

- Visual Hull: real-world object의 초기 3D point cloud 생성용
- SAM2 feature: Stage 1 geometry feature supervision용
- SAM2 mask: Visual Hull을 만들 때 object mask 후보로 쓸 수는 있지만, 현재 구현에서 object-aware 분리는 주로 `mask prior + object mask loss` 경로를 통해 반영됨

## 3. Stage 1 strict 모드

`--sg_gs_stage1` 옵션은 논문 원문 용어가 아니라, 우리가 구현 상태를 명확하게 하기 위해 만든 strict mode입니다.

기본 3DGS 학습은 RGB reconstruction을 중심으로 사진을 맞추는 방향입니다. 반면 논문 Stage 1에서는 geometry와 appearance를 분리해서 object Gaussian을 초기화하는 것이 중요합니다.

그래서 `--sg_gs_stage1`를 켜면 다음을 강제합니다.

- alternating optimization 켜기
- joint optimization 끄기
- SAM2 feature 필수화
- `geometry_rgb_weight = 0.0`
- `sam_feature_weight > 0` 필수

의도는 다음과 같습니다.

- geometry step이 RGB 색 맞추기에 끌려가지 않게 함
- geometry step에서 SAM2 feature / scale regularization / optional depth 쪽을 보게 함
- appearance step에서 RGB L1 + DSSIM + opacity regularization을 보게 함
- SAM2 feature 없이 Stage 1을 돌리고 성공한 것처럼 보이는 상황을 막음

## 4. 수정한 파일 전체 목록

현재 코드 수정 파일은 다음과 같습니다.

- `.gitignore`
- `README.md`
- `gaussian_initiailization/arguments/__init__.py`
- `gaussian_initiailization/train.py`
- `gaussian_initiailization/extract_sam2_features.py`
- `gaussian_initiailization/tools/render_object_views_blender.py`
- `gaussian_initiailization/tools/export_sibr_viewer_ply.py`
- `gaussian_initiailization/EXPLAIN2.md`

## 5. `.gitignore`

추가한 내용:

```gitignore
external/
```

이유:

SAM2 official repo와 checkpoint를 `external/sam2` 아래에 둘 수 있게 했습니다. SAM2 checkpoint는 용량이 크고 외부 dependency이므로 git에 올리지 않는 것이 맞습니다.

현재 local SAM2 관련 경로:

```text
external/sam2/
external/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

## 6. `arguments/__init__.py`

Stage 1 strict mode와 관련된 optimization option을 추가했습니다.

추가한 옵션:

```python
self.sg_gs_stage1 = False
self.geometry_rgb_weight = 1.0
self.require_sam_features = False
```

의미:

- `sg_gs_stage1`
  - 논문 Stage 1에 맞춘 strict mode
  - 켜면 SAM2 feature를 필수로 요구하고 geometry/appearance 분리 학습을 강제

- `geometry_rgb_weight`
  - geometry step에서 RGB reconstruction loss를 얼마나 허용할지 정함
  - 기본값은 기존 코드 호환을 위해 `1.0`
  - `--sg_gs_stage1`에서는 내부적으로 `0.0`으로 override

- `require_sam_features`
  - SAM2 feature map이 없으면 즉시 에러를 내게 하는 옵션
  - strict mode에서는 자동으로 켜짐

수정 위치에는 사용자가 요청한 방식대로 `# edit this` 주석을 남겼습니다.

## 7. `train.py`

Stage 1 strict mode의 핵심 수정이 들어간 파일입니다.

### 7.1 `--sg_gs_stage1` 처리

학습 시작 시 `opt.sg_gs_stage1`가 켜져 있으면 다음을 강제합니다.

```python
opt.alternating_optimization = True
opt.joint_optimization = False
opt.require_sam_features = True
opt.geometry_rgb_weight = 0.0
```

그리고 `full_args`에도 같은 값을 반영해서 `training_args.json`에 기록되도록 했습니다.

또한 다음 조건을 검사합니다.

```text
sam_feature_weight <= 0 이면 에러
```

이유:

SAM2 feature loss 없이 strict Stage 1을 돌리면 논문 의도와 달라집니다. 그래서 strict mode에서는 feature weight가 반드시 양수여야 합니다.

### 7.2 SAM2 feature 필수 검사

`--require_sam_features`가 켜져 있는데 dataset에 SAM feature directory가 없으면 바로 에러가 납니다.

또한 training loop에서 sampled view의 feature map이 없으면 `FileNotFoundError`를 냅니다.

이유:

일부 view만 feature가 없는데 학습이 조용히 진행되면, 결과가 이상해도 원인을 찾기 어렵습니다. strict mode에서는 이런 경우를 빠르게 막습니다.

### 7.3 geometry loss 변경

기존에는 geometry step에서도 RGB loss가 섞일 수 있었습니다.

현재는 다음 구조입니다.

```text
geometry_loss =
  geometry_regularization
  + geometry_rgb_weight * geometry_appearance_loss
  + optional depth loss
  + SAM2 feature loss
```

strict mode에서는 `geometry_rgb_weight = 0.0`이므로 geometry step에서 RGB appearance loss가 빠집니다.

이유:

geometry branch가 색/texture를 맞추느라 구조가 틀어지는 것을 줄이고, SAM2 feature 기반 geometry supervision에 더 집중시키기 위해서입니다.

### 7.4 `compute_sam_feature_loss`

`feature_mask is None`일 때도 동작하도록 ones mask를 만들어 처리했습니다.

이유:

synthetic dataset이나 SAM2 feature extraction 결과에서 별도 feature mask가 없는 경우에도 feature supervision을 적용할 수 있게 하기 위해서입니다.

### 7.5 sparse Adam radii bug 수정

non-decoupled sparse Adam 경로에서 다음 버그를 수정했습니다.

```text
radii.shape[0] -> densify_radii.shape[0]
```

이유:

densification용 radii tensor와 일반 render radii tensor가 분리되는 경로에서 shape 참조가 잘못될 수 있어서 수정했습니다.

## 8. `extract_sam2_features.py`

SAM2 feature extraction script를 Stage 1 dataset layout에 맞게 수정했습니다.

### 8.1 `json` import 추가

`transforms_train.json`, `transforms_test.json`에서 frame path를 읽기 위해 `json`을 import했습니다.

### 8.2 `--images_root`

추가 옵션:

```bash
--images_root images
```

기본값은 `images`입니다.

### 8.3 image collection 순서 변경

현재 이미지를 찾는 순서는 다음과 같습니다.

1. `source_path/images/<split>/*.png`
2. `source_path/<split>/*.png`
3. `source_path/transforms_<split>.json`의 frame paths

이유:

Blender / NeRF style dataset은 이미지 경로가 transforms JSON 안에만 들어있는 경우가 있습니다. 그래서 transforms 기반 dataset도 바로 SAM2 feature를 뽑을 수 있게 했습니다.

### 8.4 확장자 없는 frame path 처리

`transforms_*.json`의 `file_path`가 `./images/train/000001`처럼 확장자 없이 들어있는 경우도 `.png`를 찾아 처리합니다.

## 9. `tools/render_object_views_blender.py`

새로 추가한 Blender dataset generator입니다.

이 스크립트는 mesh object를 Stage 1 synthetic / mesh object dataset 형태로 변환합니다.

입력:

```text
OBJ / GLB / GLTF / FBX / PLY mesh
```

출력:

```text
dataset/
  images/train/*.png
  images/test/*.png
  masks/train/*.png
  masks/test/*.png
  transforms_train.json
  transforms_test.json
  points3d.ply
  stage1_dataset_summary.json
```

### 9.1 주요 옵션

```bash
--mesh_path
--output_path
--num_views
--test_hold
--resolution
--radius_scale
--lens
--elevation_min
--elevation_max
--point_count
--seed
--white_background
--material_mode
```

### 9.2 mesh import

지원 확장자:

```text
.obj
.glb
.gltf
.fbx
.ply
```

Blender 5.x와 구버전 import API 차이를 고려해서 `bpy.ops.wm.obj_import`, `bpy.ops.wm.ply_import`가 있으면 우선 사용하고, 없으면 old import operator를 사용합니다.

### 9.3 object normalization

import된 mesh의 bounding box를 계산하고, object가 원점 중심에 오도록 normalize합니다.

이유:

카메라를 spherical view로 돌릴 때 object가 화면 밖으로 나가지 않도록 하기 위함입니다.

### 9.4 camera sampling

`fibonacci_sphere` 방식으로 여러 카메라 view를 만듭니다.

현재 bunny test에서는 다음 설정을 사용했습니다.

```text
num_views = 72
test_hold = 8
num_train_views = 63
num_test_views = 9
resolution = 384
radius_scale = 5.0
lens = 28
elevation_min = -20
elevation_max = 55
point_count = 100000
```

### 9.5 mask 생성

렌더 이미지는 RGBA로 저장됩니다.

alpha channel을 읽어서 object mask를 만듭니다.

Blender 5에서 `write_still` 직후 Render Result buffer가 비는 경우가 있어서, 저장된 RGBA 이미지를 다시 load해서 alpha mask를 만드는 workaround를 넣었습니다.

이 위치에도 `# edit this` 주석을 남겼습니다.

### 9.6 `points3d.ply`

mesh surface triangle area에 비례해서 point를 sampling합니다.

출력 PLY는 다음 속성을 가집니다.

```text
x y z
nx ny nz
red green blue
```

이 PLY가 Stage 1 Gaussian initialization의 초기 point cloud가 됩니다.

### 9.7 `--material_mode`

추가한 material mode:

```text
gray
position_bands
```

`gray`:

- 기본 geometry 확인용
- 회색 object라 color optimization이 눈에 잘 안 보임

`position_bands`:

- color / appearance optimization 확인용
- object 위치에 따라 빨강, 초록, 파랑, 노랑 band material을 할당
- 초기 `points3d.ply` 색도 같은 규칙으로 저장

현재 10k bunny color 실험은 `position_bands`를 사용했습니다.

## 10. `tools/export_sibr_viewer_ply.py`

새로 추가한 SIBR viewer 전용 PLY 변환 스크립트입니다.

### 10.1 왜 필요한가

Stage 1 학습 결과 PLY에는 표준 3DGS field 외에 다음 field가 추가됩니다.

```text
f_geo_0
...
f_geo_8
object_id
```

Python `render.py`는 이 field들을 알고 읽습니다.

하지만 SIBR Gaussian Viewer는 보통 표준 3DGS PLY layout을 기대합니다.

표준 layout은 대략 다음 순서입니다.

```text
x y z
nx ny nz
f_dc_*
f_rest_*
opacity
scale_*
rot_*
```

우리 Stage 1 PLY는 중간에 `f_geo_*`, `object_id`가 끼어 있어서 SIBR이 `opacity`, `scale`, `rot` 값을 잘못 읽을 수 있었습니다.

그 결과 SIBR에서 검정 화면이 나오거나 Gaussian이 제대로 보이지 않았습니다.

### 10.2 변환 방식

`export_sibr_viewer_ply.py`는 원본 PLY에서 다음 field를 제거합니다.

```text
f_geo_0 ~ f_geo_8
object_id
```

그리고 SIBR이 읽을 수 있는 표준 3DGS field만 남깁니다.

중요:

이 변환본은 viewer 확인용입니다. Stage 1 학습/재사용용 원본은 `f_geo_*`가 들어있는 원본 model path를 사용해야 합니다.

## 11. 현재 dataset 생성 명령

Blender로 color bunny dataset을 만든 명령은 다음과 같습니다.

```bat
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python gaussian_initiailization\tools\render_object_views_blender.py -- ^
  --mesh_path gaussian_initiailization\output\stage1_assets\bunny\bunny\reconstruction\bun_zipper.ply ^
  --output_path gaussian_initiailization\output\stage1_bunny_color ^
  --num_views 72 ^
  --resolution 384 ^
  --radius_scale 5.0 ^
  --lens 28 ^
  --elevation_min -20 ^
  --elevation_max 55 ^
  --point_count 100000 ^
  --material_mode position_bands
```

생성 결과:

```text
stage1_bunny_color/
  images/
    train/*.png
    test/*.png
  masks/
    train/*.png
    test/*.png
  sam_features/
    train/*.npy
    test/*.npy
  points3d.ply
  transforms_train.json
  transforms_test.json
  stage1_dataset_summary.json
```

`stage1_dataset_summary.json` 기준:

```json
{
  "num_views": 72,
  "num_train_views": 63,
  "num_test_views": 9,
  "resolution": 384,
  "point_count": 100000,
  "material_mode": "position_bands"
}
```

## 12. SAM2 feature extraction 명령

현재 사용한 SAM2 checkpoint:

```text
external/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

실행 명령:

```bat
conda run -n gs python gaussian_initiailization\extract_sam2_features.py ^
  --source_path gaussian_initiailization\output\stage1_bunny_color ^
  --checkpoint external\sam2\checkpoints\sam2.1_hiera_tiny.pt ^
  --config configs\sam2.1\sam2.1_hiera_t.yaml ^
  --output_dir sam_features ^
  --output_channels 9
```

결과:

```text
Saved 72 SAM2 feature maps
```

각 `.npy` feature shape는 384x384 이미지에 맞춰 저장되며, channel 수는 `--output_channels 9`입니다.

학습 시에는 반드시 다음과 맞아야 합니다.

```text
--geometry_feature_dim 9
```

## 13. 10k Stage 1 학습 명령

실행 명령:

```bat
conda run -n gs python gaussian_initiailization\train.py ^
  --source_path gaussian_initiailization\output\stage1_bunny_color ^
  --model_path gaussian_initiailization\output\stage1_bunny_color_train_10k ^
  --sam_features sam_features ^
  --sam_feature_weight 0.1 ^
  --geometry_feature_dim 9 ^
  --iterations 10000 ^
  --eval ^
  --disable_viewer ^
  --quiet ^
  --sg_gs_stage1 ^
  --geometry_iters 25 ^
  --appearance_iters 25 ^
  --test_iterations 5000 10000 ^
  --save_iterations 5000 10000 ^
  --checkpoint_iterations 5000 10000 ^
  --resolution 1
```

학습 시간:

```text
약 3분
```

최종 output:

```text
gaussian_initiailization/output/stage1_bunny_color_train_10k
```

## 14. 10k render 명령

학습 직후 `render.py`로 test view를 렌더링했습니다.

```bat
conda run -n gs python gaussian_initiailization\render.py ^
  --source_path gaussian_initiailization\output\stage1_bunny_color ^
  --model_path gaussian_initiailization\output\stage1_bunny_color_train_10k ^
  --iteration 10000 ^
  --skip_train ^
  --eval ^
  --quiet ^
  --resolution 1
```

렌더 결과:

```text
stage1_bunny_color_train_10k/test/ours_10000/renders/*.png
stage1_bunny_color_train_10k/test/ours_10000/gt/*.png
```

대표 관찰:

- 색 band는 학습됨
- 3k보다 10k에서 색이 더 강하고 안정적으로 반영됨
- 아직 halo / blur가 있음
- strict Stage 1은 일반 RGB-only 3DGS처럼 사진을 끝까지 선명하게 맞추는 세팅은 아님
- 현재 결과는 object Gaussian initialization 확인용 결과로 보는 것이 맞음

## 15. 10k 저장 구조

현재 model path 구조:

```text
stage1_bunny_color_train_10k/
  cameras.json
  cfg_args
  exposure.json
  input.ply
  training_args.json
  chkpnt5000.pth
  chkpnt10000.pth
  events.out.tfevents.*
  point_cloud/
    iteration_5000/
      point_cloud.ply
    iteration_10000/
      point_cloud.ply
  test/
    ours_10000/
      renders/*.png
      gt/*.png
```

각 파일 의미:

- `input.ply`
  - dataset의 `points3d.ply`를 학습 시작용으로 복사한 초기 point cloud

- `chkpnt5000.pth`, `chkpnt10000.pth`
  - 학습 재개용 checkpoint

- `point_cloud/iteration_*/point_cloud.ply`
  - 해당 iteration의 Gaussian export
  - Stage 1 원본 PLY이므로 `f_geo_*`, `object_id` 포함

- `test/ours_10000/renders`
  - `render.py`가 만든 예측 이미지

- `test/ours_10000/gt`
  - 비교용 GT 이미지

- `events.out.tfevents.*`
  - TensorBoard log

## 16. SIBR viewer용 변환

10k 원본 PLY를 SIBR에서 바로 열면 black viewport가 나올 수 있습니다.

이유:

```text
Stage 1 PLY에는 f_geo_* / object_id가 포함됨
SIBR viewer는 표준 3DGS PLY layout을 기대함
opacity / scale / rotation field를 잘못 읽을 수 있음
```

그래서 다음 viewer-only output을 만들었습니다.

```text
gaussian_initiailization/output/stage1_bunny_color_train_10k_sibr
```

변환 명령:

```bat
mkdir gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr
copy gaussian_initiailization\output\stage1_bunny_color_train_10k\cameras.json gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr\cameras.json
copy gaussian_initiailization\output\stage1_bunny_color_train_10k\cfg_args gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr\cfg_args
copy gaussian_initiailization\output\stage1_bunny_color_train_10k\exposure.json gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr\exposure.json
mkdir gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr\point_cloud\iteration_10000

conda run -n gs python gaussian_initiailization\tools\export_sibr_viewer_ply.py ^
  --input gaussian_initiailization\output\stage1_bunny_color_train_10k\point_cloud\iteration_10000\point_cloud.ply ^
  --output gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr\point_cloud\iteration_10000\point_cloud.ply
```

SIBR 실행:

```bat
"C:\Users\Admin\CG-WM\gaussian_initiailization\SIBR_viewers\install\bin\SIBR_gaussianViewer_app_rwdi.exe" ^
  -m "C:\Users\Admin\CG-WM\gaussian_initiailization\output\stage1_bunny_color_train_10k_sibr" ^
  --iteration 10000
```

주의:

`stage1_bunny_color_train_10k_sibr`는 viewer 확인용입니다. 이후 Stage 1 feature를 사용하는 코드나 physics/object-level pipeline으로 이어갈 때는 원본 `stage1_bunny_color_train_10k`를 기준으로 봐야 합니다.

## 17. 현재 결과 해석

10k color bunny 결과에서 확인된 점:

- color / appearance optimization은 동작함
- RGB band가 Gaussian rendering에 반영됨
- geometry는 bunny silhouette를 유지함
- 3k보다 10k에서 color가 더 강하게 반영됨

남은 문제:

- halo / blur가 있음
- 표면이 GT처럼 선명하지 않음
- SIBR에서는 splat size / view / exposure 느낌 때문에 Python render와 다르게 보일 수 있음

가능한 다음 개선:

- appearance-only fine-tuning phase 추가
- `geometry_rgb_weight`를 strict mode가 아닌 실험에서 조금 올려보기
- densification/pruning schedule 조정
- opacity regularization 조정
- 더 많은 view 또는 더 높은 resolution 사용
- color material을 band가 아니라 texture로 바꿔 확인

## 18. 현재 상태 한 줄 요약

현재 구현은 논문 Stage 1 전체 중 synthetic / mesh object 경로를 로컬에서 재현하는 방향입니다.

real-world object 경로에 필요한 Visual Hull은 아직 구현하지 않았고, 현재는 mesh가 있는 bunny를 사용했기 때문에 mesh surface sampled `points3d.ply`로 Stage 1 Gaussian initialization을 진행했습니다.
