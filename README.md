# CG-WM (Gaussian Splatting) Documentation

## 프로젝트 개요
CG-WM은 ContactGaussian-WM 논문을 직접 구현해보는 것을 목표로 하는 실험용 저장소입니다.

## SubGoal1: Scene Initialization
- 3D Gaussian 모델 학습 및 테스트
- 렌더링 파이프라인 (train/test view)
- 자동화된 품질 평가 (PSNR 등)
- 구조 변경: geometry/appearance alternation
- 체크포인트 재개 및 optimizer 상태 보존

## 디렉터리 구조

`gaussian_initiailization/`
- `train.py`: 학습 루프, densification, optimizer, 로깅
- `render.py`: 검사/테스트 렌더링, GT 비교
- `scene/`: 데이터 로드, 카메라, GaussianModel
- `gaussian_renderer/`: 렌더러 wrapper + SH 변환
- `arguments/`: CLI 옵션, ModelParams/PipelineParams/OptimizationParams
- `output/`: 결과 모델, point cloud, 렌더 이미지 저장

`SIBR_viewers/`, `third_party/`:
- 서드파티 뷰어, 확장 라이브러리

## 설치

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git CG-WM
cd CG-WM/gaussian_initiailization
conda env create --file environment.yml
conda activate gaussian_splatting
pip install -e submodules/diff-gaussian-rasterization/
```

필요한 패키지
- PyTorch + CUDA (11.8 추천)
- PIL/pillow, torchvision, numpy, tqdm, plyfile

## 기본 사용법

### 학습
```bash
python train.py \
  --source_path <path-to-your-source> \
  --model_path <path-to-your-model> \
  --iterations 30000 \
  --eval --disable_viewer --quiet
```

### 렌더
```bash
python render.py \
  --model_path output/lego_test \
  --iteration 30000 \
  --resolution 4 \
  --skip_train
```

### alternate optimization (geometry/appearance 교차)
```bash
python train.py \
  --source_path /path/to/nerf_synthetic/lego \
  --model_path output/lego_test_alt \
  --iterations 30000 \
  --eval --disable_viewer --quiet \
  --alternating_optimization --geometry_iters 100 --appearance_iters 100
```

### joint optimization (geometry/appearance 동시 분리)
```bash
python train.py \
  --source_path /path/to/nerf_synthetic/lego \
  --model_path output/lego_test_joint \
  --iterations 30000 \
  --eval --disable_viewer --quiet \
  --joint_optimization
```

### 체크포인트 저장 및 재개
```bash
python train.py \
  --source_path /path/to/nerf_synthetic/lego \
  --model_path output/lego_resume \
  --iterations 30000 \
  --checkpoint_iterations 5000 10000 20000 \
  --eval --disable_viewer --quiet

python train.py \
  --source_path /path/to/nerf_synthetic/lego \
  --model_path output/lego_resume \
  --iterations 30000 \
  --start_checkpoint output/lego_resume/chkpnt10000.pth \
  --eval --disable_viewer --quiet
```

## 추가 기능 (GC-WM 커스텀)
- `--alternating_optimization` 옵션: geometry(xyz+scale) vs appearance(color+opacity) 단계별 학습
- `--joint_optimization` 옵션: geometry와 appearance를 분리된 optimizer로 매 iteration 함께 갱신
- `scene/gaussian_model.py`:
  - rotation freeze, scaling 등방성
  - geometry/appearance optimizer 분리
  - alternating/joint 모드에서도 `feature_lr`, `opacity_lr`, `scaling_lr`를 각각 유지
- `train.py`:
  - SAM2 시도 후 롤백 완료
  - 체크포인트 재개 시 optimizer/exposure 상태 복구

- `scene/gaussian_model.py`:
  - isotropic spherical Gaussian 제약 추가
  - `get_scaling`: 3축 scale을 평균값으로 묶어 `(s, s, s)` 형태로 사용
  - `get_rotation`: 학습된 rotation 대신 identity quaternion 사용
  - `get_covariance`: isotropic scaling + identity rotation 기준으로 계산
  - `densify_and_split`: 새로 생성되는 Gaussian도 동일한 sphere constraint 유지
  - geometry/appearance optimizer 분리
  - alternating/joint 모드에서도 `feature_lr`, `opacity_lr`, `scaling_lr`를 각각 유지

## 검증
1. 학습 및 렌더 실행 후 `output/{model}/test/ours_{iter}/renders` 폴더 확인
2. PSNR 계산 스크립트: `tools/` 미리 작성 가능

### 예시 PSNR 계산(간단)
```python
from PIL import Image
import numpy as np, os
render_dir='output/lego_test/test/ours_30000/renders'
gt_dir='output/lego_test/test/ours_30000/gt'
psnr=[]
for f in sorted(os.listdir(render_dir))[:20]:
    r=np.array(Image.open(os.path.join(render_dir,f)), dtype=np.float32)
    g=np.array(Image.open(os.path.join(gt_dir,f)), dtype=np.float32)
    mse=np.mean((r-g)**2)
    psnr.append(20*np.log10(255.0/np.sqrt(mse)))
print(np.mean(psnr), np.min(psnr), np.max(psnr))
```

## 문제 대응
- 출력이 전체 회색으로 퍼질 경우: SH 컬러 발산, `clamp`가 원인일 수 있음.
- 현재 실험 기준 10k/30k/50k에서 PSNR이 아직 낮은 편이며, alternating/joint는 기능 정합성은 맞췄지만 품질 튜닝은 계속 필요
- GPU 메모리 OOM 시 `--resolution 4` 사용

## 현재 검증 상태
- `render.py`:
  - `gaussian_initiailization/output/lego_test_joint` 기준으로 실행 확인
  - 기본 해상도는 현재 GPU(11.56GB)에서 OOM 가능
  - `--resolution 4` 옵션 사용 시 test view 200장 렌더 완료
- `train.py`:
  - `--joint_optimization` 기준 2 iteration smoke test 통과
  - `point_cloud`, `exposure.json`, `chkpnt*.pth` 생성 확인
- 체크포인트 재개:
  - `chkpnt2.pth`에서 이어서 1 iteration 추가 학습 성공
  - optimizer/exposure 상태 저장 및 복구 확인

### 바로 실행 가능한 검증 명령
```bash
python gaussian_initiailization/render.py \
  --model_path gaussian_initiailization/output/lego_test_joint \
  --iteration 20 \
  --skip_train \
  --resolution 4 \
  --quiet
```

```bash
python gaussian_initiailization/train.py \
  --source_path /home/cgr-ugrad-2026/Downloads/nerf_synthetic/lego \
  --model_path gaussian_initiailization/output/smoke_joint \
  --iterations 2 \
  --resolution 8 \
  --eval --disable_viewer --quiet \
  --joint_optimization \
  --test_iterations -1 \
  --save_iterations 2 \
  --checkpoint_iterations 2
```

## 커밋/수정 로그 요약
- `2026-04-03`: alternating optimization 구현 및 시범 적용
- `2026-04-03`: SAM2 추가 시도 후 롤백 완료
- `2026-04-03`: README 정리 (CG-WM 추가 안내)
- `2026-04-06`: scene initialization 실험을 위해 Gaussian의 회전을 고정하고 등방성 스케일을 적용하도록 수정


## 주의
현재 구현은 ContactGaussian-WM의 scene initialization 아이디어를 참고한 실험 버전이며,
논문의 decoupled optimization 전체를 완전히 재현한 상태는 아닙니다.
현재 단계에서는 isotropic spherical Gaussian constraint를 우선 반영했습니다.
---

추가적으로 구조적 개선(클램프 대신 `sh2rgb * C0 + 0.5`, `loss color regularization`)을 바로 코드에서 적용 가능합니다.


