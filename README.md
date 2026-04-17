한양대학교 졸프

# CG-WM

ContactGaussian-WM의 `scene initialization`을 중심으로 실험 중인 저장소입니다.
현재 구현의 무게중심은 `gaussian_initiailization/` 아래에 있습니다.


## 현재 범위

지금 코드에서 직접 다루는 범위는 아래입니다.

- spherical Gaussian 기반 scene initialization
- geometry / appearance decoupled optimization
- SAM2 feature supervision
- object mask prior 기반 object-aware supervision
- learned foreground score와 debug render
- optional post-hoc grouping
- rigid-friendly physics export 직전 단계

즉, ContactGaussian-WM 전체를 완성한 저장소라기보다는
`stage 1 + 그 이후를 위한 연결 실험`에 가깝습니다.

## 문서

실제 구현과 가장 잘 맞는 문서는 아래 세 개입니다.

- [gaussian_initiailization/README.md](gaussian_initiailization/README.md)
  - 설치, 실행 방법, 주요 옵션, 최신 실험 경로
- [gaussian_initiailization/EXPLAIN.md](gaussian_initiailization/EXPLAIN.md)
  - 파일별 역할과 코드 흐름
- [gaussian_initiailization/TODO.md](gaussian_initiailization/TODO.md)
  - 현재 완료 항목과 다음 우선순위

보조 메모:

- [gaussian_initiailization/REPORT.md](gaussian_initiailization/REPORT.md)
- [gaussian_initiailization/EXPLAIN2.md](gaussian_initiailization/EXPLAIN2.md)

## 빠른 시작

학습과 렌더는 `gaussian_splatting` 환경 기준입니다.

```bash
conda env create -f gaussian_initiailization/environment.yml
conda activate gaussian_splatting
```

기본 scene initialization 예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/scene_init \
  --iterations 10000 \
  --eval \
  --disable_viewer
```

object-aware supervision 예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/train.py \
  --source_path /path/to/scene \
  --model_path gaussian_initiailization/output/scene_objaware \
  --sam_features sam_features_sam2 \
  --sam_feature_weight 0.1 \
  --geometry_feature_dim 9 \
  --masks_dir /path/to/masks \
  --object_mask_weight 0.1 \
  --iterations 10000 \
  --eval \
  --disable_viewer \
  --sg_gs_stage1
```

렌더 예시:

```bash
conda run -n gaussian_splatting python gaussian_initiailization/render.py \
  --source_path /path/to/scene \SIBR_viewers
  --model_path gaussian_initiailization/output/scene_objaware \
  --iteration 10000 \
  --skip_train \
  --eval
```

## GitHub 업로드 원칙

커밋 대상:

- 코드
- 문서
- 설정 파일

커밋 제외 대상:

- `gaussian_initiailization/output/`
- `sam_features_sam2/`
- `masked_colmap/`, `visual_hull/`, `physics_export/`
- build log, cache, `__pycache__`
- checkpoint, render 이미지, 중간 산출물
