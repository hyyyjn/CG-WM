# Gaussian Initialization TODO

이 문서는 현재 `gaussian_initiailization` 코드 기준으로, ContactGaussian-WM scene initialization 재현에서
이미 된 것과 아직 부족한 것을 우선순위 중심으로 정리한 메모입니다.

## 이미 완료된 항목

- isotropic spherical Gaussian 제약
- 기본 3DGS 학습 / 저장 / render
- exposure optimization
- geometry / appearance decoupled optimization
- alternating / joint optimization 옵션
- geometry / appearance loss 분리
- decoupled optimizer state checkpoint 저장 / 복구
- old decoupled checkpoint 호환 resume
- `--disable_viewer` fallback
- SAM2 feature supervision 경로
- shared screen-space gradient 연결
- `sam_feature_normalization` 옵션 추가
- SAM2 feature extraction 스크립트
- configurable `--output_channels` 기반 richer SAM feature extraction
- configurable `--geometry_feature_dim` 기반 richer geometry feature supervision
- Blender synthetic split-aware SAM feature loading
- Gaussian별 `object_id` 저장 / 복구 / export
- manual / automatic object grouping
- automatic foreground object mask extraction
- object-only render
- physics stage intermediate export
- rigid-friendly physics metadata export
- 외부 instance segmentation 결과 normalize helper
- variant 비교용 `compare_variants.py`
- densification statistics logging (`densification_stats.jsonl`)
- `compare_variants.py`의 densification summary 비교
- richer feature supervision smoke / 5k train+render 검증

## P0. 지금 바로 할 것

### 1. Richer SAM feature supervision 고도화

기본 richer feature 경로 자체는 구현됐습니다.
현재는 `--output_channels`, `--geometry_feature_dim`, multi-chunk supervision까지 붙어 있어서
3채널 축약 없이 실험은 가능합니다.

이제 남은 건 "더 좋은 richer feature"로 다듬는 단계입니다.

필요 작업:

- 3채널 평균 축약 대신 fixed projection 또는 learned projection 도입
- SAM2 feature의 multi-scale 활용 검토
- 3채널 vs 9채널 이상의 정량 비교 리포트 정리

완료 기준:

- richer feature 차원별 비교표와 추천 설정 확보

### 2. Feature-aware densification 분석

shared screen-space gradient 연결은 끝났지만,
실제로 feature loss가 split / clone / prune 판단에 얼마나 영향을 주는지는 아직 잘 안 보입니다.

현재 기본 densification 통계 로그(`densification_stats.jsonl`)와
`compare_variants.py`의 densification summary 비교는 추가되어 있습니다.
이제는 그 로그를 더 해석하기 쉽게 만들고, 장기 실험 비교에 연결하는 단계가 남았습니다.

필요 작업:

- densification 직전 gradient 통계를 branch별로 기록
- feature supervision 유무에 따른 split / prune 개수 비교
- 어떤 iteration에서 geometry prior가 성장 규칙을 바꾸는지 로그화

완료 기준:

- feature-aware densification이 실제로 작동하는지 실험 로그로 확인 가능

### 3. 장기 비교 실험 자동화

1k smoke 비교와 5k 3채널/9채널 train+render 검증은 했지만,
5k~10k 장기 학습에서 안정적인 개선인지에 대한 정량 비교는 아직 부족합니다.

필요 작업:

- baseline / SAM-init / shared-grad 변형 비교 스크립트 확장
- train/test PSNR뿐 아니라 SSIM / LPIPS도 함께 수집
- checkpoint 시점별 성능 추적
- 3채널 vs 9채널 richer feature의 장기 성능 비교 자동화

완료 기준:

- 장기 학습 비교 결과를 한 번에 재현할 수 있는 실험 루프 확보

## P1. 다음 단계

### 4. Initial Gaussian 품질 개선

현재 초기 Gaussian은 3DGS 기본 초기화에 isotropic constraint를 얹은 형태에 가깝습니다.
ContactGaussian-WM 관점에서는 object-aware / geometry-aware 초기화 prior가 더 필요합니다.

필요 작업:

- object-aware initial Gaussian seeding 가능성 검토
- foreground / background prior를 초기화 단계에서 반영하는 방법 정리
- scale / opacity / geometry feature 초기값 민감도 실험
- 초기 Gaussian 분포가 이후 grouping 품질에 미치는 영향 분석

완료 기준:

- 기본 3DGS 초기화와 개선된 initialization을 정량 비교 가능

### 5. Geometry supervision 강화

지금은 geometry supervision이 SAM2 feature와 optional depth 위주입니다.
더 직접적인 geometry target이 추가되면 initialization 품질을 더 끌어올릴 수 있습니다.

후보:

- silhouette / alpha mask loss
- monocular depth prior 강화
- normal prior
- segmentation boundary consistency

필요 작업:

- `masks/`, `normals/`, `depths/` 같은 추가 입력 경로 정의
- 카메라 객체에 geometry target 확장
- geometry step에서 supervision 항목별 가중치 옵션 추가

완료 기준:

- geometry step에서 mask / depth / normal supervision을 선택적으로 조합 가능

### 6. SG-GS 효과 검증

isotropic spherical Gaussian 제약은 구현되어 있지만,
"실제로 더 좋은 scene initialization을 만드는가"는 아직 충분히 닫히지 않았습니다.

필요 작업:

- anisotropic baseline 대비 SG-GS 초기화 성능 비교
- densification 이후 spherical constraint 유지가 품질에 미치는 영향 분석
- 7k / 10k checkpoint 등 iteration별 품질 변화 추적
- shared screen-space gradient 연결 전후 비교 실험

완료 기준:

- SG-GS 유무에 따른 정량 비교표 확보

### 7. Decoupled optimization 효과 검증

구조와 loss 책임은 이전보다 정리됐지만,
실제로 언제 도움이 되는지에 대한 정량 실험이 더 필요합니다.

필요 작업:

- alternating / joint / baseline 모드 비교 실험 자동화
- decoupled optimization의 장단점 정리
- geometry / appearance branch별 gradient / loss 통계 기록

완료 기준:

- decoupled optimization이 유리한 설정과 불리한 설정을 정리한 비교 결과 확보

## P2. Object / Physics 연결 고도화

### 8. Object-level 시각화와 디버깅

기본 object-only render는 구현됐지만, grouped 결과를 더 잘 디버깅할 수 있는 도구가 여전히 부족합니다.

필요 작업:

- foreground / background 분리 render
- object id별 색칠 render
- object별 Gaussian subset 시각화

완료 기준:

- object 단위로 grouped 결과를 바로 확인 가능

### 9. Physics export 확장

scene initialization 결과를 이후 rigid-body / physics 단계로 연결하는 포맷은 생겼고,
기본 body frame / sphere+AABB proxy / default mass metadata도 들어갑니다.
이제는 그 metadata를 더 물리적으로 유용하게 다듬는 쪽이 남았습니다.

필요 작업:

- 더 나은 collision primitive 구성
- inertia / center of mass 추정 고도화
- object별 안정성 점검 통계 추가

완료 기준:

- physics stage에서 바로 쓸 수 있는 object metadata 확보

## P3. 평가 / 재현성 / 문서화

### 10. 평가 지표 확장

현재는 렌더 품질 중심으로 보기 쉽고, geometry quality를 체계적으로 재는 도구는 부족합니다.

필요 작업:

- mask IoU
- depth error
- object separation quality
- Gaussian scale / opacity distribution 분석
- geometry feature consistency 분석

완료 기준:

- rendering metric 외에 geometry initialization metric을 같이 기록

### 11. Resume / training ergonomics 보강

기본 checkpoint 복구는 되지만, 실험 편의성은 더 좋아질 수 있습니다.

필요 작업:

- mode mismatch 경고 메시지 개선
- config snapshot 정리
- feature extraction부터 training까지의 end-to-end helper script 추가

완료 기준:

- 다른 사람이 같은 절차를 문서만 보고 재현 가능

### 12. 저장소 정리 규칙 유지

필요 작업:

- README와 `.gitignore`를 계속 동기화
- output / cache / checkpoint 관리 규칙 문서화
- 실제 실험 예시 명령 축적

완료 기준:

- GitHub에는 코드와 문서만, 산출물은 로컬 보존이라는 원칙이 유지됨
