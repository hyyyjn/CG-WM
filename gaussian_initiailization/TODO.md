# Gaussian Initialization TODO

이 문서는 현재 `gaussian_initiailization` 코드에서 이미 구현된 것과, ContactGaussian-WM 방향으로 더 진행하기 위해 남아 있는 작업을 정리한 것입니다.

## 이미 완료된 항목

- isotropic spherical Gaussian 제약
- 기본 3DGS 학습 / 저장 / render
- exposure optimization
- geometry / appearance decoupled optimization
- alternating / joint optimization 옵션
- geometry / appearance loss 분리
- decoupled optimizer state checkpoint 저장 / 복구
- SAM2 feature supervision 경로
- SAM2 feature extraction 스크립트
- Blender synthetic split-aware SAM feature loading

## P0. Object-level 분리와 시각화

현재 가장 부족한 부분은 "scene 전체"가 아니라 "object만 떼어서 다루는 능력"입니다.

필요 작업:

- foreground / background 분리 경로 추가
- object id 또는 instance mask 입력 경로 정의
- object-only render 모드 추가
- object별 Gaussian subset export
- 간단한 시각화 도구 추가
  - 특정 object만 렌더
  - 배경 제거 렌더
  - object id별 색칠 렌더

완료 기준:

- 단일 object scene에서 foreground object만 따로 렌더 가능
- multi-object scene에서 object id 기준으로 Gaussian 필터링 가능

## P0. Rigid object grouping

ContactGaussian-WM는 rigid-body world model이므로, Gaussian을 object 단위로 묶는 계층이 필요합니다.

필요 작업:

- Gaussian별 object id 저장
- instance segmentation 또는 mask 기반 grouping
- object별 중심, 반경, bounding volume 계산
- background 분리

완료 기준:

- 각 Gaussian이 background 또는 object id를 가짐
- object 단위 통계와 export가 가능함

## P1. Geometry supervision 강화

지금은 geometry supervision이 SAM2 feature와 optional depth 위주입니다.
더 직접적인 geometry target이 추가되면 scene initialization 품질이 더 좋아질 수 있습니다.

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

- geometry step에서 mask/depth/normal supervision을 선택적으로 조합 가능

## P1. Geometry feature 표현 고도화

현재 geometry feature는 3채널 supervision에 맞춘 1차 구현입니다.

필요 작업:

- 더 높은 차원의 geometry feature 저장 방식 검토
- projection 또는 learned head 추가
- SAM2 feature의 multi-scale 활용
- feature normalization 전략 개선

완료 기준:

- 3채널 평균 축약 없이 더 풍부한 geometry feature supervision 실험 가능

## P1. Physics 단계로 넘기는 export

scene initialization 결과를 이후 rigid-body / physics 단계로 연결할 포맷이 아직 없습니다.

필요 작업:

- export script 추가
  - object id
  - centers
  - isotropic scale
  - opacity
  - appearance feature
  - geometry feature
- collision proxy 또는 bounding primitive 생성
- body frame 정의

완료 기준:

- physics stage가 읽을 수 있는 중간 파일 포맷 생성 가능

## P2. 평가 지표 확장

현재는 렌더 품질 중심으로 보기 쉽고, geometry quality를 체계적으로 재는 도구는 부족합니다.

필요 작업:

- mask IoU
- depth error
- object separation quality
- Gaussian scale / opacity distribution 분석
- geometry feature consistency 분석

완료 기준:

- rendering metric 외에 geometry initialization metric을 같이 기록

## P2. Resume / training ergonomics 보강

기본 checkpoint 복구는 되지만, 실험 편의성은 더 좋아질 수 있습니다.

필요 작업:

- mode mismatch 경고 메시지 개선
- current training mode를 결과 폴더에 명시 저장
- config snapshot 정리
- feature extraction부터 training까지의 end-to-end helper script 추가

완료 기준:

- 다른 사람이 같은 절차를 문서만 보고 재현 가능

## P3. 문서화와 정리

필요 작업:

- README에 환경 분리 이유와 설치 절차 계속 동기화
- `.gitignore` 정리
- output / cache / checkpoint 관리 규칙 문서화
- 실제 실험 예시 명령 축적

## 다음 구현 추천

가장 우선순위 높은 다음 단계:

1. object-only render / visualization
2. Gaussian별 object id 저장 구조
3. object grouping + export
4. physics stage용 intermediate format

즉, 다음부터는 "더 잘 학습시키는 것"보다 "학습된 scene을 object 단위로 쓸 수 있게 만드는 것"이 우선입니다.
