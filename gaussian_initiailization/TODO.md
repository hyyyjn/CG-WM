# Scene Initialization TODO

이 문서는 현재 `gaussian_initiailization` 코드베이스를 ContactGaussian-WM의 scene initialization 단계에 더 가깝게 만들기 위해 필요한 작업을 정리한 것입니다.

기준:

- 현재 구현 완료:
  - isotropic spherical Gaussian 제약
  - 기본 3DGS 학습/렌더/저장
  - exposure 학습
  - densification / pruning
- 아직 부족한 부분:
  - geometry / appearance decoupled optimization
  - geometry-aware supervision
  - rigid-body / physics 단계로 넘길 수 있는 표현

## 우선순위

### P0. Geometry / Appearance 분리 최적화 구현

현재 코드는 하나의 Gaussian optimizer로 `xyz`, `feature`, `opacity`, `scaling`, `rotation`을 함께 업데이트합니다.
ContactGaussian-WM의 scene initialization 방향에 맞추려면 geometry와 appearance를 분리해서 제어할 수 있어야 합니다.

필요 작업:

- `arguments/__init__.py`
  - 아래 CLI 옵션 추가
  - `--alternating_optimization`
  - `--joint_optimization`
  - `--geometry_iters`
  - `--appearance_iters`
  - 필요 시 `--freeze_geometry_after`
  - 필요 시 `--freeze_appearance_after`
- `scene/gaussian_model.py`
  - geometry parameter group 정의
    - `xyz`
    - `scaling`
    - 필요 시 `rotation`은 현재 identity 고정이므로 제외 가능
  - appearance parameter group 정의
    - `features_dc`
    - `features_rest`
    - `opacity`
    - 필요 시 `exposure`
  - optimizer를 하나가 아니라 둘로 분리
    - `geometry_optimizer`
    - `appearance_optimizer`
  - optimizer state 저장/복구 지원
- `train.py`
  - alternating update 루프 추가
  - joint-but-decoupled update 루프 추가
  - 어떤 iteration에서 geometry step / appearance step을 할지 스케줄링
  - 현재 loss에서 geometry 관련 항과 appearance 관련 항을 분리할 준비

완료 기준:

- CLI에서 `--alternating_optimization` 또는 `--joint_optimization` 사용 가능
- checkpoint 저장/복구 시 분리 optimizer state까지 복원 가능
- 최소 2-iteration smoke test 통과

## P0. Checkpoint 복구 범위 확장

현재 checkpoint는 메인 optimizer state는 저장하지만 exposure tensor와 exposure optimizer 상태를 완전히 복구하지 않습니다.

필요 작업:

- `scene/gaussian_model.py`
  - `capture()`에 아래 항목 추가
    - `_exposure`
    - `exposure_optimizer.state_dict()`
    - decoupled optimization 도입 시 `geometry_optimizer.state_dict()`
    - `appearance_optimizer.state_dict()`
  - `restore()`에서 모두 복구하도록 수정
- `train.py`
  - resume 직후 학습이 실제로 이어지는지 smoke test 추가

완료 기준:

- `start_checkpoint`로 재시작했을 때 exposure와 optimizer 상태가 이어짐
- 재시작 후 1~2 iteration 추가 학습 성공

## P1. Geometry-aware supervision 추가

현재 scene initialization은 기본 RGB reconstruction 중심입니다.
ContactGaussian-WM 방향으로 가려면 geometry를 appearance와 분리해서 더 안정적으로 학습할 supervision이 필요합니다.

가능한 입력:

- SAM2 mask / feature
- depth prior
- normal prior
- silhouette / segmentation prior

필요 작업:

- 새로운 입력 경로 설계
  - 예: `source_path/geometry/`, `source_path/masks/`, `source_path/features/`
- `scene/dataset_readers.py`
  - geometry supervision 파일 로딩
- `utils/camera_utils.py`
  - 카메라별 geometry 입력을 `Camera` 객체로 전달
- `scene/cameras.py`
  - `mask`, `feature`, `geometry_target` 등을 멤버로 보관
- `train.py`
  - geometry loss 추가
    - silhouette consistency
    - mask alignment
    - depth consistency 강화
    - feature consistency

완료 기준:

- geometry supervision을 옵션으로 켜고 끌 수 있음
- geometry-only step에서 해당 supervision이 실제 loss에 반영됨

## P1. Geometry / Appearance loss 구조 분리

decoupled optimization을 넣더라도 loss가 완전히 분리되지 않으면 효과가 제한적입니다.

필요 작업:

- `train.py`
  - 아래 loss 항목을 명시적으로 분리
    - appearance loss
      - L1
      - SSIM
      - exposure 관련 항
    - geometry loss
      - depth
      - silhouette/mask
      - scale regularization
      - opacity regularization
  - geometry step에서는 appearance gradient 차단
  - appearance step에서는 geometry gradient 차단 또는 제한

추천 regularization:

- opacity collapse 방지
- scale explosion 방지
- geometry-only 단계에서 color feature drift 방지

완료 기준:

- 코드상 loss logging이 geometry / appearance 별도로 분리
- TensorBoard나 로그에 항목별 출력 가능

## P1. Scene Initialization 결과를 Physics 단계로 넘기기 위한 표현 정리

ContactGaussian-WM는 rigid-body world model이므로, scene initialization 결과가 물리 단계 입력으로 이어질 수 있어야 합니다.

현재 없는 것:

- object instance 단위 분리
- rigid body별 Gaussian group
- body frame / canonical frame
- mass / inertia / contact proxy 연결점

필요 작업:

- 새 모듈 추가 검토
  - 예: `scene/object_groups.py`
  - 예: `physics/init_state.py`
- object별 Gaussian index 관리
- 각 object에 대한 중심, 반경, bounding volume 계산
- collision geometry 추출 인터페이스 정의

완료 기준:

- 학습된 Gaussian들을 object 단위로 묶어 export 가능
- 이후 physics optimizer가 읽을 수 있는 중간 포맷 정의

## P2. Object-level rigid segmentation / grouping

scene initialization만으로는 부족하고, ContactGaussian-WM 전체 흐름을 생각하면 rigid body 단위 분리가 중요합니다.

필요 작업:

- mask/instance 기반 object assignment
- Gaussian별 object id 저장
- 학습 중 object별 통계 추적
- 배경 / 물체 분리

가능한 구현 방향:

- 입력 segmentation 사용
- Gaussian center와 feature 기반 clustering
- 후처리 단계에서 instance grouping

완료 기준:

- 각 Gaussian이 background 또는 object id를 가짐
- object별 export 가능

## P2. Physics refinement를 위한 인터페이스 초안 작성

scene initialization 구현 뒤 바로 다음 단계로 넘어가려면, 최소한 인터페이스 초안이 필요합니다.

필요 작업:

- 저장 포맷 설계
  - object id
  - Gaussian centers
  - isotropic scale
  - opacity
  - visual features
  - collision proxy
- export script 추가
  - 예: `export_scene_init.py`
- 문서화
  - 다음 stage에서 필요한 필드 명시

## P2. 평가 지표 보강

현재는 주로 렌더 품질 평가 중심입니다.
scene initialization 단계에서는 geometry robustness도 같이 봐야 합니다.

필요 작업:

- `metrics.py` 또는 별도 스크립트에 geometry 지표 추가
- 예시:
  - mask IoU
  - depth error
  - object separation quality
  - Gaussian scale distribution

## P3. 코드 정리 및 문서화

필요 작업:

- `README.md`와 실제 구현 상태 계속 동기화
- scene initialization 단계와 physics 단계의 경계 명시
- smoke test 명령 정리
- 결과 디렉터리 구조 문서화

## 바로 다음으로 구현 추천

가장 먼저 할 일:

1. `arguments/__init__.py`에 decoupled optimization용 CLI 추가
2. `scene/gaussian_model.py`에 geometry / appearance optimizer 분리
3. `train.py`에 alternating / joint update 루프 추가
4. checkpoint에 exposure 및 분리 optimizer state 저장

이 4개가 들어가야 현재 코드가 "단순 spherical 3DGS"에서 "ContactGaussian-WM scene initialization에 가까운 구현"으로 넘어갈 수 있습니다.

## 참고 메모

- 현재 isotropic spherical Gaussian 제약은 이미 구현됨
- 현재 README에 적혀 있던 alternating/joint optimization은 실제 구현되어 있지 않았음
- 따라서 다음 작업의 핵심은 "문서에 있던 방향을 실제 코드로 복구/완성"하는 것에 가깝다
