# Gaussian Initialization TODO

이 문서는 현재 `gaussian_initiailization` 코드 기준으로,
ContactGaussian-WM scene initialization 재현에서 이미 된 것과 아직 남은 것을 우선순위 중심으로 정리한 메모입니다.

## 현재 기준으로 이미 완료된 항목

- isotropic spherical Gaussian 제약
- 기본 3DGS 학습 / 저장 / render
- geometry / appearance decoupled optimization
- alternating / joint optimization 옵션
- geometry / appearance loss 분리
- exposure optimization 및 checkpoint 복구
- old checkpoint 호환 resume
- `--disable_viewer` fallback
- SAM2 feature supervision 경로
- richer `--geometry_feature_dim` supervision
- split-aware SAM feature loading
- visual hull seed initialization
- masked COLMAP 재추정 경로
- Gaussian별 `object_id` 저장 / 복구 / export
- manual / automatic object grouping
- physics stage intermediate export
- densification statistics logging
- object mask prior loading
- object mask BCE + L1 geometry supervision
- per-Gaussian learned foreground score
- foreground score PLY save/load 및 checkpoint round-trip
- foreground-aware debug render
- `--foreground_threshold` 기반 tighter render
- `lego_objaware_mask_10k` 10k train + render 검증

## P0. 지금 바로 이어갈 것

### 1. Foreground 억제와 경계 정리

object mask prior와 foreground score는 꽤 잘 맞기 시작했지만,
RGB render에서는 여전히 경계 주변 halo나 잔여 Gaussian이 보일 수 있습니다.

필요 작업:

- foreground threshold를 render 전용이 아니라 학습 regularization에도 연결
- 낮은 foreground score를 가진 Gaussian의 opacity 억제
- boundary 근처에서 foreground score를 더 날카롭게 만드는 regularization
- foreground/background-aware pruning 규칙 추가

완료 기준:

- object-only render에서 halo가 크게 줄고, threshold를 과하게 올리지 않아도 object가 깔끔하게 보임

### 2. Post-hoc grouping 의존 줄이기

현재는 논문에 더 가까운 object-aware initialization이 어느 정도 들어갔지만,
여전히 multi-object 실험이나 physics export에서는 `auto_assign_object_ids.py`에 기대는 부분이 남아 있습니다.

필요 작업:

- foreground score만으로 foreground/background 분리 export 가능하게 만들기
- `object_id` 없는 상태에서도 object-aware render / metric 계산 가능하게 만들기
- grouping을 핵심 경로가 아니라 fallback 경로로 문서와 코드에서 더 분명히 분리

완료 기준:

- 단일 object foreground 분리는 grouping 없이 확인 가능
- grouping은 multi-object physics export가 필요할 때만 선택적으로 사용

### 3. Object-aware 평가 지표 추가

지금은 `foreground_scores`, `object_mask_prior`, RGB render를 눈으로 비교하기는 쉽지만,
분리 품질을 정량적으로 재는 루프는 아직 약합니다.

필요 작업:

- mask IoU
- precision / recall on foreground occupancy
- foreground score histogram
- threshold sweep 결과 저장
- render 품질과 separation 품질을 함께 요약하는 비교 스크립트

완료 기준:

- 한 실험 폴더에서 reconstruction 품질과 separation 품질을 같이 읽을 수 있음

## P1. 논문 정렬을 위해 다음에 할 것

### 4. Initial Gaussian을 더 object-aware하게 만들기

지금은 학습 도중 object-aware signal이 들어가지만,
초기 seed 자체는 아직 visual hull / PCD 중심입니다.

필요 작업:

- foreground seed와 background seed를 다르게 주는 초기화 실험
- mask/visual hull를 scale, opacity 초기값에 더 직접 반영
- 초기 foreground logit priors 도입 검토

완료 기준:

- object-aware supervision이 초기 몇 백 iteration부터 더 안정적으로 작동

### 5. SAM2 supervision을 object separation 쪽으로 강화

현재 SAM2는 geometry feature supervision으로는 잘 연결돼 있지만,
object consistency까지 직접적으로 밀어주는 loss는 아직 약합니다.

필요 작업:

- cross-view object consistency loss 검토
- foreground/background feature contrastive loss 검토
- SAM2 feature와 mask prior를 함께 쓰는 hybrid geometry loss 설계

완료 기준:

- SAM2가 단순 feature prior를 넘어서 object separation quality에도 직접 기여

### 6. Richer feature supervision 장기 비교

3채널 대비 9채널 이상 supervision 경로는 구현됐지만,
장기 학습에서 얼마나 안정적으로 좋은지는 더 비교가 필요합니다.

필요 작업:

- 3채널 / 9채널 / 더 높은 채널 수 비교
- `sam_feature_weight` sweep
- 장기 학습에서 densification 차이 분석

완료 기준:

- 추천 feature 차원과 weight를 표로 정리 가능

### 7. SG-GS / decoupled optimization 효과 검증

구현은 되어 있지만, 어떤 설정에서 실제로 제일 유리한지 정량 비교는 더 필요합니다.

필요 작업:

- spherical vs baseline 비교
- alternating vs joint vs baseline 비교
- geometry / appearance branch별 loss와 gradient 통계 정리

완료 기준:

- 기본 설정 추천값을 문서로 고정할 수 있음

## P2. Object / Physics 연결 고도화

### 8. Multi-object separation을 학습 안쪽으로 더 끌어오기

지금 foreground score는 우선 foreground/background 분리 쪽에 가깝습니다.
여러 object instance를 학습 내부에서 직접 나누는 단계는 아직 멀었습니다.

필요 작업:

- multi-instance mask prior 설계
- Gaussian별 soft object embedding 또는 label distribution 검토
- 이후 rigid grouping과 자연스럽게 이어지는 representation 정의

완료 기준:

- 여러 object가 있는 장면에서 post-hoc grouping 없이도 instance-aware 구조가 드러남

### 9. Physics export 확장

scene initialization 결과를 이후 rigid-body / physics 단계로 연결하는 포맷은 이미 있지만,
메타데이터 품질은 더 좋아질 수 있습니다.

필요 작업:

- collision primitive 개선
- inertia / center of mass 추정 고도화
- object별 안정성 점검 통계 추가

완료 기준:

- physics stage에서 바로 쓰기 쉬운 object metadata 확보

## P3. 평가 / 재현성 / 문서화

### 10. 실험 자동화와 metrics 정리

필요 작업:

- train + render + metrics + debug summary를 한 번에 도는 helper 정리
- threshold별 render 자동 생성
- 주요 실험 설정별 결과 표 자동 저장

완료 기준:

- 다른 사람이 문서만 보고 같은 비교 실험을 재현 가능

### 11. README / EXPLAIN / TODO 동기화 유지

최근처럼 구현이 빠르게 바뀌면 문서가 쉽게 뒤처집니다.

필요 작업:

- 새 옵션 추가 시 README와 EXPLAIN 동시 반영
- 완료된 TODO를 바로 이동
- 대표 실험 결과 경로를 최신 상태로 유지

완료 기준:

- 문서가 실제 코드 상태를 계속 따라감
