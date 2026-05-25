# Stage 1/2 Validation Fix — 수정 사항 정리

> 브랜치: `stage1_2_validation_fix` (base: `stage2_collision_cleanup`)

## 이전 브랜치 대비 이번에 새로 수정한 것

`stage2_collision_cleanup`에서 좌표 contract, object mask, floor normal 등 기초를 잡았고, 이번 브랜치에서는 실제 sphere demo를 end-to-end로 돌렸을 때 나온 문제들을 추가로 수정했다.

| 항목 | 이전 브랜치 | 이번 브랜치 |
|------|------------|------------|
| `sphere_solref` | `-1000 0` (MuJoCo 불안정, floor 관통) | `"0.02 0.2"` (안정, restitution ≈ 0.28) |
| Stage 2 restitution 수렴 | v0와 e를 동시에 학습 → gradient vanishing | `--freeze_initial_velocity`로 v0 고정, e만 학습 |
| `run_demo.ps1` | `conda run --no-capture-output` → 19회 중복 실행 | `--no-capture-output` 제거 |
| Stage 1 PLY 경로 | `-SkipStage1` 시 iteration 불일치하면 에러 | 최신 checkpoint 자동 fallback |
| stage1_centroid | `"0,0,0.08"` 하드코딩 | object type별 `cz` 동적 반영 |
| 비교 렌더링 | 없음 | `render_trajectory_comparison.py` 추가 (GT vs 3DGS GIF) |
| 검증 결과 | sphere end-to-end 미확인 | `position_rmse = 0.013 m`, `learned_restitution = 0.285` 확인 |

## 전체 요약

Stage 1에서 만든 3DGS PLY를 Stage 2 물리 fitting에 안정적으로 연결하고, sphere demo를 실제로 돌려 수렴하는 것을 확인했다. 핵심 문제는 두 가지였다 — MuJoCo sphere GT가 바닥을 관통하는 것, Stage 2 optimizer의 gradient vanishing.

## 주요 수정

### Stage 1 object mask 적용

- 기본 학습 경로에서도 `object_mask_loss`가 실제 loss에 더해지도록 수정했다.
- 이전에는 decoupled optimization 경로가 아니면 `--object_mask_weight`를 줘도 object mask loss가 반영되지 않을 수 있었다.
- 이 문제 때문에 Stage 1 PLY에 바닥/배경 Gaussian이 섞이고, Stage 2 collision proxy가 커지는 문제가 생길 수 있었다.

### Stage 1 PLY audit 추가

- `gaussian_initiailization/tools/audit_stage1_ply.py`를 추가했다.
- foreground Gaussian bbox가 manifest의 metric bbox와 어느 정도 맞는지 확인한다.
- 낮은 iteration의 빠른 실험에서는 audit이 실패할 수 있으므로 `run_demo.ps1`에서는 현재 warn-only로 둔다.

### Stage 1 / Stage 2 좌표 contract

- object asset과 episode manifest에 `stage1_gaussian_body`를 기록한다.
- Stage 2는 이 manifest contract를 기준으로 world-frame Stage 1 PLY를 object-local frame으로 변환한다.
- 암묵적인 bbox recentering은 기본 동작에서 제거했다.

### Collision proxy 정리

- `collision_bbox_margin_z_ratio` 같은 수동 margin 보정은 논문 방법이 아니라 box demo를 위한 실험적 보정으로 취급한다.
- 대신 `floor_clip_collision_proxy`를 추가해서, manifest bbox 기준 바닥 아래로 새는 Gaussian만 제거할 수 있게 했다.
- 전체 proxy를 세로로 압축하지 않으므로 위쪽/측면 geometry를 덜 왜곡한다.

### Sphere dataset 안정화

- `generate_mujoco_fall_dataset.py`의 sphere 기본 `solref`를 `"0.02 0.2"`로 변경했다.
  - 기존 `"-1000 0"` 설정은 음수 timeconst가 `mjMINVAL`에 clamp되어 MuJoCo constraint가 불안정해지고, sphere 중심이 바닥 아래까지 내려가는 비정상 GT가 만들어졌다.
  - `"0.02 0.2"` (timeconst=20ms, dampratio=0.2)는 안정적이면서도 관찰 가능한 반발을 만든다 (restitution ≈ 0.28).
- sphere rollout에서 바닥 관통이 너무 크면 dataset 생성 단계에서 바로 실패하도록 `--max_floor_penetration` 검사를 추가했다.

### Floor contact normal 수정

- Stage 2 floor restitution/impedance dynamics에서 반발 법선을 Gaussian SDF 표면 법선이 아니라 floor plane normal로 고정했다.
- SDF는 penetration/contact gate 계산에 계속 사용한다.
- 이 수정으로 sphere proxy에서 bounce 이후 XY 위치가 폭주하던 문제가 크게 줄었다.

### Stage 2 gradient vanishing 수정 (`--freeze_initial_velocity`)

- Stage 2 optimizer가 initial velocity (`v0`)와 restitution (`e`)을 동시에 학습할 때, `v0`가 모든 frame에서 dense gradient를 가져 `e`의 gradient를 압도하는 문제가 있었다.
- `e`는 contact window (frame 13-30)에서만 gradient가 생기므로, Adam 기준으로 `v0`에 수백 step이 소모된 후에야 `e`가 움직이기 시작한다.
- `--freeze_initial_velocity` flag를 추가해서 `v0`를 GT `trajectory.json`에서 읽은 값으로 고정하고, optimizer는 `e`와 gravity만 학습하도록 했다.
- `--initial_velocity_source trajectory` 옵션과 함께 쓴다.
- 이 수정으로 500 Adam step 이내에 GT restitution에 수렴하는 것을 확인했다.

### `run_demo.ps1` conda 중복 실행 버그 수정

- `conda run --no-capture-output` 명령이 Windows에서 batch activation script의 CALL 루프로 인해 Python 프로세스가 N번 중복 실행되는 버그가 있었다.
- Step 1의 `[DONE]`이 21번 출력되고, Step 2의 train.py가 19회 이상 연속 실행되어 Stage 1 학습에 ~18분이 소요됐다.
- `Py` 함수에서 `--no-capture-output` 옵션을 제거해서 해결했다. 출력은 명령 완료 후 일괄 표시된다.

### 비교 렌더링 추가

- `gaussian_initiailization/tools/render_trajectory_comparison.py`를 추가했다.
- MuJoCo GT 영상과 Stage 2 predicted trajectory를 Stage 1 3DGS로 렌더링한 결과를 side-by-side GIF로 만든다.
- camera convention과 target orientation 반영을 맞췄다.

## 실행 방법

PowerShell에서 repository root 기준으로 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
```

기본 conda 환경 이름은 `gs`다. 다른 환경 이름을 쓰는 경우:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 -CondaEnv my_env_name
```

기존 Stage 1 checkpoint를 재사용하려면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 -SkipStage1
```

`-SkipStage1`을 쓸 때 요청한 iteration의 PLY가 없으면 가장 최신 `iteration_*` checkpoint를 자동으로 찾는다. 그래도 PLY가 없으면 manifest를 잘못 만들지 않고 즉시 실패한다.

Sphere demo 예시는 다음처럼 실행할 수 있다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1 -SceneName sphere_demo -ObjectType sphere -Stage1Iters 3000 -Stage2FitIters 120
```

## 확인한 결과

Box demo 기준으로는 여러 test episode에서 안정적인 trajectory fitting을 확인했다.

- component RMSE mean: 약 `0.0257 m`
- component RMSE max: 약 `0.0280 m`
- vector RMSE mean: 약 `0.0445 m`
- vector RMSE max: 약 `0.0486 m`

Sphere demo는 `sphere_solref="0.02 0.2"`, `--freeze_initial_velocity`, `Stage2FitIters=500` 기준으로 end-to-end 파이프라인 실행 결과:

- `learned_restitution`: 약 `0.285`
- `position_rmse`: 약 `0.013 m`
- `first_contact_frame`: 13
- `max_contact_gate`: 0.997 (contact detection 정상)

## 남은 한계

- 현재 demo는 여전히 floor contact 중심이다. object-object collision이나 회전까지 논문 수준으로 검증하려면 pairwise impedance mode와 orientation loss를 별도로 더 확인해야 한다.
- `collision_bbox_margin*` 옵션은 호환성 때문에 남아 있지만, 기본 demo에서는 쓰지 않는 방향이 맞다.
- demo 산출물인 `demo_data/`, `demo_output/`은 실험 결과이며 업로드 전에는 정리 대상이다.
