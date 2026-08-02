# Stage 2 쿼리 포인트 방식 비교 — 수정사항

접촉 감지의 **쿼리 포인트를 주는 방식**만 바꿔가며 Stage 2 물리 최적화를 비교하고,
예측 궤적이 MuJoCo GT를 따라가도록 접촉 동역학을 보강한 작업.

Stage 1 가우시안은 재학습하지 않고 기존 120k 학습 결과(`point_cloud.ply`)를 강체로 사용한다.
학습 대상은 물리 스칼라(초기속도, 중력, 반발계수, 마찰계수)뿐이다.

---

## 1. 쿼리 포인트 방식 (원래 목적)

`run_stage2_mujoco_stage1_fit.py`에 `--query_mode` 추가.

| 모드 | 쿼리 포인트 | 평가 방식 |
|---|---|---|
| `floor_disk` (기본) | 바닥 평면에 깔린 동심원 원판 161점 | 점들에서 물체의 union-of-spheres SDF를 평가 |
| `body_surface` (신규) | 가우시안 프리미티브 표면 1080점 | 점들을 월드로 변환해 바닥 평면과 평가 |

`body_surface`는 기존 `make_gaussian_proxy_query_points()`를 재사용한다. 물체가 회전하면
쿼리 포인트도 함께 움직이므로, 기울어진 실린더가 테두리(rim)로 착지할 때 실제 최저점을
잡는다. `floor_disk`는 XY 원판이라 이 상황에서 최저점을 놓친다.

관련 인자: `--body_query_dirs` (프리미티브당 방향 수, 6 = 축 방향, 그 외 = Fibonacci)

> 기본값이 `floor_disk`라 기존 동작은 그대로 유지된다.

## 2. 접촉점 마찰 (XY 드리프트 대응)

기존 시뮬레이터는 접촉 임펄스를 **바닥 법선 방향으로만** 가했다. 그래서 물체가 회전해도
옆으로 밀리지 않고 제자리에서만 튀었다 (XY RMSE 0.39/0.51).

회전하는 물체의 접촉점 속도는 `v + ω×r`이고, 마찰이 이 미끄러짐을 방해하면서
그 반작용이 물체를 병진 운동시킨다. 이를 구현:

- `angular_velocity_from_quaternions()` — GT 쿼터니언 시퀀스에서 유한차분으로 각속도 ω 추출
  (`dq = q_{t+1} ⊗ conj(q_t)`, ω = axis·angle/dt)
- `_floor_contact_response()` — 접촉 응답을 별도 함수로 분리하고 Coulomb 마찰 임펄스 계산

관련 인자:
- `--floor_friction_mode {off, fixed, learned}` (기본 `off`)
- `--floor_friction_init` — 초기(또는 고정) 마찰계수

**다중 접촉점.** 접촉 중인 점들을 평균해 하나로 뭉치지 않고, 점마다 자기 지렛대 팔 `r_i`로
`ω×r_i`를 계산하고 자기 몫의 수직 임펄스로 Coulomb 상한 `min(μ·j_n,i, |slip_i|)`을 정한 뒤
합산한다. 마찰은 슬립에 대해 비선형(방향 정규화 + min 클램프)이므로 `f(평균) ≠ 평균(f)`이고,
먼저 평균 내면 물리적으로 존재하지 않는 가상의 점에 대한 마찰을 계산하게 된다.

## 3. 서브스텝 적분

`--substeps` 추가 (기본 1, restitution 동역학에만 적용).

30fps 기준 한 스텝이 33ms인데 실제 충돌은 1~5ms 만에 끝난다. 3m/s로 낙하하면 33ms 동안
10cm를 이동하므로, 한 스텝 만에 물체가 바닥 위에서 아래로 통과해버려 임펄스가 뭉개진다.
한 프레임을 N등분해 각 서브스텝마다 접촉을 다시 감지하고 임펄스를 가한다.

## 4. 최적화 안정화

접촉 loss 지형에 **국소최소**가 존재한다. 마찰 초기값 0.5에서 출발하면 `e=0.30, μ=0.15`에
갇혀 총 RMSE 0.098에서 정체하고, 이때 z 바운스가 GT의 절반으로 죽는다.
`(e, μ)` 격자를 직접 스윕한 결과 `e≈0.55, μ≈0.10` 근처에 총 RMSE 0.048인 해가 존재함을 확인했다.
z와 XY는 **트레이드오프가 아니라** 둘 다 만족 가능하다.

- `--init_restitution` — 반발계수 초기값 (`--floor_friction_init`과 함께 수렴 basin을 결정)
- `--freeze_gravity` — 중력을 -9.81로 고정. 마찰·반발과 서로 보상하는 축퇴를 제거
- best-iteration 파라미터 복원 — loss가 진동할 때 마지막 iteration이 아니라 가장 좋았던
  파라미터를 채택
- `--log_every` — N iteration마다 position loss 출력 (수렴 진단용)

**권장 초기값**: `--init_restitution 0.55 --floor_friction_init 0.10`

## 5. 카메라 (`generate_mujoco_fall_dataset.py`)

기존에는 카메라 틸트가 하드코딩(고정 30°)이라 가까이 당기면 낙하 상단이 잘렸다.
일반 look-at으로 교체하고 조준점을 인자로 노출:

- `--camera_target_x`, `--camera_target_y`, `--camera_target_z`

`--camera_target_z`만 있던 부분 구현을 3축 일반형으로 확장했다. 카메라 위치
`(0, -distance, height)`에서 타겟을 향하는 forward/right/up을 계산해 `xyaxes`를 생성한다.

---

## 결과

에피소드: 콜라캔이 1.8m에서 낙하 → rim 착지 → 2회 바운스 → 직립 정착 (80프레임 사용)

| 구성 | x | y | z | 총 RMSE |
|---|---|---|---|---|
| 마찰 없음 | 0.394 | 0.507 | 0.082 / 0.029 | 0.374 / 0.373 |
| 마찰 (단일 접촉점) | 0.318 | 0.460 | 0.092 | 0.327 |
| + 다중접촉 · 서브스텝 | 0.058 | 0.068 | 0.144 | 0.098 |
| **+ 국소최소 탈출 (최종)** | **0.037 / 0.056** | **0.057 / 0.090** | **0.037 / 0.018** | **0.045 / 0.062** |

(두 값이 있는 칸은 `floor_disk / body_surface`)

**쿼리 방식 비교**: z축 RMSE `body_surface 0.018` vs `floor_disk 0.037`로 2배 우수.
z축은 쿼리 방식이 직접 작용하는 축이다. 바운스 정점은 floor 0.611 / body 0.580 (GT 0.568).

## 재현

```bash
# 1) GT 에피소드 생성
python  tools/generate_mujoco_fall_dataset.py \
  --dataset_root <DATA_ROOT> --object_name cola_can_dynamic --split train \
  --camera_distance 2.5 --camera_height 1.3 \
  --camera_target_x 0.72 --camera_target_y -0.62 --camera_target_z 0.32 \
  --drop_height_train 1.8 --max_tilt_deg_train 6 --spin_speed_train 2.0 \
  --planar_speed_train 0.35 \
  --cylinder_solref "0.01 0.2" --floor_solref "0.01 0.2" --seed 2

# 2) Stage 2 fit (쿼리 모드만 바꿔 2회)
python  tools/run_stage2_mujoco_stage1_fit.py \
  --episode_root <EPISODE> --stage1_ply <STAGE1_PLY> --output_dir <OUT> \
  --dynamics restitution --query_mode {floor_disk|body_surface} \
  --floor_friction_mode learned --floor_friction_init 0.10 --init_restitution 0.55 \
  --freeze_gravity --substeps 6 --floor_tangential_damping 0.0 \
  --max_frames 80 --fit_iters 600 --lr 0.02 \
  --foreground_threshold 0.99 --opacity_threshold 0.02 --max_primitives 180 \
  --radius_scale 0.1 --initial_velocity_source trajectory --freeze_initial_velocity
```

> MuJoCo 반발은 **두 geom의 solref를 섞는다**. 실린더만 설정하면 바닥 기본값에 희석되므로
> `--cylinder_solref`와 `--floor_solref` 양쪽에 같은 값을 줘야 한다.
> `"0.004 0.05"`처럼 과한 값은 에너지가 발산한다.

## 알려진 한계

- **회전은 예측하지 않는다.** 각속도 ω를 GT 자세 시퀀스에서 읽어온다. 즉 "정답 회전이
  주어졌을 때 그 회전이 만드는 병진 운동을 재현"하는 것이며, 완전 자립 롤아웃이 아니다.
  회전까지 시뮬레이션하려면 강체 동역학(`--dynamics pairwise_impedance`) 경로가 필요하다.
- **점 수가 다르다.** `body_surface`(1080점)가 `floor_disk`(161점)보다 6.7배 많다.
  방식 자체의 우열을 주장하려면 점 수를 맞춘 실험이 추가로 필요하다.
- `--query_mode body_surface`는 `--dynamics pairwise_impedance`와 함께 쓸 수 없다.
- `--floor_friction_mode`는 `--dynamics restitution`에만 적용된다.
- 정지 구간이 긴 에피소드는 안착 오차가 loss를 지배해 그래디언트가 죽는다.
  `--max_frames`로 모션 구간만 잘라 쓸 것.

---

## 6. 신규 쿼리 방식 `body_lowest_k` — 마찰 과함 해결 (2026-07-29)

`body_surface`(가우시안별 1080점)는 접촉 중인 **모든** 표면점의 `ω×rᵢ` 슬립을 합산하므로
접촉 패치가 몸체 전체로 번져 마찰이 과하게 걸린다(정지 꼬리·평면운동 과감쇠). 이를 검증/해결하려
`run_stage2_mujoco_stage1_fit.py`에 `--query_mode body_lowest_k` + `--body_lowest_k K` 추가.

`body_surface`와 동일하게 프리미티브 표면점을 월드로 변환하되, **매 스텝 월드 z가 가장 낮은 K개만**
`torch.topk`로 골라 접촉 패치로 쓴다(`_floor_contact_response`). 마찰이 실제 접촉점 K개에만
합산되므로 물리적으로 올바른 총 Coulomb 마찰이 유지된다. topk 선택은 조각별 미분가능.

**콜라캔 3방식 비교(250 iter 수렴, per-frame Euclidean total RMSE):**

| 방식 | 접촉점 | x | y | z(바운스) | total | 최종 y (GT -0.905) |
|---|---|---|---|---|---|---|
| floor_disk | 161 | 0.044 | **0.065** | 0.027 ✗ | **0.082** | -0.849 |
| body_surface | 1080 | 0.056 | 0.090 | 0.018 | 0.108 ✗ | -0.782 (최악) |
| **body_lowest_k K=32** | 32 | **0.038** | 0.081 | **0.016** ✓ | 0.091 | -0.793 |

**K 스윕(4~128)**: 점 수↑ → 마찰↑ → 최종 y가 GT에서 **단조 이탈**(K=8 -0.830 → K=128 -0.782,
body_surface에 수렴). total·z는 K=32에서 최소. 즉 **접촉 패치 크기가 총 마찰을 직접 제어**함을
정량 확인. → 마찰 과함의 원인이 "가우시안별 중복 접촉점"임이 증명됨.

**(잠정 채택이었음: `body_lowest_k K=32`.** base 에피소드에서 body_surface를 전 축에서 지배하고
바운스(z)도 최고였음. **→ 아래 §8의 검증에서 이 주장은 기각됨. 반드시 §8을 함께 읽을 것.)**

**재현**: `python stage2/compare_query_modes.py --episode_root <EP> --stage1_ply <PLY>
--output_root <OUT> --fit_iters 250 --max_frames 80 --body_lowest_k 32`
(3방식 fit→렌더→4패널 GIF `query_mode_comparison.gif` + 축별 궤적 `trajectory_axes.png`
+ K스윕 `k_sweep_over_friction.png` + RMSE 표 `query_mode_comparison_metrics.json`).

## 8. 쿼리포인트 샘플링 방식 실험 + 검증 → **§6의 채택 주장 기각** (2026-07-29)

`--body_query_scheme {axis6, fibonacci, analytic}` 추가(모두 물체측 쿼리, floor_disk는 비교에서 제외).
- `axis6`: 로컬 축 6방향(기존). `fibonacci`: `--body_query_dirs` 개수의 Fibonacci 격자.
- `analytic`: 구의 **월드 프레임 진짜 최저점** `c_world − r·n`. 회전과 무관하게 정확하고 점 1개/프리미티브.
  (axis6/fibonacci는 로컬 샘플링이라 물체가 기울면 최저점을 최대 `r(1−cos θ)`만큼 놓침.)

**단일 실행 결과(초기값 e.55/μ.10)는 오해를 부름**: fib48 0.078 < fib26 0.086 < axis6 0.091 <
fib96 0.089 < analytic 0.109. 접촉 패치 집중도(접촉 프레임 평균 RMS 반경)와 순위가 맞아떨어져
"패치가 좁을수록 좋다"는 그럴듯한 설명이 가능했음(analytic 0.049m/32프리미티브 → fib48 0.0002m/1개).

**그러나 초기값 3종으로 재현성을 검증하니 기각됨:**

| 방식 | e.55/μ.10 | e.50/μ.12 | e.60/μ.08 | 평균 | 편차 |
|---|---|---|---|---|---|
| fib48 | 0.078 | 0.099 | 0.086 | 0.088 | **0.022** |
| fib26 | 0.086 | 0.091 | 0.096 | 0.091 | 0.010 |
| axis6 | 0.091 | 0.098 | 0.090 | 0.093 | 0.008 |
| fib96 | 0.089 | 0.098 | 0.094 | 0.094 | 0.010 |
| analytic | 0.109 | 0.121 | 0.110 | 0.113 | 0.011 |

**초기값에 따른 편차(≤0.022)가 방식 간 평균 차이(0.088~0.094, 폭 0.006)보다 크다** →
axis6/fib26/fib48/fib96은 통계적으로 구별 불가. (analytic만 base에서 일관되게 나쁨: 0.113.)

**홀드아웃 전이 검증** (seed 11, base와 동일 조건·다른 실현. base에서 학습한 e·μ를 동결 전이,
재학습 없음. `--fit_iters 1 --lr 0.0`으로 옵티마이저 스텝 무효화 — `fit_iters 0`은 상류에서
`max(1,·)`로 클램프되고 `best_state`가 step 이후에 저장되므로 파라미터가 바뀜):

| 방식 | BASE total | HOLDOUT total | hold z | hold finXY |
|---|---|---|---|---|
| analytic | 0.113 (최악) | **0.332 (최선)** | 0.093 | 0.533 |
| fib26 | 0.091 | 0.333 | 0.092 | 0.538 |
| fib96 | 0.094 | 0.335 | 0.076 | 0.546 |
| axis6 | 0.093 | 0.339 | 0.089 | 0.548 |
| fib48 | 0.088 (최선) | **0.349 (최악)** | 0.074 | 0.572 |
| body_surface(패치 무제한) | 0.108 | 0.335 | 0.079 | 0.545 |
| body_lowest_k(K=32) | 0.091 | 0.339 | 0.089 | 0.548 |

**순위가 완전히 뒤집힘.** base 최악(analytic)이 홀드아웃 최선, base 최선(fib48)이 홀드아웃 최악.
§6에서 주장한 "body_lowest_k > body_surface"도 홀드아웃에서 역전됨(0.339 vs 0.335).

**결론**:
1. **쿼리포인트 주는 방식은 정확도의 지렛대가 아니다.** base 에피소드 순위는 옵티마이저 노이즈와
   해당 에피소드 과적합의 산물이며 일반화되지 않는다. §6의 채택 근거(단일 에피소드·단일 초기값)는
   불충분했다.
2. 홀드아웃 오차는 모든 방식이 0.332~0.349로 뭉치고 **최종 XY 오차 ~0.55가 지배**한다. 즉 지배적
   오차원은 접촉 쿼리가 아니라 **restitution+GT-ω 모델이 스핀→평면운동 변환을 못 맞추는 것**(§5 한계).
   이 큰 오차가 쿼리 방식 차이를 덮어버려 홀드아웃은 방식 판별력 자체가 낮다는 점도 감안해야 한다.
3. 실질적 개선을 원하면 쿼리 샘플링이 아니라 **자립 강체 회전 예측**(`--dynamics pairwise_impedance`)
   또는 마찰 모델 자체를 손봐야 한다.
4. 방식 비교를 다시 한다면: 초기값 다중 시드 + 홀드아웃 에피소드 다수 + 방식별 K 재최적화가 필요
   (지금 비교는 K=32 고정인데 후보 풀이 1080~17280로 8~16배 달라 통제되지 않았음).

## 9. 판별 가능한 시나리오 설계 → 쿼리 방식 우열 확인 (2026-07-31)

§8에서 방식들이 구별 불가했던 **진짜 원인**을 측정으로 규명함:
- `--radius_scale 0.1` 때문에 충돌 구 반지름이 **0.37mm**(캔 반지름 33mm의 1/90). 구가 사실상
  점이라 표면 샘플링이 중심 샘플링과 같아짐 → 방식별 **접촉점 위치 차이가 0.03mm 이하**.
  틸트 90°(옆으로 누운 자세)에서도 동일. radius_scale을 10배로 키워도 0.37mm에 그침.
- 즉 쿼리 방식이 바꾸는 건 접촉점 *위치*가 아니라 **접촉 패치 크기** → 이는 바운스가 아니라
  **마찰**에 작용. 따라서 공중 바운스 시나리오는 원리적으로 판별력이 없음.

**판별 가능한 시나리오 설계**: 지속적 마찰 접촉이 필요 →
43° rim 착지 후 회전하며 좌→우 1.45m 슬라이드(126프레임 지속 접촉).
`--drop_height 1.1 --max_tilt_deg 55 --spin_speed 6.0 --planar_speed 1.1 --seed 21` ep011.
이 조건에서 패치 크기가 fib26 19mm / axis6 44mm / analytic 74mm로 **4배** 벌어짐
(바운스 시나리오에서는 사실상 동일했음).

**결과 (초기값 3종 검증, per-frame Euclidean total RMSE):**

| 방식 | 쿼리 부여법 | e.55/μ.10 | e.50/μ.12 | e.60/μ.08 | 평균 | 편차 |
|---|---|---|---|---|---|---|
| **analytic** | 구의 월드 최저점 `c−r·n`, 1점/프리미티브 | 0.051 | 0.047 | 0.062 | **0.054** | 0.015 |
| axis6 | 로컬 축 6방향 표면점 | 0.078 | 0.097 | 0.090 | 0.088 | 0.019 |
| fib26 | Fibonacci 26방향 표면점 | 0.137 | 0.143 | 0.116 | 0.132 | 0.027 |

**1-2위 평균 차이 0.034 > 최대 초기값 편차 0.027 → 유의미.** (§8과 달리 노이즈에 안 묻힘.)
궤적도 방식별로 6~15cm 벌어져 육안 식별 가능.

**해석**: 지속 슬라이딩에서는 캔의 rim이 실제로 넓게 닿으므로, 32개 서로 다른 프리미티브에
점을 분산시키는 `analytic`(패치 74mm)이 실제 접촉면을 가장 잘 표현. 반대로 `fib26`은 점이
프리미티브 2개에 뭉쳐(19mm) 접촉면을 과소평가 → 마찰이 부족해 GT보다 뒤처짐.
**바운스 시나리오(§8)와 순위가 반대**인 점에 주의: 그때는 패치가 좁을수록 좋아 보였음.
즉 최적 패치 크기는 접촉 형태에 의존하며, 단일 시나리오로 일반화하면 안 됨.

**산출물**: `output/actual_cola_can_slide_output/`
- `mujoco_slide_reference.gif` — MuJoCo GT(고정 카메라)
- `query_scheme_slide_comparison.gif` — 4패널 [GT | axis6 | fib26 | analytic]
- `fit_{axis6,fib26,analytic}/` — 각 fit 결과

## 7. 결과물 폴더 정리 (2026-07-29)

MuJoCo 데이터셋·Stage 2 결과물을 저장소 루트 `output/` 아래로 모으고 `.gitignore`에 추가.
(top-level `stage1/`·`stage2/` 래퍼 폴더도 시도했으나 실익이 없어 되돌림 — 진입점은
` tools/` 아래 기존 위치를 그대로 쓴다.)
