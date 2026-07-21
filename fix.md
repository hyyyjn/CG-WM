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
python gaussian_initiailization/tools/generate_mujoco_fall_dataset.py \
  --dataset_root <DATA_ROOT> --object_name cola_can_dynamic --split train \
  --camera_distance 2.5 --camera_height 1.3 \
  --camera_target_x 0.72 --camera_target_y -0.62 --camera_target_z 0.32 \
  --drop_height_train 1.8 --max_tilt_deg_train 6 --spin_speed_train 2.0 \
  --planar_speed_train 0.35 \
  --cylinder_solref "0.01 0.2" --floor_solref "0.01 0.2" --seed 2

# 2) Stage 2 fit (쿼리 모드만 바꿔 2회)
python gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py \
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
