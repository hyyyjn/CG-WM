# Stage 2 — Phys-Geo Refinement

ContactGaussian-WM 논문 ([arXiv:2602.11021](https://arxiv.org/abs/2602.11021)) Section III-D에 해당.
Stage 1에서 얻은 spherical Gaussian 표현 `G*`와 MuJoCo trajectory를 받아서, **미분 가능한
collision detection + impedance contact dynamics**를 통해 물리 파라미터 (gravity, K, D, …)
및 초기 상태를 backprop으로 학습한다.

이번주 1차 목표: **구체-바닥 충돌 시나리오만 동작 확인**.

---

## 디렉토리 구조

```
gaussian_initiailization/stage2/
├── README.md                                       ← 본 문서
├── __init__.py                                     ← 빈 (패키지 표시)
├── differentiable_collision_detection.py           ← LSE + sigmoid penalty SDF (paper III-D-1)
├── differentiable_complementarity_free_contact_dynamics.py  ← restitution & impedance dynamics
├── gaussian_splatting_rendering.py                 ← SIBR viewer 시퀀스 export (렌더링 학습 X)
├── _smoke_test_sphere_floor.py                     ← 합성 sphere 낙하 단위 테스트
├── _smoke_test_fit_loop.py                         ← 합성 데이터에서 fit script 동작 확인
├── _smoke_test_sphere_end_to_end.py                ← MuJoCo + stage1합성 PLY + fit + 평가 풀 파이프
├── _smoke_test_render_3d_compare.py                ← GT vs 예측 3D 오버레이 렌더
└── _outputs/                                       ← 산출물 (gitignore 권장)
    ├── e2e/                                        ← MuJoCo dataset, fit 결과, 3D GIF
    └── smoke/                                      ← 합성 smoke test 결과
```

연관 파일 (stage2 폴더 밖):
- `gaussian_initiailization/tools/run_stage2_mujoco_stage1_fit.py` — fit script 본체
- `gaussian_initiailization/__init__.py` — 패키지 임포트용 빈 파일

---

## GitHub main 대비 변경 사항

이번 stage2 작업으로 추가/수정된 파일들.

| 파일 | 상태 |
|---|---|
| `stage2/differentiable_collision_detection.py` | 신규 → LSE + sigmoid penalty + analytic normal로 paper III-D-1에 정렬 |
| `stage2/differentiable_complementarity_free_contact_dynamics.py` | 신규 → `ImpedanceFloorContactDynamics` (paper III-D-2 식) 추가 |
| `stage2/gaussian_splatting_rendering.py` | 신규 (SIBR PLY export 헬퍼) |
| `stage2/__init__.py` | 신규 (빈 패키지 마커) |
| `stage2/_smoke_test_*.py` (4개) | 신규 (검증/시각화 스크립트) |
| `stage2/_outputs/` | 신규 (생성 산출물) |
| `tools/run_stage2_mujoco_stage1_fit.py` | 신규 (trajectory L2 기반 fit) |
| `gaussian_initiailization/__init__.py` | 신규 (빈 패키지 마커) |

---

## Paper와의 정렬 상태

| 컴포넌트 | Paper | 현재 구현 |
|---|---|---|
| Gaussian collision SDF | ✓ LSE + sigmoid penalty + ∇φ normal | ✓ 동일 식 (`detect_gaussian_union_contacts`) |
| Closed-form contact dynamics | ✓ `λ = SoftPlus(-K(hJ̃b+ϕ̃) - D(J̃b))` | ✓ `ImpedanceFloorContactDynamics` (1-pair, frictionless) |
| Forward kinematics on Gaussians | ✓ `G*_TF = FK(G*, q̂)` (자세 회전 포함) | ✗ 위치만 평행 이동 (sphere이라 OK) |
| Quaternion + angular velocity | ✓ | ✗ — sphere는 회전 불변이라 보류 |
| Friction Jacobian `J̃(μ)` | ✓ | ✗ — push-slide 시나리오 갈 때 추가 |
| Differentiable 3DGS rasterizer | ✓ | ✗ — `gaussian_splatting_rendering.py`는 viewer export만 |
| Image loss `L = L_Loft + L_L1` | ✓ | ✗ — 현재는 trajectory L2 사용 |
| 학습 파라미터 | `θ = (M, μ, K, D)` + `G_geo` | (v0, g_z, K, D) — log-space 학습 |

---

## 실행 방법 

각 스크립트는 독립적으로 실행 가능. 모든 산출물은 `stage2/_outputs/` 아래에 떨어진다.

### 1. 합성 sphere 낙하 단위 테스트
```bash
conda run -n gs python gaussian_initiailization/stage2/_smoke_test_sphere_floor.py
```
- SDF가 paper의 hard-min과 비교해서 얼마나 가까운지 확인
- Plane / Gaussian-union / Impedance 3가지 dynamics path 각각 굴려서 bounce 검증
- 산출물: `_outputs/smoke/sphere_*.png`, `smoke_summary.json`

### 2. fit 루프 합성 회로 동작 확인
```bash
conda run -n gs python gaussian_initiailization/stage2/_smoke_test_fit_loop.py
```
- 합성 bounce trajectory + 합성 stage1 PLY로 `run_stage2_mujoco_stage1_fit.py` 직접 호출
- impedance 모드와 restitution 모드 둘 다 돌려서 비교

### 3. **end-to-end (메인 데모)**
```bash
conda run -n gs python gaussian_initiailization/stage2/_smoke_test_sphere_end_to_end.py
```
파이프라인:
1. `tools/generate_mujoco_fall_dataset.py` 호출 → 실제 MuJoCo trajectory + RGB 영상 (train 1 + test 3)
2. Stage 1을 우회한 **합성 sphere PLY** 생성 (단일 spherical Gaussian, r=0.10)
3. `tools/run_stage2_mujoco_stage1_fit.py` 호출 → impedance dynamics로 (v0, g, K, D) 학습
4. test 3개 episode에서 학습된 파라미터로 open-loop rollout → MuJoCo GT와 frame-by-frame 비교
5. 산출물: `_outputs/e2e/eval_summary.json`, `test_episode_000_compare.png`, fit 디렉토리

### 4. 3D 오버레이 비교 렌더
```bash
conda run -n gs python gaussian_initiailization/stage2/_smoke_test_render_3d_compare.py
```
- 3번 실행 후, MuJoCo 렌더러로 GT(빨강) + 예측(파랑) 두 공을 한 장면에 띄운 비교 GIF + montage 생성
- 산출물: `_outputs/e2e/test_episode_000_3d_overlay.gif`, `*_montage.png`

---

## 현재 결과 (sphere 낙하 end-to-end)

| 지표 | 값 |
|---|---|
| Fit final position loss | 8.39e-05 (43× 감소) |
| Learned gravity_z | −9.60 m/s² (GT −9.81) |
| Learned K, D | 1418, 47 |
| Test mean translation error | **31.7 mm** (3 episode 평균) |
| Test settled_z 오차 | 17 mm (GT r=0.10에 정착 / pred 0.082에 정착) |
| First-contact frame 오차 | 1 frame (16 ms) |

**참고**: paper Table I Camera 객체 fall-and-rebound가 11.6 mm. 우리 31.7 mm — 동일 자릿수.

---

## 알려진 부족분

1. **xy drift** — friction 미구현이라 sphere의 수평 velocity가 줄지 않고 누적. 3초 시점에 ±26 mm 오차.
2. **settled z 17 mm 차이** — impedance K=1418이 explicit Euler 안정성 한계 (`dt·√(K/m) < 1`). semi-implicit 또는 implicit integrator 도입 시 K 더 크게 → 0에 가깝게 줄일 수 있음.
3. **Stage 1 우회 중** — multi-view 렌더 + `train.py` 학습 미실행. 실제 stage1 PLY는 multi-primitive 대칭 깨짐 → contact normal 미세 편향 → fit 불안정 가능성.
4. **Image loss 미구현** — paper의 `L_Loft + L_L1` 대신 trajectory L2 사용. real-world video 적용 못 함.
5. **Quaternion 미구현** — sphere는 영향 없지만 다음 객체 (box, camera, duck) 가면 필수.

---

## 다음 단계 후보 (우선순위 순)

1. **(stage1 실제 실행)** multi-view 렌더 + `train.py`로 진짜 sphere PLY 만들고 e2e 재검증
2. **multi-primitive contact normal 안정화** — 위 (3) 대비. 1번 결과 보고 결정
3. **friction Jacobian `J̃(μ)`** — push-slide-settle 시나리오 진입 준비
4. **Quaternion / angular velocity** — RigidState 확장, FK 구현 → box 같은 비대칭 객체 지원
5. **Differentiable 3DGS rasterizer + image loss** — real-world video로 확장

---

## 산출물 git ignore

`stage2/_outputs/` 는 binary (PNG, GIF, PLY, MuJoCo rgb 프레임 등) 가 다량 생성되므로
repo 루트의 `.gitignore`에 다음 한 줄 추가 권장:

```
gaussian_initiailization/stage2/_outputs/
```
