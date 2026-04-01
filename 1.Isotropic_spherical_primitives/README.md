GaussianModel 수정 내용

이번 Stage I 구현에서는 scene/gaussian\_model.py를 중심으로 수정했다.

원본 3DGS는 초기화 시점에는 isotropic scale과 identity rotation으로 시작하지만, 내부적으로는 scale을 3축 (sx, sy, sz) 형태로 들고 있어서 이후 학습 과정에서 축별로 다른 값으로 업데이트될 수 있다. 이를 논문의 SG-GS 설정에 맞추기 위해 내부 scale 표현을 단일 반지름 r 하나로 바꾸고, get\_scaling()에서만 \[r, r, r]로 확장되도록 수정했다. 이렇게 하면 Gaussian이 구조적으로 isotropic sphere 형태를 유지할 수 있다.



또한 rotation은 원본 코드에서도 초기값은 identity quaternion으로 시작하지만, 학습 과정에서는 optimizer를 통해 업데이트된다. Stage I에서는 논문에 맞게 rotation update를 막아야 하므로, training\_setup()에서 rotation learning rate를 0.0으로 고정했다. 이로써 초기 rotation은 유지되지만 이후에는 더 이상 학습되지 않는다.



추가로 PLY 저장/로드 부분도 현재 표현에 맞게 수정했다. 저장할 때는 기존 3DGS 포맷과의 호환을 위해 여전히 scale\_0, scale\_1, scale\_2 형태로 저장하되, 내부적으로는 하나의 반지름에서 복제된 값이 저장되도록 했다. 반대로 불러올 때는 저장된 3축 scale 값을 다시 단일 반지름으로 복원하도록 처리했다. Densification 과정에서는 densify\_and\_clone()은 별도 수정 없이 그대로 사용할 수 있었고, densify\_and\_split()만 새로운 단일 반지름 표현에 맞게 scale 계산 부분을 수정했다.



train.py를 수정하지 않은 이유

train.py는 학습 루프를 실행할 뿐이고, Gaussian의 scale 표현이나 rotation optimizer 설정은 모두 scene/gaussian\_model.py 내부에서 처리된다. 따라서 이번 Stage I 수정은 모델 정의만 바꾸면 충분했기 때문에 train.py는 건드리지 않았다

