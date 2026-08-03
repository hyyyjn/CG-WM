# Contact test v3: Stage2 rerun

## Inputs

- Dataset: `output/can_floor_contact_test_dataset_v3`
- Frames used: `0, 5, ..., 70` (15 frames)
- Camera: episode calibration (`distance=2.2`, `height=1.1`, target z `0.82`, FOV `45°`)
- Training: image-only, 30 iterations, Adam `lr=0.002`, `l1_ssim`
- Rendering: learned can Gaussian + learned/calibrated floor Gaussian
- Collision: can Gaussian-union + analytic plane
- GT trajectory: excluded from training; read only for the evaluation below

## Results

| Metric | Value |
|---|---:|
| Initial loss | 0.00340885 |
| Final loss | 0.00325158 |
| Loss reduction | 4.61% |
| Foreground PSNR | 7.2496 dB |
| Evaluation position RMSE | 0.1717 m |
| Learned stiffness | 30.0053 |
| Learned damping | 5.0050 |
| Learned friction | 0.4969 |

The previous explosive rebound was removed. The result still fails the physical
acceptance criterion: after contact, the predicted can drifts laterally and
slowly penetrates the plane. At frame 70, predicted position is
`[0.0273, -0.1753, -0.0076]`, while evaluation GT is
`[0.0036, 0.0175, 0.0997]`.

Visual comparison also exposes an asset-domain mismatch. The Gaussian floor on
the prediction side has a dark background and a finite gray patch, whereas the
new MuJoCo observation has a white background and checker floor filling the
view. The can orientation/appearance differs around contact as well. The mask
objective limits the influence of the floor mismatch but cannot make the final
rendering visually equivalent.

Artifacts:

- `output/native_contact_test_v3_fit/fit_result.json`
- `output/native_contact_test_v3_fit/render/prediction.gif`
- `output/native_contact_test_v3_fit/render/gt_left_prediction_right.gif`
- per-frame PNGs in the render subdirectories

Verdict: software execution passes; image fit improves; physical trajectory and
Gaussian scene reconstruction do not yet pass.
