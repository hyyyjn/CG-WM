# Realistic can-drop Stage2 test

The realistic 90-frame MuJoCo episode was fitted with 18 sampled frames
(`0..85`, stride 5), 30 image-only iterations, calibrated camera parameters,
and no GT trajectory in training.

| Metric | Value |
|---|---:|
| Initial loss | 0.00350461 |
| Final loss | 0.00317238 |
| Best loss | 0.00315880 |
| Loss reduction | 9.48% |
| Foreground PSNR | 7.6928 dB |
| Evaluation position RMSE | 0.2159 m |

Learned contact values changed from `(30, 5, 0.6)` to stiffness `30.0190`,
damping `5.0169`, and friction `0.5902`.

The test does not pass physically. MuJoCo GT tips and settles near
`[0.085, -0.091, 0.070]` by frame 30. The prediction continues sliding in the
negative-y direction and penetrates the plane after frame 65; at frame 85 it is
`[-0.077, -0.337, 0.021]`.

The comparison also confirms a scene-asset domain mismatch: the existing floor
Gaussian has a black background and finite gray patch, unlike the realistic
sky and tiled floor observations. The input episode is now suitable, but the
Stage1 floor/can assets and contact dynamics are not yet adequate for it.

Artifacts:

- `output/native_realistic_contact_fit/fit_result.json`
- `output/native_realistic_contact_fit/render/prediction.gif`
- `output/native_realistic_contact_fit/render/gt_left_prediction_right.gif`
