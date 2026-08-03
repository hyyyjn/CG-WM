# Native Stage2 image-only integration test

## Test contract

- Scene: one dynamic can Gaussian and one static floor Gaussian
- Collision: can `gaussian_union`, floor analytic `plane`
- Supervision: RGB and foreground mask only; no GT trajectory in training
- Frames: `0, 10, 20, 30, 40, 50, 60, 70`
- Optimization: 30 iterations, Adam learning rate `0.002`, `l1_ssim`
- Rendering: both can and floor are rendered by the Gaussian rasterizer

Command:

```bash
conda run -n gaussian_splatting python tools/run_native_multibody_manifest.py \
  --manifest configs/scene_manifest.native_multibody_smoke.json \
  --output output/native_multibody_testcase_02_contact/fit_result.json \
  --fit_iters 30 --lr 0.002 --device cuda \
  --render_stride 10 --render_max_frames 8 \
  --render_width 160 --render_height 120 --image_loss l1_ssim \
  --render_output_dir output/native_multibody_testcase_02_contact/render
```

## Result

| Metric | Value |
|---|---:|
| Initial optimization loss | 0.00384910 |
| Final optimization loss | 0.00383158 |
| Best optimization loss | 0.00382938 |
| Loss reduction | 0.46% |
| Foreground L1 | 0.305897 |
| Foreground PSNR | 8.3647 dB |
| Full-frame PSNR | 6.1805 dB |

The software path passed: manifest loading, hybrid contact rollout, differentiable
multi-body Gaussian rendering, backward, optimizer update, PNG/GIF export, and
GT-free result serialization all completed. The physical fit did **not** pass.

The learned contact parameters stayed at their initial values and the predicted
can height after contact diverged:

```text
frame  0: z =  1.1064
frame 10: z =  0.3958
frame 20: z =  6.9244
frame 30: z = 14.4452
frame 40: z = 20.8761
frame 70: z = 33.6286
```

The current foreground-mask composition removes useful corrective gradients when
the predicted object leaves the target mask or image. The saturated contact
response can therefore launch the object out of view while the optimizer mostly
adjusts early-frame pose. Before adding unrelated features, the next required fix
is an asymmetric silhouette/occupancy loss (penalizing predicted foreground
outside the GT mask), contact-force regularization, and a staged contact-parameter
initialization/curriculum.

Artifacts are under `output/native_multibody_testcase_02_contact/`:

- `fit_result.json`
- `render/prediction.gif`
- `render/gt_left_prediction_right.gif`
- per-frame prediction and comparison PNG directories
