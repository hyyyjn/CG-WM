# SG-GS Stage2 LoFTR Weight Ablation

## Fixed setup

- Manifest: `configs/scene_manifest.sggs_prefit_test.json`
- Initial state: image-only prefit estimate (no GT trajectory supervision)
- Frames: `0,3,6,9,12,15,18,21` at 30 fps
- Resolution: 160×120
- Optimizer iterations: 30
- Physics LR: 0.002
- Geometry refinement LR: 0.001
- LoFTR: Kornia 0.6.12, frozen outdoor weights
- Confidence threshold: 0.05
- Maximum matches: 128
- Minimum matches: 1
- Patch radius: 2

## Results

| LoFTR weight | Final objective | Final pixel L1 term | Final LoFTR loss | Matches | Foreground L1 ↓ | Foreground PSNR ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (L1 only) | 0.002820 | 0.002820 | – | – | **0.373162** | **6.4888** |
| 0.02 | 0.005614 | 0.002868 | 0.137267 | 11 | 0.379736 | 6.4009 |
| 0.05 | 0.009646 | 0.002886 | 0.135199 | 11 | 0.381999 | 6.3447 |
| 0.10 | 0.016227 | 0.002962 | **0.132643** | 11 | 0.392311 | 6.1471 |

All LoFTR runs started at feature loss 0.150336. Higher feature weight reduced the
feature loss more strongly, but worsened the held-out foreground pixel metric in
this short synthetic-can experiment. Weight 0.02 is the least harmful LoFTR
setting, while weight 0 remains the best choice if foreground L1/PSNR is the
selection criterion.

The likely bottleneck is not failure to find correspondences: the final LoFTR
runs retained 11 foreground matches with mean confidence around 0.90. Instead,
the Stage1 model is only a 600-iteration reconstruction and its local rendered
appearance differs from the MuJoCo video. The normalized patch objective can
therefore favor local feature agreement at the expense of exact RGB alignment.

## Artifacts

- `output/sggs_stage2_loftr_ablation/w0/fit.json`
- `output/sggs_stage2_loftr_ablation/w002/fit.json`
- `output/sggs_stage2_loftr_ablation/w005/fit.json`
- `output/sggs_stage2_loftr_ablation/w01/fit.json`
- Each run also contains `render/gt_left_prediction_right.gif`.

## Current recommendation

Use L1-only for the current test pipeline. Keep LoFTR weight 0.02 as an optional
ablation until Stage1 appearance quality and multi-view Stage2 observations are
improved. Do not use 0.1 as the default for this dataset.
