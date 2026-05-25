# ContactGaussian-WM Demo Pipeline
# Run from repo root: C:\Users\Admin\CG-WM
# Prerequisites: conda env 'gs' with PyTorch+CUDA+MuJoCo installed

param(
    [string]$OutputRoot   = "$PSScriptRoot\demo_data",
    [string]$ModelRoot    = "$PSScriptRoot\demo_output",
    [string]$SceneName    = "sphere_demo",
    [string]$ObjectType   = "sphere",     # box | sphere | cylinder
    [string]$CondaEnv     = "gs",
    [int]$Stage1Iters     = 10000,
    [int]$Stage2FitIters  = 500,
    [double]$ForegroundThreshold = 0.50,  # sphere: slightly lower than box (0.55)
    [switch]$SkipStage1   = $false,       # -SkipStage1 to reuse existing model
    [switch]$DryRun       = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = "$PSScriptRoot\gaussian_initiailization"
$Conda = "conda"

# Derived paths
$DatasetDir     = "$OutputRoot\$SceneName"
$ModelDir       = "$ModelRoot\${SceneName}_stage1"
$AssetJson      = "$OutputRoot\${SceneName}_asset.json"
$Stage2Root     = "$OutputRoot"
$FitOutputDir   = "$ModelRoot\${SceneName}_stage2_fit"

function Run([string[]]$Cmd) {
    $pretty = ($Cmd | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "
    Write-Host "`n[RUN] $pretty" -ForegroundColor Cyan
    if (-not $DryRun) {
        & $Cmd[0] $Cmd[1..($Cmd.Length - 1)]
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
    }
}

function Py([string[]]$PythonArgs) {
    Run (@($Conda, "run", "-n", $CondaEnv, "python") + $PythonArgs)
}

Write-Host "=== ContactGaussian-WM Demo ===" -ForegroundColor Green
Write-Host "Dataset : $DatasetDir"
Write-Host "Model   : $ModelDir"
Write-Host "Stage2  : $FitOutputDir"
Write-Host "Conda   : $CondaEnv"
Write-Host ""

# ─────────────────────────────────────────────
# STEP 1  Generate static multi-view dataset
# ─────────────────────────────────────────────
Write-Host "[STEP 1] Generating MuJoCo synthetic dataset..." -ForegroundColor Yellow

Py @(
    "$ScriptDir\generate_mujoco_synthetic_dataset.py",
    "--output_root",  $OutputRoot,
    "--scene_name",   $SceneName,
    "--object_type",  $ObjectType,
    "--train_views",  "32",
    "--test_views",   "8",
    "--width",        "512",
    "--height",       "512",
    "--mujoco_gl",    "glfw"
)

# ─────────────────────────────────────────────
# STEP 2  Stage 1 training
# ─────────────────────────────────────────────
if (-not $SkipStage1) {
    Write-Host "`n[STEP 2] Running Stage 1 Gaussian training ($Stage1Iters iters)..." -ForegroundColor Yellow
    Write-Host "         This takes ~15-25 min on GPU. Watch the PSNR climb in the log."

    Py @(
        "$ScriptDir\train.py",
        "--source_path",     $DatasetDir,
        "--model_path",      $ModelDir,
        "--iterations",      "$Stage1Iters",
        "--save_iterations", "$Stage1Iters",
        "--test_iterations", "$Stage1Iters",
        "--masks_dir",       "$DatasetDir\masks",
        "--object_mask_weight", "1.0",
        "--object_mask_bce_weight", "2.0",
        "--sam_feature_weight", "0.0",
        "--disable_viewer",
        "--quiet",
        "--eval"
    )
} else {
    Write-Host "`n[STEP 2] Skipping Stage 1 (--SkipStage1 flag set)." -ForegroundColor DarkYellow
}

# ─────────────────────────────────────────────
# STEP 3  Write object_asset.json
# ─────────────────────────────────────────────
Write-Host "`n[STEP 3] Writing object asset JSON..." -ForegroundColor Yellow

# Resolve PLY path: prefer the exact training iteration, fall back to the
# highest available checkpoint (needed when -SkipStage1 reuses a model trained
# at a different iteration count).
$PlyPath = "$ModelDir\point_cloud\iteration_${Stage1Iters}\point_cloud.ply"
if (-not (Test-Path $PlyPath)) {
    $found = Get-ChildItem "$ModelDir\point_cloud\iteration_*\point_cloud.ply" -ErrorAction SilentlyContinue |
             Sort-Object { [int]($_.Directory.Name -replace 'iteration_','') } |
             Select-Object -Last 1
    if ($found) { $PlyPath = $found.FullName }
}
if (-not (Test-Path $PlyPath)) {
    throw "Stage 1 PLY not found. Expected $PlyPath. Run without -SkipStage1 or lower -Stage1Iters to an existing checkpoint."
}

# Object-specific geometry: half-extent in metres (matches MuJoCo scene.xml)
$geomParams = switch ($ObjectType) {
    "box"      { @{ bmin="-0.08, -0.08, -0.08"; bmax="0.08, 0.08, 0.08"; cz=0.08 } }
    "sphere"   { @{ bmin="-0.09, -0.09, -0.09"; bmax="0.09, 0.09, 0.09"; cz=0.09 } }
    "cylinder" { @{ bmin="-0.07, -0.07, -0.10"; bmax="0.07, 0.07, 0.10"; cz=0.10 } }
    default    { @{ bmin="-0.08, -0.08, -0.08"; bmax="0.08, 0.08, 0.08"; cz=0.08 } }
}

$DatasetFwd = $DatasetDir -replace "\\", "/"
$PlyFwd     = $PlyPath    -replace "\\", "/"
$cz         = $geomParams.cz

# Write JSON directly to avoid PowerShell 5.1 nested-array serialization quirks
$assetContent = @"
{
  "object_name": "$SceneName",
  "mesh_path": "",
  "stage1_dataset_path": "$DatasetFwd",
  "stage1_points_ply": "$PlyFwd",
  "physics_shape": "$ObjectType",
  "normalization": {
    "bbox_min": [$($geomParams.bmin)],
    "bbox_max": [$($geomParams.bmax)],
    "scale": 1.0
  },
  "stage1_gaussian_body": {
    "coordinate_frame": "world",
    "world_pose": {
      "translation": [0.0, 0.0, $cz],
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
  }
}
"@

New-Item -ItemType Directory -Force (Split-Path $AssetJson) | Out-Null
$assetContent | Out-File -Encoding UTF8 $AssetJson
Write-Host "  Wrote: $AssetJson"

Write-Host "`n[STEP 3b] Auditing Stage 1 Gaussian object mask/scale (warn-only)..." -ForegroundColor Yellow

# Audit is non-fatal: low-iter quick runs will fail the ratio check but Stage 2 can still proceed.
try {
    Py @(
        "$ScriptDir\tools\audit_stage1_ply.py",
        "--stage1_ply",          $PlyPath,
        "--object_asset",        $AssetJson,
        "--foreground_threshold","$ForegroundThreshold",
        "--max_extent_ratio",    "3.0"
    )
} catch {
    Write-Host "[WARN] audit_stage1_ply failed (expected for low-iter runs). Continuing." -ForegroundColor DarkYellow
}

# ─────────────────────────────────────────────
# STEP 4  Create Stage 2 dataset layout
# ─────────────────────────────────────────────
Write-Host "`n[STEP 4] Creating Stage 2 dataset layout..." -ForegroundColor Yellow

Py @(
    "$ScriptDir\tools\create_contactwm_stage2_layout.py",
    "--dataset_root",      $Stage2Root,
    "--object_asset",      $AssetJson,
    "--scenario",          "fall_and_rebound",
    "--train_episodes",    "2",
    "--test_episodes",     "4",
    "--frames_per_episode","120",
    "--fps",               "30",
    "--overwrite"
)

# ─────────────────────────────────────────────
# STEP 5  Generate fall trajectories (MuJoCo)
# ─────────────────────────────────────────────
Write-Host "`n[STEP 5] Generating MuJoCo fall trajectories..." -ForegroundColor Yellow

# generate_mujoco_fall_dataset.py does not expose --mujoco_gl, set env directly
$env:MUJOCO_GL = "glfw"

# Object-specific fall parameters: set variables, then call Py once outside if/else
# (PowerShell 5.1 parser issue with multiline Py @(...) inside if blocks)
$FallDropTrain  = "0.75"; $FallDropTest  = "0.95"
$FallXyTrain    = "0.12"; $FallXyTest    = "0.25"
$FallPlanTrain  = "0.8";  $FallPlanTest  = "1.1"
$FallSpinTrain  = "8.0";  $FallSpinTest  = "12.0"
$FallTiltTrain  = "18.0"; $FallTiltTest  = "38.0"

if ($ObjectType -eq "sphere") {
    $FallDropTrain = "0.80"; $FallDropTest = "1.20"
    $FallXyTrain   = "0.02"; $FallXyTest   = "0.03"
    $FallPlanTrain = "0.05"; $FallPlanTest = "0.05"
    $FallSpinTrain = "0.1";  $FallSpinTest = "0.1"
    $FallTiltTrain = "1.0";  $FallTiltTest = "1.0"
}

Py @(
    "$ScriptDir\tools\generate_mujoco_fall_dataset.py",
    "--dataset_root",       $Stage2Root,
    "--object_name",        $SceneName,
    "--split",              "all",
    "--camera_distance",    "1.35",
    "--camera_height",      "0.75",
    "--ground_size",        "2.0",
    "--drop_height_train",  $FallDropTrain,
    "--drop_height_test",   $FallDropTest,
    "--xy_range_train",     $FallXyTrain,
    "--xy_range_test",      $FallXyTest,
    "--planar_speed_train", $FallPlanTrain,
    "--planar_speed_test",  $FallPlanTest,
    "--spin_speed_train",   $FallSpinTrain,
    "--spin_speed_test",    $FallSpinTest,
    "--max_tilt_deg_train", $FallTiltTrain,
    "--max_tilt_deg_test",  $FallTiltTest,
    "--sphere_solref",      "0.02 0.2"
)

# ─────────────────────────────────────────────
# STEP 6  Stage 2 dynamics fitting
# ─────────────────────────────────────────────
Write-Host "`n[STEP 6] Fitting Stage 2 contact dynamics (restitution)..." -ForegroundColor Yellow

$EpisodeRoot = "$Stage2Root\stage2\fall_and_rebound\test\$SceneName\episode_000"

# Object-specific Stage 2 params
# sphere: radius_scale 크게 (0.1은 box 전용 — sphere Gaussian이 너무 작아져 contact gradient 소멸)
# sphere: tangential damping 불필요 (등방성 접촉)
$RadiusScale       = if ($ObjectType -eq "sphere") { "0.8" } else { "0.1" }
$TangentialDamping = if ($ObjectType -eq "sphere") { "5.0" } else { "80.0" }

Py @(
    "$ScriptDir\tools\run_stage2_mujoco_stage1_fit.py",
    "--episode_root",     $EpisodeRoot,
    "--stage1_model_path",$ModelDir,
    "--output_dir",       $FitOutputDir,
    "--dynamics",         "restitution",
    "--fit_iters",        "$Stage2FitIters",
    "--lr",               "0.015",
    "--device",           "cuda",
    "--gif_fps",          "20",
    "--max_primitives",   "2000",
    "--foreground_threshold", "$ForegroundThreshold",
    "--opacity_threshold", "0.02",
    "--radius_scale",     $RadiusScale,
    "--initial_velocity_source", "trajectory",
    "--freeze_initial_velocity",
    "--floor_tangential_damping", $TangentialDamping
)

# ─────────────────────────────────────────────
# STEP 7  Export MuJoCo GT frames as GIF
# ─────────────────────────────────────────────
Write-Host "`n[STEP 7] Exporting MuJoCo ground truth episode as GIF..." -ForegroundColor Yellow

$GtGif = "$FitOutputDir\gt_episode.gif"

Py @(
    "$ScriptDir\tools\export_episode_gif.py",
    "--rgb_dir",      "$EpisodeRoot\rgb",
    "--output_gif",   $GtGif,
    "--fps",          "20",
    "--resize_width", "480"
)

# ─────────────────────────────────────────────
# STEP 8  Side-by-side 3DGS rendering comparison
# ─────────────────────────────────────────────
Write-Host "`n[STEP 8] Rendering 3DGS side-by-side comparison GIF..." -ForegroundColor Yellow

$PredTraj      = "$FitOutputDir\predicted_trajectory.json"
$ComparisonGif = "$FitOutputDir\comparison_gt_vs_3dgs.gif"

Py @(
    "$ScriptDir\tools\render_trajectory_comparison.py",
    "--episode_root",         $EpisodeRoot,
    "--predicted_trajectory", $PredTraj,
    "--stage1_model_path",    $ModelDir,
    "--output_gif",           $ComparisonGif,
    "--stage1_centroid",      "0,0,$cz",
    "--cam_distance",         "1.35",
    "--cam_height",           "0.75",
    "--cam_fovy_deg",         "45.0",
    "--image_width",          "640",
    "--image_height",         "480",
    "--panel_width",          "480",
    "--fps",                  "20",
    "--foreground_threshold", "$ForegroundThreshold",
    "--device",               "cuda"
)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Results:"
Write-Host "  [GIF 1] GT video     : $GtGif"
Write-Host "  [GIF 2] Traj compare : $FitOutputDir\stage2_fit_follow_view.gif"
Write-Host "  [GIF 3] 3DGS compare : $ComparisonGif  <-- MAIN"
Write-Host "  [JSON]  Fit stats    : $FitOutputDir\fit_summary.json"
