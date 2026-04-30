# LearnEqConstraints

Learning equality constraints from demonstrations.

Authors:
- Baiyu Peng, LASA, EPFL, `baiyu.peng@epfl.ch`
- Albert Árboles, Universitat Autònoma de Barcelona, `albert.arbolesi@autonoma.cat`
- Aude Billard, LASA, EPFL, `aude.billard@epfl.ch`

This work was supported by the euROBIN Grant.

Project summary:

Equality constraints are common in robotics, arising in applications such as pose-constrained manipulation, coordinated motion, and task-space planning. Classical constrained planners typically assume an analytic constraint function, whereas in practice only feasible demonstrations may be available.

We study how to learn explicit equality constraints directly from such demonstrations. We propose Orthonormal Neural Constraint Learning (ONCL), a method that learns multiple scalar constraint functions jointly and represents the equality constraint by their common zero set.

Rather than estimating local normal directions and generating large amounts of supervised off-constraint data as in related work, ONCL regularizes the gradients of different constraint functions to be normalized and mutually orthogonal, inducing a structured local normal space during learning. This yields an explicit differentiable constraint model in the original state space for evaluation and downstream planning.

We evaluate ONCL on synthetic and robotic datasets spanning dimensions from 2 to 12 and compare it with strong baselines. The experiments show competitive or best mean error across the benchmark suite, improved robustness when training from trajectory demonstrations rather than uniformly scattered feasible samples, and successful downstream use of the learned constraints in planning and control, including transfer to new tasks with different objectives but the same constraints.

This repository contains:
- method implementations for `ONCL`, `DataAug`, `VAE`, and `ECoMaNN`
- dataset generation code for 2D, 3D, 6D, and 12D tasks
- unified training and evaluation pipelines
- planning and simulation scripts for supported transfer tasks
- benchmark aggregation and paper-figure utilities

## Overview

Typical workflow:

1. choose a method and dataset
2. train with `runners/run_one.py` or `runners/run_benchmark.py`
3. inspect learned-constraint metrics and plots
4. for supported tasks, run planning and simulation scripts
5. aggregate results into paper figures

## Repository Layout

- `methods/`: learning methods and method-side helpers
  - `methods/models/`: small neural architectures
  - `methods/utils/`: feature normalization and projection helpers
- `datasets/`: dataset generation and task geometry
- `experiments/`: unified experiment orchestration
- `evaluation/`: evaluation metrics and helper code
- `planning/`: learned-constraint planning scripts and planning utilities
- `simulation/`: transfer-task simulation, IK control, and rendering
- `plotting/`: reusable plotting code
- `benchmark/`: benchmark plotting scripts
- `tools/`: auxiliary analysis and preview scripts
- `configs/`: layered config files
- `runners/`: CLI entrypoints for training and benchmarks
- `outputs/`: generated results

Dataset naming, paper benchmark environments, additional project datasets, and trajectory-vs-scatter notes are summarized in:
- [DATASETS.md](DATASETS.md)

## Environment
```bash
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `wandb` is optional.
- The repository includes an integrated copy of the original ECoMaNN codebase with the modifications needed to fit this repository structure.
- ECoMaNN reference: Sutanto, G., Rayas Fernández, I., Englert, P., Ramachandran, R. K., and Sukhatme, G. (2021). "Learning Equality Constraints for Motion Planning on Manifolds." In *Proceedings of the 2020 Conference on Robot Learning*, PMLR 155, 2292-2305.
- We use the original authors' implementation from `https://github.com/gsutanto/smp_manifold_learning`, with the necessary modifications to integrate it into this codebase.

## Configuration

Configs are layered in this order:

1. `configs/base.json`
2. `configs/methods/<method>.json`
3. `configs/datasets/<dataset>.json`
4. `configs/method_dataset/<method>_<dataset>.json`
5. CLI `--override key=value`

Supported method config families include:
- `oncl`
- `dataaug`
- `vae`
- `ecomann`

Common examples:
- `--override n_train=1500`
- `--override lr=0.002`
- `--override epochs=2200`
- `--override eval_tau_ratio=0.02`

Nested override syntax is also supported when the target config uses nested objects, e.g.:

- `--override train.lr=0.0001`


## Training and Evaluation

Run one experiment:

```bash
python -m runners.run_one \
  --method oncl \
  --dataset 3d_torus_surface_traj \
  --outdir outputs/bench/example_one
```

Run a benchmark:

```bash
python -m runners.run_benchmark \
  --methods oncl,dataaug,vae,ecomann \
  --datasets 3d_torus_surface_traj,3d_spatial_arm_ellip_n3_traj \
  --outdir outputs/bench/example_benchmark
```

Generate benchmark plots from an existing benchmark directory:

```bash
python -m benchmark.plot_paper_benchmarks \
  --bench example_benchmark
```

### Important Training Arguments

For `run_one` and `run_benchmark`, the most important arguments are:

- `--method`: learning method, e.g. `oncl`, `dataaug`, `vae`, `ecomann`
- `--dataset`: dataset id, e.g. `6d_workspace_sine_surface_pose_traj`
- `--outdir`: output directory for checkpoints, summaries, and plots
- `--rewrite`: overwrite an existing benchmark directory
- `--override key=value`: override config values without editing config files

## Planning and Simulation

Not every dataset in this repository has planning and simulation scripts. The transfer-task scripts currently focus on:

- `6d_workspace_sine_surface_pose_traj`
  - obstacle-avoidance transfer
  - waypoint-constrained transfer
- `12d_dual_arm_traj`
  - learned-constraint planning
  - planned-path simulation
  - demonstration rendering

For most other datasets, the repository provides training and evaluation only.

### GUI Modes

For scripts that expose `--gui`, the modes are:

- `--gui 0`: run without interactive GUI; no main scene video is rendered, but summaries are still written and auxiliary error videos may still be exported
- `--gui 1`: off-screen rendering with saved video outputs
- `--gui 2`: interactive PyBullet GUI for manual inspection

### 6DSinePose

Main scripts:

- `simulation/sim_sine_pose_obsavoid.py`
- `simulation/sim_sine_pose_waypoints.py`

Example:

```bash
python simulation/sim_sine_pose_obsavoid.py \
  --gui 0 \
  --plot-n-paths 3 \
  --ckpt outputs/bench/<bench>/oncl/6d_workspace_sine_surface_pose_traj_oncl_model.pt
```

Important arguments:
- `--ckpt`: learned model checkpoint
- `--plot-n-paths`: number of transfer trajectories to generate and execute
- `--seed`: transfer-task seed
- `--keep-traces`: whether to keep traces from previous trajectories in the same render
- `--save-error-video`: export the side error video

### 12DDualArm

Workflow:

1. generate planned paths with `planning/plan_dual_arm.py`
2. simulate planned paths or demonstration paths with `simulation/sim_dual_arm_planned_path.py`

Planning example:

```bash
python planning/plan_dual_arm.py \
  --ckpt outputs/bench/<bench>/oncl/12d_dual_arm_traj_oncl_model.pt \
  --outdir outputs/bench/<bench>/oncl_plan
```

Important planning arguments:
- `--ckpt`: learned model checkpoint
- `--outdir`: planning output directory
- `--n-trajs`: number of planned trajectories

Demo rendering example:

```bash
python simulation/sim_dual_arm_planned_path.py \
  --path-source demo \
  --demo-count 5 \
  --render-outdir outputs/bench/<bench>/sim_dual_arms \
  --gui 1
```

Important simulation arguments:
- `--plan-outdir`: directory containing planned paths
- `--planned-path-key`: which saved path to execute
- `--path-source`: `planned` or `demo`
- `--demo-count` / `--demo-indices`: choose rendered demonstration paths
- `--render-outdir`: video and summary output directory
- `--save-snapshots`: whether to save selected still frames

## Naming Conventions

- dataset ids in configs and code use internal names such as `6d_workspace_sine_surface_pose_traj`
- paper-facing scripts often remap these to labels such as `6DSinePose`
- when both scatter and trajectory variants exist, paper benchmarks usually use the `*_traj` versions
- non-`_traj` datasets use scatter samples of the true constraint, while `_traj` datasets use trajectory demonstrations on the same underlying constraint

## Reference Commands

The following commands are kept here as commonly used references.

```bash
python simulation/render_sine_pose_demonstrations.py --demo-indices 1,2,3,4,5,6 --draw-ref-trace 0 --draw-ee-trace 1 --trace-surface-clearance 0.018 --trace-stride 1 --demo-smooth-passes 1 --play-fps 45 --video-slowdown 1.0

python simulation/sim_sine_pose_obsavoid.py --seed 12266 --gui 0 --plot-n-paths 3 --ckpt outputs/bench/paper_scale_150_3env_7seed/oncl/6d_workspace_sine_surface_pose_traj_oncl_model.pt --keep-traces 1

python simulation/sim_sine_pose_waypoints.py --gui 0 --seed 61526 --ckpt outputs/bench/paper_scale_150_3env_7seed/oncl/6d_workspace_sine_surface_pose_traj_oncl_model.pt

python planning/plan_dual_arm.py --ckpt outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/12d_dual_arm_traj_oncl_model.pt --outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl --seed 112 --n-trajs 7

python simulation/sim_dual_arm_planned_path.py --plan-outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_dual_arms/ --gui 0 --save-snapshots 1 --snapshot-steps 100,500,750,1000,1250,1600 --planned-path-key path_6

python simulation/sim_dual_arm_planned_path.py --path-source demo --demo-count 5 --render-outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_dual_arms --gui 1

python simulation/sim_dual_arm_planned_path.py --path-source demo --demo-indices 0,2,5 --demo-boundary-hold-steps 4 --render-outdir outputs/bench/dual_arm_demo_path_0_2_5 --gui 2 --orientation-mode task

python runners/run_benchmark.py --methods oncl --datasets 12d_dual_arm_traj --seeds 0,1,2,3,4,5,6 --outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed --rewrite
```
