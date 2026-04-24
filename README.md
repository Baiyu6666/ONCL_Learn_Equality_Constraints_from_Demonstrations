python analyze/render_sine_pose_demonstrations.py --demo-indices 1,2,3,4,5,6 --draw-ref-trace 0 --draw-ee-trace 1 --trace-surface-clearance 0.018 --trace-stride 1 --demo-smooth-passes 1 --play-fps 45 --video-slowdown 1.0
python analyze/sim_sine_pose_obsavoid.py --seed 12266 --gui 0 --plot-n-paths 3  --ckpt outputs/bench/paper_scale_150_3env_7seed/oncl/6d_workspace_sine_surface_pose_traj_oncl_model.pt --keep-traces 1
python analyze/sim_sine_pose_waypoints.py --gui 0  --seed 61526  --ckpt outputs/bench/paper_scale_150_3env_7seed/oncl/6d_workspace_sine_surface_pose_traj_oncl_model.pt

python analyze/plan_dual_arm_from_learned_constraint.py --ckpt outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/12d_dual_arm_traj_oncl_model.pt --outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl --seed 112 --n-trajs 7 --planner-mode traj_opt --traj-opt-objective minimal_refinit_zlinear --init-mode nearest_demo --plot-arms 0 --opt-steps 3000 --opt-lr 0.001 --lam-manifold 1.0 --lam-ref-path 0.02 --lam-center-z-ref 0.01
python analyze/sim_dual_arm_planned_path.py --plan-outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_dual_arms/ --gui 0 --save-snapshots 1 --snapshot-steps 100,500,750,1000,1250,1600 --planned-path-key path_6
python analyze/sim_dual_arm_planned_path.py --path-source demo --demo-count 5 --render-outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_dual_arms --gui 1


python runners/run_benchmark.py \
  --methods oncl \
  --datasets 12d_dual_arm_traj \
  --seeds 0,1,2,3,4,5,6 \
  --outdir outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed\
  --rewrite

python analyze/sim_dual_arm_planned_path.py --path-source demo --demo-indices 0,2,5 --demo-boundary-hold-steps 4 --render-outdir outputs/bench/dual_arm_demo_path_0_2_5 --gui 2 --orientation-mode task


# LearnEqConstraints

Learning Equality Constraints from Demonstrations.

This repository contains:
- training and evaluation code for `oncl`, `igr`, `dataaug`, `vae`, and `ecomann`
- dataset generation and dataset-specific configs
- benchmark runners and analysis scripts
- saved benchmark outputs under [`outputs/`](/home/baiyu/PycharmProjects/equality_manofld/outputs)

## Repository Layout

- `methods/`: method implementations
- `experiments/`: experiment orchestration and unified run logic
- `evaluation/`: evaluation metrics and evaluation pipeline
- `plotting/`: shared and method-specific plotting code
- `datasets/`: dataset definitions and dataset notes
- `configs/`: base, method, dataset, and method-dataset configs
- `runners/`: CLI entrypoints for single runs and benchmarks
- `analyze/`: analysis and paper figure scripts
- `outputs/`: generated experiment outputs and benchmark summaries

## Environment

Python `3.10` is the recommended baseline.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `wandb` is optional. The runners work without it.
- The repository includes a third-party baseline package under `smp_manifold_learning-master/`, which is the original code implementation of ECoMaNN methods.

## Quick Start

Run one experiment:

```bash
python -m runners.run_one --method oncl --dataset 3d_torus_surface_traj
```

Run one benchmark:

```bash
python -m runners.run_benchmark --methods oncl,dataaug,vae,ecomann --datasets 3d_torus_surface_traj,3d_spatial_arm_ellip_n3_traj
```

Generate paper boxplots from an existing benchmark:

```bash
python -m analyze.benchmark_paper_boxplots --bench paper_mix_2d_3d6d_traj_vs_nontraj_7seed
```

## Configs

The config system is layered:
- `configs/base.json`
- `configs/methods/<method>.json`
- `configs/datasets/<dataset>.json`
- `configs/method_dataset/<method>_<dataset>.json`

Overrides can be passed from the CLI with repeated `--override key=value`.

## Outputs

By default, experiment outputs are written under `outputs/`.

Common locations:
- `outputs/runs/`: ad hoc single runs
- `outputs/bench/`: benchmark runs and aggregated metrics
- `outputs/analysis/`: generated analysis figures and tables

## Metrics

Metric definitions and the mapping between paper-facing names and legacy/internal keys are documented in [docs/metrics.md](/home/baiyu/PycharmProjects/LearnEqConstraints/docs/metrics.md).

## Dataset Naming

Dataset ids in code are the internal ids used by configs and outputs. The mapping to paper environment names is documented in [datasets/README.md](/home/baiyu/PycharmProjects/equality_manofld/datasets/README.md).

For the paper-facing environments, the default benchmark setting uses the trajectory-enabled versions when both `traj` and non-`traj` variants exist.


