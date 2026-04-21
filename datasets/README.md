# Datasets Overview

This folder contains dataset generators and dataset-related assets used by the project.

Main entry:
- `datasets/constraint_datasets.py`
- function: `generate_dataset(name, cfg)`

## Naming Notes

- Repository dataset names such as `3d_torus_surface_traj` are internal ids used by configs and code.
- The paper uses cleaner environment names such as `3DTorus`.
- When both a base version and a `_traj` version exist, the `_traj` version is the default environment variant used in the paper experiments.
- Non-`_traj` versions use scatter samples of true constraint; `_traj` versions use trajectory demonstrations on the same underlying constraint.
- If a dataset does not appear in the paper, the paper-name column is left blank.

## Dataset to Paper-Environment Mapping

| Dataset ID | Data Dim | Codim | Paper Env Name | Traj Note | Description |
|---|---:|---:|---|---|---|
| `2d_discontinuous` | 2 | 1 |  | no traj variant | Piecewise/discontinuous sine-like curve |
| `2d_ellipse` | 2 | 1 | `2DEllipse` | no traj variant | 2D ellipse curve |
| `2d_figure_eight` | 2 | 1 |  | no traj variant | 2D figure-eight curve |
| `2d_hetero_noise` | 2 | 1 |  | no traj variant | Manifold with non-uniform noise level |
| `2d_looped_spiro` | 2 | 1 |  | no traj variant | Multi-loop spirograph-like curve |
| `2d_noisy_sine` | 2 | 1 |  | no traj variant | Sine curve with stronger noise |
| `2d_planar_arm_line_n2` | 2 | 1 | `2DPlanarArmLine` | no traj variant | 2-DoF planar arm, end-effector constrained on a workspace line |
| `2d_sharp_star` | 2 | 1 |  | no traj variant | Star-like closed curve with sharp corners |
| `2d_sine` | 2 | 1 |  | no traj variant | Clean sine curve |
| `2d_sparse_sine` | 2 | 1 | `2DSineSparse` | no traj variant | Sparse sine samples |
| `2d_square` | 2 | 1 |  | no traj variant | 2D square boundary manifold |
| `3d_0z_2d_discontinuous` | 3 | 2 |  | derived lift, non-traj | Lifted 2D discontinuous dataset with `z=0` |
| `3d_0z_2d_ellipse` | 3 | 2 |  | derived lift, non-traj | Lifted 2D ellipse dataset with `z=0` |
| `3d_0z_2d_figure_eight` | 3 | 2 |  | derived lift, non-traj | Lifted 2D figure-eight dataset with `z=0` |
| `3d_0z_2d_hetero_noise` | 3 | 2 |  | derived lift, non-traj | Lifted 2D hetero-noise dataset with `z=0` |
| `3d_0z_2d_looped_spiro` | 3 | 2 |  | derived lift, non-traj | Lifted 2D looped-spiro dataset with `z=0` |
| `3d_0z_2d_noisy_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D noisy-sine dataset with `z=0` |
| `3d_0z_2d_planar_arm_line_n2` | 3 | 2 |  | derived lift, non-traj | Lifted 2D planar-arm-line dataset with `z=0` |
| `3d_0z_2d_sharp_star` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sharp-star dataset with `z=0` |
| `3d_0z_2d_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sine dataset with `z=0` |
| `3d_0z_2d_sparse_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sparse-sine dataset with `z=0` |
| `3d_0z_2d_square` | 3 | 2 |  | derived lift, non-traj | Lifted 2D square dataset with `z=0` |
| `3d_paraboloid` | 3 | 1 |  | base version | 3D paraboloid surface manifold |
| `3d_paraboloid_traj` | 3 | 1 |  | traj version | 3D paraboloid surface manifold with trajectory-style sampling |
| `3d_planar_arm_line_n3` | 3 | 1 |  | base version; paper uses `_traj` by default | 3-DoF planar arm, end-effector constrained on a workspace line |
| `3d_planar_arm_line_n3_traj` | 3 | 1 | `3DPlanarArmLine` | default paper variant | 3-DoF planar arm line manifold with trajectory-style sampling |
| `3d_saddle_surface` | 3 | 1 |  | base version | Saddle-type surface in 3D |
| `3d_saddle_surface_traj` | 3 | 1 |  | traj version | Saddle-type surface in 3D with trajectory-style sampling |
| `3d_spatial_arm_circle_n3` | 3 | 2 |  | no traj variant | 3-DoF spatial arm, end-effector constrained on a workspace circle |
| `3d_spatial_arm_ellip_n3` | 3 | 2 | | base version; paper uses `_traj` by default | 3-DoF spatial arm, end-effector constrained on a workspace ellipse |
| `3d_spatial_arm_ellip_n3_traj` | 3 | 2 | `3DArmEllipse` | default paper variant | 3-DoF spatial arm ellipse manifold with trajectory-style sampling |
| `3d_spatial_arm_plane_n3` | 3 | 1 |  | base version | 3-DoF spatial arm, end-effector constrained on a workspace plane |
| `3d_spatial_arm_plane_n3_traj` | 3 | 1 |  | traj version | 3-DoF spatial arm plane manifold with trajectory-style sampling |
| `3d_sphere_surface` | 3 | 1 |  | base version | Sphere surface |
| `3d_sphere_surface_traj` | 3 | 1 |  | traj version | Sphere surface with trajectory-style sampling |
| `3d_spiral` | 3 | 2 |  | no traj variant | 3D helix/spiral curve manifold |
| `3d_torus_surface` | 3 | 1 | | base version; paper uses `_traj` by default | Torus surface |
| `3d_torus_surface_traj` | 3 | 1 | `3DTorus` | default paper variant | Torus surface with trajectory-style sampling |
| `3d_twosphere` | 3 | 1 |  | base version; paper uses `_traj` by default | Outer boundary of two-sphere union |
| `3d_twosphere_traj` | 3 | 1 | `3DTwoSphere` | default paper variant | Two-sphere manifold with trajectory-style sampling |
| `3d_vz_2d_discontinuous` | 3 | 2 |  | derived lift, non-traj | Lifted 2D discontinuous dataset with varying `z(x,y)` |
| `3d_vz_2d_ellipse` | 3 | 2 |  | base version; paper uses `_traj` by default | Lifted 2D ellipse dataset with varying `z(x,y)` |
| `3d_vz_2d_ellipse_traj` | 3 | 2 | `3DTwistedEllip` | default paper variant | Lifted 2D ellipse dataset with varying `z(x,y)` and trajectory-style sampling |
| `3d_vz_2d_figure_eight` | 3 | 2 |  | derived lift, non-traj | Lifted 2D figure-eight dataset with varying `z(x,y)` |
| `3d_vz_2d_hetero_noise` | 3 | 2 |  | derived lift, non-traj | Lifted 2D hetero-noise dataset with varying `z(x,y)` |
| `3d_vz_2d_looped_spiro` | 3 | 2 |  | derived lift, non-traj | Lifted 2D looped-spiro dataset with varying `z(x,y)` |
| `3d_vz_2d_noisy_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D noisy-sine dataset with varying `z(x,y)` |
| `3d_vz_2d_planar_arm_line_n2` | 3 | 2 |  | derived lift, non-traj | Lifted 2D planar-arm-line dataset with varying `z(x,y)` |
| `3d_vz_2d_sharp_star` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sharp-star dataset with varying `z(x,y)` |
| `3d_vz_2d_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sine dataset with varying `z(x,y)` |
| `3d_vz_2d_sparse_sine` | 3 | 2 |  | derived lift, non-traj | Lifted 2D sparse-sine dataset with varying `z(x,y)` |
| `3d_vz_2d_square` | 3 | 2 |  | derived lift, non-traj | Lifted 2D square dataset with varying `z(x,y)` |
| `6d_spatial_arm_up_n6` | 6 | 2 |  | base version, pybullet backend | 6-DoF UR5-style arm upward-orientation set using the pybullet backend |
| `6d_spatial_arm_up_n6_py` | 6 | 2 | | base version, analytic backend | 6-DoF UR5-style arm upward-orientation set using the analytic backend |
| `6d_spatial_arm_up_n6_py_traj` | 6 | 2 | `6DArmUp` | default paper variant | 6-DoF UR5-style arm upward-orientation set with trajectory-style sampling |
| `6d_workspace_sine_surface_pose` | 6 | 3 |  | base version; paper uses `_traj` by default | Workspace pose `[x,y,z,roll,pitch,yaw]`: position on a sine-wave surface, local z-axis aligned with surface normal, free spin around that normal |
| `6d_workspace_sine_surface_pose_traj` | 6 | 3 | `6DSinePose` | default paper variant | Sine-surface workspace-pose manifold with trajectory-style sampling |
| `12d_dual_arm` | 12 | 10 |  | base version; paper uses `_traj` by default | Dual-arm virtual-link pose `[pose1(6), pose2(6)]`: link center moves on a vertical ribbon from a planar guide curve |
| `12d_dual_arm_traj` | 12 | 10 | `12DDualArm` | default paper variant | Dual-arm vertical-ribbon virtual-link pose with trajectory-style sampling |

## Derived Naming Rules

- `3d_vz_<base_2d_dataset>`: lift a 2D base dataset to 3D with varying `z(x,y)` (`data_dim=3`, `codim=2`)
- `3d_0z_<base_2d_dataset>`: lift a 2D base dataset to 3D with `z=0` (`data_dim=3`, `codim=2`)
- `_traj` suffix: trajectory-style sampling variant of the same underlying constraint

## Return Format

- `x_train`: training points on manifold
- `grid`: dense manifold samples used as reference/evaluation set

## UR5 Dataset Helpers

Files:
- `datasets/ur5_n6_dataset.py`
- `datasets/ur5_pybullet_utils.py`

Key functions:
- `sample_ur5_upward_dataset(...)`: pybullet-based UR5 upward-orientation dataset sampling
- `sample_ur5_upward_dataset_analytic(...)`: analytic/no-pybullet approximation sampler

## Vendored UR5 Assets

Path:
- `datasets/assets/UR5+gripper/`

Contains URDF plus mesh/texture resources copied into this repo for reproducible open-source usage:
- `ur5_gripper.urdf`
- `mesh/`
- `textures/`
