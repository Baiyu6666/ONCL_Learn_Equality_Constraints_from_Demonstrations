# Datasets Overview

This file summarizes the dataset generators and naming conventions used in this project.

Main entry:
- `datasets/constraint_datasets.py`
- function: `generate_dataset(name, cfg)`

## Naming Notes

- Repository dataset names such as `3d_torus_surface_traj` are internal ids used by configs and code.
- The paper uses cleaner environment names such as `3DTorus`.
- When both a base version and a `_traj` version exist, the `_traj` version is the default environment variant used in the paper experiments.
- Non-`_traj` versions use scatter samples of the true constraint; `_traj` versions use trajectory demonstrations on the same underlying constraint.

## Benchmark Datasets Used in the Paper

These are the main environments used in the paper benchmark.

| Paper Env Name    | Dataset ID                            | Data Dim | Codim | Traj | Description                                                                                                                                                              |
| ----------------- | ------------------------------------- | -------: | ----: | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `2DEllipse`       | `2d_ellipse`                          | 2        | 1     | no   | 2D ellipse curve with gaps in training coverage used to test interpolation across missing segments.                                                                      |
| `2DPlanarArmLine` | `2d_planar_arm_line_n2`               | 2        | 1     | no   | 2-DoF planar arm whose end-effector is constrained to a workspace line; learning is performed in joint space.                                                            |
| `2DSineSparse`    | `2d_sparse_sine`                      | 2        | 1     | no   | Sparse samples from a planar sinusoidal curve.                                                                                                                           |
| `3DTwistEllipse`  | `3d_vz_2d_ellipse_traj`               | 3        | 2     | yes  | Twisted 3D lifting of a 2D ellipse, used in trajectory-demonstration form in the paper.                                                                                  |
| `3DPlanarArmLine` | `3d_planar_arm_line_n3_traj`          | 3        | 1     | yes  | 3-DoF planar arm whose end-effector is constrained to a workspace line; paper uses the trajectory variant.                                                               |
| `3DArmEllipse`    | `3d_spatial_arm_ellip_n3_traj`        | 3        | 2     | yes  | 3-DoF spatial arm whose end-effector follows an elliptical task-space curve; paper uses the trajectory variant.                                                          |
| `3DTorus`         | `3d_torus_surface_traj`               | 3        | 1     | yes  | Torus surface in 3D, used in trajectory-demonstration form in the paper.                                                                                                 |
| `3DTwoSphere`     | `3d_twosphere_traj`                   | 3        | 1     | yes  | Outer boundary of a two-sphere union, used in trajectory-demonstration form in the paper.                                                                                |
| `6DSinePose`      | `6d_workspace_sine_surface_pose_traj` | 6        | 3     | yes  | 6D workspace pose constraint: position lies on a sine surface, local tool axis aligns with the surface normal, and spin about the normal is free.                        |
| `6DArmUp`         | `6d_spatial_arm_up_n6_py_traj`        | 6        | 2     | yes  | 6-DoF UR5-style arm with an upward-facing end-effector orientation constraint and unconstrained position; paper uses the trajectory variant.                             |
| `12DDualArm`      | `12d_dual_arm_traj`                   | 12       | 10    | yes  | Dual-arm virtual-link pose constraint: two end-effectors keep fixed span while the object center moves on a vertical ribbon and the link axis follows the guide tangent. |

## Additional Datasets Provided by the Project

The repository also includes several additional datasets that are not part of the main paper benchmark. These are useful for debugging, ablations, or extending the benchmark suite.

| Dataset ID                       | Data Dim | Codim | Traj Note                      | Description                                                                       |
| -------------------------------- | -------: | ----: | ------------------------------ | --------------------------------------------------------------------------------- |
| `2d_discontinuous`               | 2        | 1     | no traj variant                | Piecewise/discontinuous sine-like curve                                           |
| `2d_figure_eight`                | 2        | 1     | no traj variant                | 2D figure-eight curve                                                             |
| `2d_hetero_noise`                | 2        | 1     | no traj variant                | Manifold with non-uniform noise level                                             |
| `2d_looped_spiro`                | 2        | 1     | no traj variant                | Multi-loop spirograph-like curve                                                  |
| `2d_noisy_sine`                  | 2        | 1     | no traj variant                | Sine curve with stronger noise                                                    |
| `2d_sharp_star`                  | 2        | 1     | no traj variant                | Star-like closed curve with sharp corners                                         |
| `2d_sine`                        | 2        | 1     | no traj variant                | Clean sine curve                                                                  |
| `2d_square`                      | 2        | 1     | no traj variant                | 2D square boundary manifold                                                       |
| `3d_0z_2d_discontinuous`         | 3        | 2     | derived lift, non-traj         | Lifted 2D discontinuous dataset with `z=0`                                        |
| `3d_0z_2d_ellipse`               | 3        | 2     | derived lift, non-traj         | Lifted 2D ellipse dataset with `z=0`                                              |
| `3d_0z_2d_figure_eight`          | 3        | 2     | derived lift, non-traj         | Lifted 2D figure-eight dataset with `z=0`                                         |
| `3d_0z_2d_hetero_noise`          | 3        | 2     | derived lift, non-traj         | Lifted 2D hetero-noise dataset with `z=0`                                         |
| `3d_0z_2d_looped_spiro`          | 3        | 2     | derived lift, non-traj         | Lifted 2D looped-spiro dataset with `z=0`                                         |
| `3d_0z_2d_noisy_sine`            | 3        | 2     | derived lift, non-traj         | Lifted 2D noisy-sine dataset with `z=0`                                           |
| `3d_0z_2d_planar_arm_line_n2`    | 3        | 2     | derived lift, non-traj         | Lifted 2D planar-arm-line dataset with `z=0`                                      |
| `3d_0z_2d_sharp_star`            | 3        | 2     | derived lift, non-traj         | Lifted 2D sharp-star dataset with `z=0`                                           |
| `3d_0z_2d_sine`                  | 3        | 2     | derived lift, non-traj         | Lifted 2D sine dataset with `z=0`                                                 |
| `3d_0z_2d_sparse_sine`           | 3        | 2     | derived lift, non-traj         | Lifted 2D sparse-sine dataset with `z=0`                                          |
| `3d_0z_2d_square`                | 3        | 2     | derived lift, non-traj         | Lifted 2D square dataset with `z=0`                                               |
| `3d_paraboloid`                  | 3        | 1     | base version                   | 3D paraboloid surface manifold                                                    |
| `3d_paraboloid_traj`             | 3        | 1     | traj version                   | 3D paraboloid surface manifold with trajectory-style sampling                     |
| `3d_planar_arm_line_n3`          | 3        | 1     | base version                   | 3-DoF planar arm line manifold without trajectory grouping                        |
| `3d_saddle_surface`              | 3        | 1     | base version                   | Saddle-type surface in 3D                                                         |
| `3d_saddle_surface_traj`         | 3        | 1     | traj version                   | Saddle-type surface in 3D with trajectory-style sampling                          |
| `3d_spatial_arm_circle_n3`       | 3        | 2     | no traj variant                | 3-DoF spatial arm constrained on a workspace circle                               |
| `3d_spatial_arm_ellip_n3`        | 3        | 2     | base version                   | 3-DoF spatial arm ellipse manifold without trajectory grouping                    |
| `3d_spatial_arm_plane_n3`        | 3        | 1     | base version                   | 3-DoF spatial arm constrained on a workspace plane                                |
| `3d_spatial_arm_plane_n3_traj`   | 3        | 1     | traj version                   | 3-DoF spatial arm plane manifold with trajectory-style sampling                   |
| `3d_sphere_surface`              | 3        | 1     | base version                   | Sphere surface                                                                    |
| `3d_sphere_surface_traj`         | 3        | 1     | traj version                   | Sphere surface with trajectory-style sampling                                     |
| `3d_spiral`                      | 3        | 2     | no traj variant                | 3D helix/spiral curve manifold                                                    |
| `3d_torus_surface`               | 3        | 1     | base version                   | Torus surface without trajectory grouping                                         |
| `3d_twosphere`                   | 3        | 1     | base version                   | Two-sphere manifold without trajectory grouping                                   |
| `3d_vz_2d_discontinuous`         | 3        | 2     | derived lift, non-traj         | Lifted 2D discontinuous dataset with varying `z(x,y)`                             |
| `3d_vz_2d_ellipse`               | 3        | 2     | base version                   | Lifted 2D ellipse dataset with varying `z(x,y)` without trajectory grouping       |
| `3d_vz_2d_figure_eight`          | 3        | 2     | derived lift, non-traj         | Lifted 2D figure-eight dataset with varying `z(x,y)`                              |
| `3d_vz_2d_hetero_noise`          | 3        | 2     | derived lift, non-traj         | Lifted 2D hetero-noise dataset with varying `z(x,y)`                              |
| `3d_vz_2d_looped_spiro`          | 3        | 2     | derived lift, non-traj         | Lifted 2D looped-spiro dataset with varying `z(x,y)`                              |
| `3d_vz_2d_noisy_sine`            | 3        | 2     | derived lift, non-traj         | Lifted 2D noisy-sine dataset with varying `z(x,y)`                                |
| `3d_vz_2d_planar_arm_line_n2`    | 3        | 2     | derived lift, non-traj         | Lifted 2D planar-arm-line dataset with varying `z(x,y)`                           |
| `3d_vz_2d_sharp_star`            | 3        | 2     | derived lift, non-traj         | Lifted 2D sharp-star dataset with varying `z(x,y)`                                |
| `3d_vz_2d_sine`                  | 3        | 2     | derived lift, non-traj         | Lifted 2D sine dataset with varying `z(x,y)`                                      |
| `3d_vz_2d_sparse_sine`           | 3        | 2     | derived lift, non-traj         | Lifted 2D sparse-sine dataset with varying `z(x,y)`                               |
| `3d_vz_2d_square`                | 3        | 2     | derived lift, non-traj         | Lifted 2D square dataset with varying `z(x,y)`                                    |
| `6d_spatial_arm_up_n6`           | 6        | 2     | base version, pybullet backend | 6-DoF UR5-style arm upward-orientation set using the pybullet backend             |
| `6d_spatial_arm_up_n6_py`        | 6        | 2     | base version, analytic backend | 6-DoF UR5-style arm upward-orientation set using the analytic backend             |
| `6d_workspace_sine_surface_pose` | 6        | 3     | base version                   | Scatter-sample version of the sine-surface pose manifold                          |
| `12d_dual_arm`                   | 12       | 10    | base version                   | Scatter-sample version of the dual-arm vertical-ribbon virtual-link pose manifold |

## Derived Naming Rules

- `3d_vz_<base_2d_dataset>`: lift a 2D base dataset to 3D with varying `z(x,y)` (`data_dim=3`, `codim=2`)
- `3d_0z_<base_2d_dataset>`: lift a 2D base dataset to 3D with `z=0` (`data_dim=3`, `codim=2`)
- `_traj` suffix: trajectory-style sampling variant of the same underlying constraint

## Return Format

- `x_train`: training points on the manifold
- `grid`: dense manifold samples used as reference and evaluation sets

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

Contains URDF plus mesh and texture resources copied into this repo for reproducible open-source usage:
- `ur5_gripper.urdf`
- `mesh/`
- `textures/`
