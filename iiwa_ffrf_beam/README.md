# iiwa + Flexible Beam (Exudyn)

This folder stores the scripts and configuration needed to simulate a rigid iiwa-14 robot that is rigidly connected to a flexible beam modeled with Exudyn's FFRF + HCB formulation. FEM data is expected from NGsolve (either generated locally or imported from external files).

## Structure
- `data/fem/beam_config.json`: beam geometry/material/mode setup; edit this before computing CMS data.
- `data/fem/beam_setup_report.json`: written by `beam_ffrf_setup.py` to log the CMS export that was generated.
- `data/trajectories/joint_trajectory.csv`: expected joint-space trajectory (time column + seven joint angles in radians).
- `data/simulation_config.json`: robot attachment, solver settings, and trajectory path.
- `scripts/beam_ffrf_setup.py`: prepares FEM data (imports NGsolve mesh, computes Hurty–Craig–Bampton modes, stores CMS file, and writes a report).
- `scripts/run_iiwa_beam_simulation.py`: assembles the robot + beam system, applies the prescribed trajectory, and runs the simulation.

## Workflow
1. **Prepare FEM data**
   - Adjust `beam_config.json` to describe your beam and interface (plane definition or explicit node list).
   - If NGsolve is available and `generate_with_ngsolve` is `true`, `beam_ffrf_setup.py` will create a simple box mesh; otherwise place your NGsolve-exported `.npz` under `data/fem` and set `generate_with_ngsolve` to `false`.
   - Run `python scripts/beam_ffrf_setup.py` to compute the CMS data. The script writes `*_cms.npz` next to the base FEM cache and records a summary in `beam_setup_report.json`.

2. **Provide joint trajectory**
   - Create `data/trajectories/joint_trajectory.csv` with columns `[time, q1, q2, q3, q4, q5, q6, q7]` in radians.
   - Times must be monotonically increasing. The simulation extrapolates the last sample if `trajectory_hold_final` is `true`.

3. **Run the coupled simulation**
   - Review `data/simulation_config.json` to match your attachment point (robot link index, marker offsets, orientation) and solver settings.
   - Execute `python scripts/run_iiwa_beam_simulation.py`. The script will automatically load the CMS file (or recompute it if only the base FEM cache exists), import the iiwa URDF, lock the chosen interface to the beam, and integrate the motion.

## Dependencies
- Exudyn (already in the repository).
- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) and `pymeshlab` for URDF import / mesh processing.
- NGsolve + Netgen (optional; only needed when regenerating the FEM mesh from geometry).

Update the configuration files before running the scripts to match the beam data, joint trajectory, and workspace layout on your machine.

## Assumptions
- The iiwa URDF from `iiwa_description/urdf/iiwa14.urdf` is available together with the meshes referenced inside the file.
- The beam is rigidly attached to the robot link specified in `simulation_config.json`; adjust `rotation_marker0/1` and marker offsets if your tool frame differs.
- NGsolve-generated FEM caches are stored under `data/fem` using the basename configured in `beam_config.json`.
- Joint angles in the trajectory CSV are expressed in radians and match the ordering of the URDF joints.


## Quick Start Script
- Run `python scripts/iiwa_cantilever_simulation.py` to regenerate the cantilever with NGsolve, attach it to the iiwa end effector, and execute the 10 s trajectory where only joint 2 moves from 0 to 1 rad. The script is self-contained and does not rely on the JSON configuration files.
