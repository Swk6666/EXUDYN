"""Flexible double pendulum using Craig–Bampton reduced-order models.

Each pendulum link is exported as a floating-frame (FFRF) flexible body built
from a Netgen/NGSolve mesh and reduced with the Hurty–Craig–Bampton method.
The two bodies are coupled with revolute joints realized via super-element
markers while initial angles/velocities mimic a classical rigid double pendulum.

The script can optionally export the full simulation history (node trajectories)
into a MATLAB-compatible ``.mat`` file for post-processing or video generation.

Prerequisites: NGsolve/Netgen must be installed and importable from Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import exudyn as exu
from exudyn.utilities import *  # noqa: F401,F403 - convenience import used by Exudyn examples
import exudyn.graphics as graphics  # noqa: F401 - convenient shorthand
from exudyn.FEM import (
    FEMinterface,
    HCBstaticModeSelection,
    ObjectFFRFreducedOrderInterface,
)


@dataclass
class FlexibleLinkParameters:
    """Geometric/material data and CMS options shared by both flexible links."""

    length_first: float = 0.5 *2          # [m]
    length_second: float = 0.4 *2         # [m]
    width: float = 0.012               # [m]
    thickness: float = 0.003           # [m]
    youngs_modulus: float = 8.1e8     # [Pa]
    density: float = 1180.0            # [kg/m^3]
    poisson: float = 0.34
    modes_first: int = 8
    modes_second: int = 8
    mesh_maxh: float = 0.01            # target element size for Netgen mesh
    mass_proportional_damping: float = 2e-3
    stiffness_proportional_damping: float = 8e-3
    gravity: float = 9.81              # [m/s^2]

    def cross_section(self) -> Tuple[float, float]:
        area = self.width * self.thickness
        inertia = self.width * self.thickness**3 / 12.0
        return area, inertia


@dataclass
class InitialState:
    """Initial joint angles (relative to gravity) and angular rates."""

    theta1: float = math.radians(45.0)   # link 1 w.r.t. downward vertical
    theta2: float = math.radians(-35.0)  # relative link 2 angle w.r.t. link 1
    omega1: float = 0.6                  # link 1 angular rate [rad/s]
    omega2: float = -0.8                 # relative link 2 angular rate [rad/s]


@dataclass
class SimulationSettings:
    end_time: float = 20.0
    step_size: float = 1e-3
    render: bool = False
    frames_per_second: int = 60
    store_trajectory: bool = True
    matlab_export: Optional[Path] = Path("output/flexible_pendulum_sim.mat")


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _planar_velocity(omega: float, position: np.ndarray) -> np.ndarray:
    return np.cross(np.array([0.0, 0.0, omega]), position)


def _collect_element_arrays(element_sets) -> Dict[str, np.ndarray]:
    """Flatten FEMinterface element dictionaries to numpy arrays.
    输入的 element_sets 是从 Exudyn 的 FEMinterface 对象中获取的单元信息。这个信息的原始格式是一个列表，列表中的每个成员都是一个字典，代表一种类型的有限元单元（比如四面体、六面体等）。
    element_sets = [
    {
        'Name': 'Tetrahedron',
        'Nodes': [[0, 1, 2, 3], [4, 5, 6, 7], ...],  # 很多四面体单元
        'SomeOtherInfo': [...] 
    },
    {
        'Name': 'Hexahedron',
        'Nodes': [[8, 9, 10, 11, 12, 13, 14, 15], ...], # 很多六面体单元
        'SomeOtherInfo': [...]
    },
    # 可能还有其他类型的单元...
]
这个函数的目标是把这些信息按“键”（key）重新组织，将所有类型单元中具有相同键（例如 'Nodes'）的数据合并到一个大的 NumPy 数组中。
{
    'Nodes': np.array([
        [0, 1, 2, 3], 
        [4, 5, 6, 7], 
        ...,
        [8, 9, 10, 11, 12, 13, 14, 15],
        ...
    ]),
    'SomeOtherInfo': np.array([...]) 
}
    """
    collected: Dict[str, list[np.ndarray]] = {} #初始化一个名为 collected 的空字典。
    for element_dict in element_sets:
        for key, value in element_dict.items():
            if key == "Name":
                continue
            arr = np.array(value, dtype=int)
            if arr.size == 0:
                continue
            collected.setdefault(key, []).append(arr)
    return {key: np.vstack(arr_list) for key, arr_list in collected.items()}


def _build_cms_beam(
    length: float,
    params: FlexibleLinkParameters,
    n_modes: int,
    label: str,
) -> Dict[str, object]:
    """Create FEM mesh, compute Craig–Bampton modes, and cache interface data."""
    try:
        import ngsolve as ngs
        from netgen.csg import CSGeometry, OrthoBrick, Pnt
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Craig–Bampton version requires NGsolve/Netgen; install it to run this example."
        ) from exc

    geo = CSGeometry()
    block = OrthoBrick(
        Pnt(0.0, -params.thickness * 0.5, -params.width * 0.5),
        Pnt(length, params.thickness * 0.5, params.width * 0.5),
    )
    geo.Add(block)

    mesh = ngs.Mesh(geo.GenerateMesh(maxh=params.mesh_maxh))
    mesh.Curve(1)

    fem = FEMinterface()
    fem.ImportMeshFromNGsolve(
        mesh,
        density=params.density,
        youngsModulus=params.youngs_modulus,
        poissonsRatio=params.poisson,
    )

    nodes_root = fem.GetNodesInPlane([0, 0, 0], [1, 0, 0])
    nodes_tip = fem.GetNodesInPlane([length, 0, 0], [1, 0, 0])
    if len(nodes_root) == 0 or len(nodes_tip) == 0:
        raise RuntimeError(f"Failed to identify boundary nodes for {label} beam")

    positions = fem.GetNodePositionsAsArray()
    root_mean = positions[nodes_root].mean(axis=0)
    tip_mean = positions[nodes_tip].mean(axis=0)

    weights_root = np.ones(len(nodes_root)) / len(nodes_root)
    weights_tip = np.ones(len(nodes_tip)) / len(nodes_tip)

    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[nodes_root, nodes_tip],
        nEigenModes=n_modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2,
    )

    if not fem.surface:
        fem.VolumeToSurfaceElements()

    surface_trigs_list = fem.GetSurfaceTriangles()
    surface_trigs = (
        np.array(surface_trigs_list, dtype=int)
        if len(surface_trigs_list) > 0
        else np.zeros((0, 3), dtype=int)
    )

    return {
        "fem": fem,
        "nodes_root": np.array(nodes_root, dtype=int),
        "nodes_tip": np.array(nodes_tip, dtype=int),
        "weights_root": weights_root,
        "weights_tip": weights_tip,
        "root_mean": root_mean,
        "tip_mean": tip_mean,
        "nodes": positions,
        "surface_trigs": surface_trigs,
        "elements": _collect_element_arrays(fem.elements),
    }


def _create_superelement_marker(
    mbs: exu.MainSystem,
    body_number: int,
    mesh_nodes: np.ndarray,
    weights: np.ndarray,
    offset: Iterable[float] | None = None,
    show: bool = True,
) -> int:
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    return mbs.AddMarker(
        MarkerSuperElementRigid(
            bodyNumber=body_number,
            meshNodeNumbers=mesh_nodes,
            weightingFactors=weights,
            offset=offset,
            useAlternativeApproach=True,
            visualization=VMarkerSuperElementRigid(show=show),
        )
    )


def _register_node_sensors(
    mbs: exu.MainSystem,
    body_number: int,
    num_nodes: int,
) -> list[int]:
    sensors: list[int] = []
    for node_idx in range(num_nodes):
        sensors.append(
            mbs.AddSensor(
                SensorSuperElement(
                    bodyNumber=body_number,
                    meshNodeNumber=int(node_idx),
                    outputVariableType=exu.OutputVariableType.Position,
                    storeInternal=True,
                )
            )
        )
    return sensors


def build_system(
    params: FlexibleLinkParameters,
    init: InitialState,
    sim: SimulationSettings,
) -> Dict[str, object]:
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    # Visual ground for reference
    background = graphics.CheckerBoard(size=params.length_first + params.length_second + 0.6)
    o_ground = mbs.AddObject(
        ObjectGround(
            referencePosition=[0, 0, 0],
            visualization=VObjectGround(graphicsData=[background]),
        )
    )

    # Build CMS representations for both links
    beam_first = _build_cms_beam(params.length_first, params, params.modes_first, "first")
    beam_second = _build_cms_beam(params.length_second, params, params.modes_second, "second")

    # Pre-compute initial kinematics
    psi1 = init.theta1 - 0.5 * math.pi
    psi2 = init.theta1 + init.theta2 - 0.5 * math.pi

    rot1 = _rot_z(psi1)
    rot2 = _rot_z(psi2)

    omega1_vec = np.array([0.0, 0.0, init.omega1])
    omega2_total = init.omega1 + init.omega2
    omega2_vec = np.array([0.0, 0.0, omega2_total])

    root_mean_1 = beam_first["root_mean"]
    tip_mean_1 = beam_first["tip_mean"]
    root_mean_2 = beam_second["root_mean"]

    # Place first flexible body so that its root coincides with the ground hinge
    pos_first = -rot1 @ root_mean_1
    vel_first = -np.cross(omega1_vec, rot1 @ root_mean_1)

    cms_first = ObjectFFRFreducedOrderInterface(beam_first["fem"])
    obj_first = cms_first.AddObjectFFRFreducedOrder(
        mbs,
        positionRef=list(pos_first),
        initialVelocity=list(vel_first),
        rotationMatrixRef=rot1,
        initialAngularVelocity=omega1_vec,
        massProportionalDamping=params.mass_proportional_damping,
        stiffnessProportionalDamping=params.stiffness_proportional_damping,
        gravity=[0.0, -params.gravity, 0.0],
        color=[0.1, 0.6, 0.9, 1.0],
    )

    hinge_vector_local = tip_mean_1 - root_mean_1
    hinge_pos = rot1 @ hinge_vector_local
    hinge_vel = np.cross(omega1_vec, hinge_pos)

    # Place second flexible body so that its root sits at the hinge position
    pos_second = hinge_pos - rot2 @ root_mean_2
    vel_second = hinge_vel - np.cross(omega2_vec, rot2 @ root_mean_2)

    cms_second = ObjectFFRFreducedOrderInterface(beam_second["fem"])
    obj_second = cms_second.AddObjectFFRFreducedOrder(
        mbs,
        positionRef=list(pos_second),
        initialVelocity=list(vel_second),
        rotationMatrixRef=rot2,
        initialAngularVelocity=omega2_vec,
        massProportionalDamping=params.mass_proportional_damping,
        stiffnessProportionalDamping=params.stiffness_proportional_damping,
        gravity=[0.0, -params.gravity, 0.0],
        color=[0.9, 0.4, 0.2, 1.0],
    )

    # Create markers for joints
    m_ground = mbs.AddMarker(
        MarkerBodyRigid(
            bodyNumber=o_ground,
            localPosition=[0.0, 0.0, 0.0],
            visualization=VMarkerBodyRigid(show=False),
        )
    )

    m_first_root = _create_superelement_marker(
        mbs,
        obj_first["oFFRFreducedOrder"],
        beam_first["nodes_root"],
        beam_first["weights_root"],
        offset=[0.0, 0.0, 0.0],
    )

    m_first_tip = _create_superelement_marker(
        mbs,
        obj_first["oFFRFreducedOrder"],
        beam_first["nodes_tip"],
        beam_first["weights_tip"],
        offset=[0.0, 0.0, 0.0],
    )

    m_second_root = _create_superelement_marker(
        mbs,
        obj_second["oFFRFreducedOrder"],
        beam_second["nodes_root"],
        beam_second["weights_root"],
        offset=[0.0, 0.0, 0.0],
    )

    m_second_tip = _create_superelement_marker(
        mbs,
        obj_second["oFFRFreducedOrder"],
        beam_second["nodes_tip"],
        beam_second["weights_tip"],
        offset=[0.0, 0.0, 0.0],
    )

    joint_axes = [1, 1, 1, 1, 1, 0]  # revolute about z-axis
    mbs.AddObject(
        GenericJoint(
            markerNumbers=[m_ground, m_first_root],
            constrainedAxes=joint_axes,
            visualization=VGenericJoint(show=False),
        )
    )

    mbs.AddObject(
        GenericJoint(
            markerNumbers=[m_first_tip, m_second_root],
            constrainedAxes=joint_axes,
            visualization=VGenericJoint(show=False),
        )
    )

    # Optional sensor at second link tip
    tip_sensor = mbs.AddSensor(
        SensorMarker(
            markerNumber=m_second_tip,
            outputVariableType=exu.OutputVariableType.Position,
            storeInternal=True,
        )
    )

    sensors_first: list[int] = []
    sensors_second: list[int] = []
    if sim.store_trajectory:
        sensors_first = _register_node_sensors(mbs, obj_first["oFFRFreducedOrder"], beam_first["nodes"].shape[0])
        sensors_second = _register_node_sensors(mbs, obj_second["oFFRFreducedOrder"], beam_second["nodes"].shape[0])

    return {
        "SC": SC,
        "mbs": mbs,
        "obj_first": obj_first,
        "obj_second": obj_second,
        "tip_sensor": tip_sensor,
        "hinge_pos": hinge_pos,
        "beam_first": beam_first,
        "beam_second": beam_second,
        "rot1": rot1,
        "rot2": rot2,
        "pos_first": pos_first,
        "pos_second": pos_second,
        "sensor_numbers_beam1": sensors_first,
        "sensor_numbers_beam2": sensors_second,
    }


def _gather_sensor_positions(
    mbs: exu.MainSystem,
    sensor_numbers: list[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not sensor_numbers:
        return np.zeros((0,), dtype=float), np.zeros((0, 0, 0), dtype=float)

    first_data = np.array(mbs.GetSensorStoredData(sensor_numbers[0]), dtype=float)
    times = first_data[:, 0]
    n_steps = first_data.shape[0]
    dim = first_data.shape[1] - 1
    num_sensors = len(sensor_numbers)

    positions = np.zeros((n_steps, num_sensors, dim), dtype=float)
    positions[:, 0, :] = first_data[:, 1:]

    for idx, sensor in enumerate(sensor_numbers[1:], start=1):
        sensor_data = np.array(mbs.GetSensorStoredData(sensor), dtype=float)
        if sensor_data.shape[0] != n_steps:
            raise ValueError("Sensor data length mismatch; ensure uniform sensorsWritePeriod")
        positions[:, idx, :] = sensor_data[:, 1:]

    return times, positions


def _export_matlab_data(
    model: Dict[str, object],
    params: FlexibleLinkParameters,
    sim: SimulationSettings,
    mbs: exu.MainSystem,
    export_path: Path,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError:  # pragma: no cover - SciPy optional
        exu.Print(f"SciPy not available; skipping MATLAB export to {export_path}")
        return

    export_path.parent.mkdir(parents=True, exist_ok=True)

    beam1 = model["beam_first"]
    beam2 = model["beam_second"]

    data: Dict[str, np.ndarray] = {
        "gravity": np.array([0.0, -params.gravity, 0.0], dtype=float),
        "frames_per_second": np.array([sim.frames_per_second], dtype=float),
    }

    def _populate(prefix: str, beam: Dict[str, object], rot: np.ndarray, pos: np.ndarray) -> None:
        nodes = np.asarray(beam["nodes"], dtype=float)
        data[f"{prefix}_nodes"] = nodes
        data[f"{prefix}_root_mean"] = np.asarray(beam["root_mean"], dtype=float)
        data[f"{prefix}_tip_mean"] = np.asarray(beam["tip_mean"], dtype=float)
        data[f"{prefix}_root_nodes"] = np.asarray(beam["nodes_root"], dtype=int) + 1
        data[f"{prefix}_tip_nodes"] = np.asarray(beam["nodes_tip"], dtype=int) + 1
        data[f"{prefix}_weights_root"] = np.asarray(beam["weights_root"], dtype=float)
        data[f"{prefix}_weights_tip"] = np.asarray(beam["weights_tip"], dtype=float)
        surface = np.asarray(beam["surface_trigs"], dtype=int)
        if surface.size:
            data[f"{prefix}_surface_trigs"] = surface + 1
        else:
            data[f"{prefix}_surface_trigs"] = np.zeros((0, 3), dtype=int)
        for elem_type, elem_array in beam["elements"].items():
            data[f"{prefix}_elements_{elem_type}"] = np.asarray(elem_array, dtype=int) + 1
        data[f"{prefix}_reference_rotation"] = np.asarray(rot, dtype=float)
        data[f"{prefix}_reference_position"] = np.asarray(pos, dtype=float)

    _populate("beam1", beam1, np.asarray(model["rot1"]), np.asarray(model["pos_first"]))
    _populate("beam2", beam2, np.asarray(model["rot2"]), np.asarray(model["pos_second"]))

    times1, pos1 = _gather_sensor_positions(mbs, model["sensor_numbers_beam1"])
    times2, pos2 = _gather_sensor_positions(mbs, model["sensor_numbers_beam2"])

    if times1.size and times2.size:
        if not np.allclose(times1, times2):
            exu.Print("Warning: time vectors for beam 1 and beam 2 differ; using beam 1 times")
        data["time"] = times1
    elif times1.size:
        data["time"] = times1
    else:
        data["time"] = times2

    if pos1.size:
        data["beam1_node_positions"] = pos1
    if pos2.size:
        data["beam2_node_positions"] = pos2

    savemat(str(export_path), data)
    exu.Print(f"Saved MATLAB simulation data to {export_path}")


def run_simulation(
    params: FlexibleLinkParameters = FlexibleLinkParameters(),
    init: InitialState = InitialState(),
    sim: SimulationSettings = SimulationSettings(),
) -> None:
    model = build_system(params, init, sim)
    SC: exu.SystemContainer = model["SC"]
    mbs: exu.MainSystem = model["mbs"]

    mbs.Assemble()

    settings = exu.SimulationSettings()
    settings.timeIntegration.endTime = sim.end_time
    settings.timeIntegration.numberOfSteps = int(sim.end_time / sim.step_size)
    settings.timeIntegration.verboseMode = 1
    settings.timeIntegration.newton.useModifiedNewton = True
    settings.timeIntegration.generalizedAlpha.spectralRadius = 0.85
    settings.timeIntegration.adaptiveStep = True
    settings.solutionSettings.writeSolutionToFile = False
    if sim.store_trajectory:
        settings.solutionSettings.sensorsWritePeriod = 1.0 / sim.frames_per_second
    else:
        settings.solutionSettings.sensorsWritePeriod = sim.step_size
    settings.linearSolverType = exu.LinearSolverType.EigenSparse
    settings.displayStatistics = True

    if sim.render:
        SC.visualizationSettings.general.autoFitScene = True
        SC.visualizationSettings.window.renderWindowSize = [1280, 720]
        SC.visualizationSettings.connectors.showJointAxes = False
        SC.visualizationSettings.bodies.deformationScaleFactor = 1.0
        SC.visualizationSettings.openGL.multiSampling = 4
        SC.visualizationSettings.nodes.drawNodesAsPoint = False
        SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.Displacement
        SC.visualizationSettings.contour.outputVariableComponent = 1
        SC.renderer.Start()
        SC.renderer.DoIdleTasks()

    try:
        success = mbs.SolveDynamic(settings, solverType=exu.DynamicSolverType.TrapezoidalIndex2)
        if not success:
            exu.Print("Dynamic solver reported failure; inspect log for details.")
    finally:
        if sim.render:
            SC.renderer.DoIdleTasks()
            SC.renderer.Stop()

    tip_pos = mbs.GetSensorValues(model["tip_sensor"])
    exu.Print(f"Tip position at t = {sim.end_time:.2f} s: {tip_pos}")

    if sim.matlab_export is not None:
        _export_matlab_data(model, params, sim, mbs, Path(sim.matlab_export))


if __name__ == "__main__":
    import time
    start_time = time.time()
    params = FlexibleLinkParameters()
    init = InitialState()
    sim = SimulationSettings()
    run_simulation(params, init, sim)
    end_time = time.time()
    exu.Print(f"Time taken: {end_time - start_time:.2f} seconds")