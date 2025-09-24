"""Generate Craig–Bampton reduced flexible pendulum data and simulate its dynamics.

This script mirrors ``flexible_pendulum.py`` but stores the intermediate FEM
artifacts and the dynamic response so they can be reused by a standalone C++
implementation. The pipeline is:

1. Build two beam meshes with Netgen/NGSolve and create Craig–Bampton reduced
   models using Exudyn's ``FEMinterface``.
2. Store the reduced-order data to disk (``*.npz`` files) alongside auxiliary
   information needed to recreate the floating-frame bodies.
3. Assemble and simulate the flexible double pendulum for 20 seconds.
4. Persist the simulated node trajectories for later comparison with a C++
   implementation.

The output directory defaults to ``data`` relative to this script and will be
created if necessary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import exudyn as exu
from exudyn.utilities import *  # noqa: F401,F403 - intentional re-use from example
import exudyn.graphics as graphics  # noqa: F401
from exudyn.FEM import (
    FEMinterface,
    HCBstaticModeSelection,
    ObjectFFRFreducedOrderInterface,
)


@dataclass
class FlexibleLinkParameters:
    length_first: float = 0.5 * 2
    length_second: float = 0.4 * 2
    width: float = 0.012
    thickness: float = 0.003
    youngs_modulus: float = 5.0e9
    density: float = 1180.0
    poisson: float = 0.34
    modes_first: int = 16
    modes_second: int = 16
    mesh_maxh: float = 0.01
    mass_proportional_damping: float = 2e-3
    stiffness_proportional_damping: float = 8e-3
    gravity: float = 9.81

    def cross_section(self) -> Tuple[float, float]:
        area = self.width * self.thickness
        inertia = self.width * self.thickness**3 / 12.0
        return area, inertia


@dataclass
class InitialState:
    theta1: float = math.radians(45.0)
    theta2: float = math.radians(-35.0)
    omega1: float = 0.6
    omega2: float = -0.8


@dataclass
class SimulationSettings:
    end_time: float = 20.0
    step_size: float = 1e-3
    render: bool = False
    frames_per_second: int = 60
    store_trajectory: bool = True


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _planar_velocity(omega: float, position: np.ndarray) -> np.ndarray:
    return np.cross(np.array([0.0, 0.0, omega]), position)


def _collect_element_arrays(element_sets) -> Dict[str, np.ndarray]:
    collected: Dict[str, list[np.ndarray]] = {}
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
    try:
        import ngsolve as ngs
        from netgen.csg import CSGeometry, OrthoBrick, Pnt
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "NGsolve/Netgen is required for the Craig–Bampton export; install it first."
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

    background = graphics.CheckerBoard(size=params.length_first + params.length_second + 0.6)
    o_ground = mbs.AddObject(
        ObjectGround(
            referencePosition=[0, 0, 0],
            visualization=VObjectGround(graphicsData=[background]),
        )
    )

    beam_first = _build_cms_beam(params.length_first, params, params.modes_first, "first")
    beam_second = _build_cms_beam(params.length_second, params, params.modes_second, "second")

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
    )

    m_first_tip = _create_superelement_marker(
        mbs,
        obj_first["oFFRFreducedOrder"],
        beam_first["nodes_tip"],
        beam_first["weights_tip"],
    )

    m_second_root = _create_superelement_marker(
        mbs,
        obj_second["oFFRFreducedOrder"],
        beam_second["nodes_root"],
        beam_second["weights_root"],
    )

    m_second_tip = _create_superelement_marker(
        mbs,
        obj_second["oFFRFreducedOrder"],
        beam_second["nodes_tip"],
        beam_second["weights_tip"],
    )

    joint_axes = [1, 1, 1, 1, 1, 0]
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
        "cms_first": cms_first,
        "cms_second": cms_second,
        "beam_first": beam_first,
        "beam_second": beam_second,
        "tip_sensor": tip_sensor,
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


def _save_beam_data(beam: Dict[str, object], cms: ObjectFFRFreducedOrderInterface, prefix: Path) -> None:
    fem_path = prefix.with_suffix(".npz")
    cms.SaveToFile(str(prefix))  # SaveToFile appends extension automatically

    aux_path = prefix.parent / f"{prefix.name}_aux.npz"
    np.savez(
        aux_path,
        nodes=beam["nodes"],
        nodes_root=beam["nodes_root"],
        nodes_tip=beam["nodes_tip"],
        weights_root=beam["weights_root"],
        weights_tip=beam["weights_tip"],
        root_mean=beam["root_mean"],
        tip_mean=beam["tip_mean"],
        surface_trigs=beam["surface_trigs"],
        **{f"elements_{key}": value for key, value in beam["elements"].items()},
    )

    exu.Print(f"Saved Craig–Bampton data: {fem_path} and {aux_path}")


def _save_simulation_response(
    output_dir: Path,
    model: Dict[str, object],
    mbs: exu.MainSystem,
) -> Path:
    beam1_times, beam1_positions = _gather_sensor_positions(mbs, model["sensor_numbers_beam1"])
    beam2_times, beam2_positions = _gather_sensor_positions(mbs, model["sensor_numbers_beam2"])

    tip_data = np.array(mbs.GetSensorStoredData(model["tip_sensor"]), dtype=float)

    response_path = output_dir / "python_response.npz"
    np.savez(
        response_path,
        beam1_times=beam1_times,
        beam1_positions=beam1_positions,
        beam2_times=beam2_times,
        beam2_positions=beam2_positions,
        tip_sensor=tip_data,
    )
    exu.Print(f"Stored simulation response to {response_path}")
    return response_path


def run_pipeline(
    params: FlexibleLinkParameters = FlexibleLinkParameters(),
    init: InitialState = InitialState(),
    sim: SimulationSettings = SimulationSettings(),
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    settings.solutionSettings.sensorsWritePeriod = 1.0 / sim.frames_per_second
    settings.linearSolverType = exu.LinearSolverType.EigenSparse
    settings.displayStatistics = True

    exu.Print("Starting Python Craig–Bampton simulation...")
    success = mbs.SolveDynamic(settings, solverType=exu.DynamicSolverType.TrapezoidalIndex2)
    if not success:
        exu.Print("Dynamic solver did not report success; inspect log output.")

    tip_pos = mbs.GetSensorValues(model["tip_sensor"])
    exu.Print(f"Python simulation tip position at t = {sim.end_time:.2f} s: {tip_pos}")

    _save_beam_data(model["beam_first"], model["cms_first"], output_dir / "beam_first")
    _save_beam_data(model["beam_second"], model["cms_second"], output_dir / "beam_second")
    _save_simulation_response(output_dir, model, mbs)


if __name__ == "__main__":
    run_pipeline()
