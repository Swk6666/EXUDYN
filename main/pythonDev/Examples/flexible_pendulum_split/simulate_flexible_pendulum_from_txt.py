"""Rebuild the flexible double pendulum simulation from exported FEM text data."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

import exudyn as exu
import exudyn.graphics as graphics
from exudyn.utilities import *  # noqa: F401,F403
from exudyn.FEM import FEMinterface, ObjectFFRFreducedOrderInterface


@dataclass
class FlexibleLinkParameters:
    length_first: float = 1.0
    length_second: float = 0.8
    width: float = 0.012
    thickness: float = 0.003
    youngs_modulus: float = 8.1e8
    density: float = 1180.0
    poisson: float = 0.34
    modes_first: int = 8
    modes_second: int = 8
    mesh_maxh: float = 0.01
    mass_proportional_damping: float = 2e-3
    stiffness_proportional_damping: float = 8e-3
    gravity: float = 9.81


@dataclass
class InitialState:
    theta1: float = math.radians(45.0)
    theta2: float = math.radians(-35.0)
    omega1: float = 0.6
    omega2: float = -0.8


@dataclass
class SimulationSettings:
    end_time: float = 10.0
    step_size: float = 1e-3
    render: bool = True
    frames_per_second: int = 60
    store_trajectory: bool = True
    matlab_export: Optional[str] = "output/flexible_pendulum_sim.mat"
    adaptive_step: bool = False
    spectral_radius: float = 0.85
    modified_newton: bool = True
    newton_tol: float = 1e-8
    newton_max_iter: int = 12


def _dict_to_csr(data: Dict[str, Any] | None) -> csr_matrix | None:
    if data is None:
        return None
    return csr_matrix(
        (
            np.asarray(data["data"], dtype=float),
            np.asarray(data["indices"], dtype=int),
            np.asarray(data["indptr"], dtype=int),
        ),
        shape=tuple(data["shape"]),
    )


def _deserialize_fem(fem_data: Dict[str, Any]) -> FEMinterface:
    fem = FEMinterface()

    nodes = {key: np.asarray(value, dtype=float) for key, value in fem_data["nodes"].items()}

    elements: List[Dict[str, Any]] = []
    for item in fem_data["elements"]:
        converted: Dict[str, Any] = {"Name": item["Name"]}
        for key, value in item.items():
            if key == "Name":
                continue
            converted[key] = np.asarray(value, dtype=int)
        elements.append(converted)

    surface: List[Dict[str, Any]] = []
    for item in fem_data["surface"]:
        converted: Dict[str, Any] = {"Name": item["Name"]}
        if "Trigs" in item:
            converted["Trigs"] = np.asarray(item["Trigs"], dtype=int)
        if "Quads" in item:
            converted["Quads"] = np.asarray(item["Quads"], dtype=int)
        surface.append(converted)

    node_sets: List[Dict[str, Any]] = []
    for item in fem_data["nodeSets"]:
        converted = {
            "Name": item["Name"],
            "NodeNumbers": np.asarray(item["NodeNumbers"], dtype=int),
            "NodeWeights": np.asarray(item["NodeWeights"], dtype=float),
        }
        node_sets.append(converted)

    element_sets: List[Dict[str, Any]] = []
    for item in fem_data["elementSets"]:
        converted = {
            "Name": item["Name"],
            "ElementNumbers": np.asarray(item["ElementNumbers"], dtype=int),
        }
        element_sets.append(converted)

    mode_basis: Dict[str, Any] = {}
    for key, value in fem_data["modeBasis"].items():
        if isinstance(value, list):
            mode_basis[key] = np.asarray(value, dtype=float)
        else:
            mode_basis[key] = value

    post_modes: Dict[str, Any] = {}
    for key, value in fem_data.get("postProcessingModes", {}).items():
        if isinstance(value, list):
            post_modes[key] = np.asarray(value, dtype=float)
        elif isinstance(value, dict):
            post_modes[key] = {
                k: (np.asarray(v, dtype=float) if isinstance(v, list) else v)
                for k, v in value.items()
            }
        else:
            post_modes[key] = value

    fem_dict = {
        "nodes": nodes,
        "elements": elements,
        "massMatrix": _dict_to_csr(fem_data.get("massMatrix")),
        "stiffnessMatrix": _dict_to_csr(fem_data.get("stiffnessMatrix")),
        "surface": surface,
        "nodeSets": node_sets,
        "elementSets": element_sets,
        "modeBasis": mode_basis,
        "eigenValues": np.asarray(fem_data.get("eigenValues", []), dtype=float),
        "postProcessingModes": post_modes,
    }

    fem.SetWithDictionary(fem_dict, warn=False)
    return fem


def _collect_elements(fem: FEMinterface) -> Dict[str, np.ndarray]:
    collected: Dict[str, List[np.ndarray]] = {}
    for element_dict in fem.elements:
        for key, value in element_dict.items():
            if key == "Name":
                continue
            collected.setdefault(key, []).append(np.asarray(value, dtype=int))
    return {key: np.vstack(value) for key, value in collected.items()}


def _load_beam(beam_data: Dict[str, Any]) -> Dict[str, Any]:
    fem = _deserialize_fem(beam_data["fem"])
    surface_trigs = np.asarray(beam_data.get("surface_trigs", []), dtype=int)
    if surface_trigs.size == 0:
        surface_trigs = np.zeros((0, 3), dtype=int)

    return {
        "fem": fem,
        "nodes_root": np.asarray(beam_data["nodes_root"], dtype=int),
        "nodes_tip": np.asarray(beam_data["nodes_tip"], dtype=int),
        "weights_root": np.asarray(beam_data["weights_root"], dtype=float),
        "weights_tip": np.asarray(beam_data["weights_tip"], dtype=float),
        "root_mean": np.asarray(beam_data["root_mean"], dtype=float),
        "tip_mean": np.asarray(beam_data["tip_mean"], dtype=float),
        "surface_trigs": surface_trigs,
        "nodes": fem.GetNodePositionsAsArray(),
        "elements": _collect_elements(fem),
    }


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def _register_node_sensors(mbs: exu.MainSystem, body_number: int, num_nodes: int) -> List[int]:
    sensors: List[int] = []
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


def _gather_sensor_positions(mbs: exu.MainSystem, sensor_numbers: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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
    model: Dict[str, Any],
    params: FlexibleLinkParameters,
    sim: SimulationSettings,
    mbs: exu.MainSystem,
    export_path: Path,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError:  # pragma: no cover
        exu.Print(f"SciPy not available; skipping MATLAB export to {export_path}")
        return

    export_path.parent.mkdir(parents=True, exist_ok=True)

    beam1 = model["beam_first"]
    beam2 = model["beam_second"]

    data: Dict[str, np.ndarray] = {
        "gravity": np.array([0.0, -params.gravity, 0.0], dtype=float),
        "frames_per_second": np.array([sim.frames_per_second], dtype=float),
    }

    def _populate(prefix: str, beam: Dict[str, Any], rot: np.ndarray, pos: np.ndarray) -> None:
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


def _build_system(
    params: FlexibleLinkParameters,
    init: InitialState,
    sim: SimulationSettings,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    background = graphics.CheckerBoard(size=params.length_first + params.length_second + 0.6)
    o_ground = mbs.AddObject(
        ObjectGround(
            referencePosition=[0, 0, 0],
            visualization=VObjectGround(graphicsData=[background]),
        )
    )

    beam_first = _load_beam(dataset["beam_first"])
    beam_second = _load_beam(dataset["beam_second"])

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

    sensors_first: List[int] = []
    sensors_second: List[int] = []
    if sim.store_trajectory:
        sensors_first = _register_node_sensors(mbs, obj_first["oFFRFreducedOrder"], beam_first["nodes"].shape[0])
        sensors_second = _register_node_sensors(mbs, obj_second["oFFRFreducedOrder"], beam_second["nodes"].shape[0])

    return {
        "SC": SC,
        "mbs": mbs,
        "cms_first": cms_first,
        "cms_second": cms_second,
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
        "vel_first": vel_first,
        "vel_second": vel_second,
        "omega1_vec": omega1_vec,
        "omega2_vec": omega2_vec,
        "sensor_numbers_beam1": sensors_first,
        "sensor_numbers_beam2": sensors_second,
    }


def _solve(model: Dict[str, Any], sim: SimulationSettings) -> None:
    mbs: exu.MainSystem = model["mbs"]
    mbs.Assemble()

    settings = exu.SimulationSettings()
    settings.timeIntegration.endTime = sim.end_time
    settings.timeIntegration.numberOfSteps = int(sim.end_time / sim.step_size)
    settings.timeIntegration.verboseMode = 1
    settings.timeIntegration.newton.useModifiedNewton = sim.modified_newton
    settings.timeIntegration.newton.relativeTolerance = sim.newton_tol
    settings.timeIntegration.newton.absoluteTolerance = sim.newton_tol
    settings.timeIntegration.newton.maxIterations = sim.newton_max_iter
    settings.timeIntegration.generalizedAlpha.spectralRadius = sim.spectral_radius
    settings.timeIntegration.adaptiveStep = sim.adaptive_step
    settings.solutionSettings.writeSolutionToFile = False
    if sim.store_trajectory:
        settings.solutionSettings.sensorsWritePeriod = 1.0 / sim.frames_per_second

    if sim.render:
        exu.StartRenderer()

    try:
        exu.SolveDynamic(mbs, settings)
    finally:
        if sim.render:
            exu.StopRenderer()


def run_simulation(
    dataset_path: Path,
    matlab_override: Optional[str] = None,
    disable_matlab: bool = False,
    render_override: Optional[bool] = None,
) -> None:
    dataset = json.loads(dataset_path.read_text())

    params = FlexibleLinkParameters(**dataset["parameters"])
    init = InitialState(**dataset["initial_state"])
    sim_dict = dict(dataset["simulation"])
    if disable_matlab:
        sim_dict["matlab_export"] = None
    elif matlab_override is not None:
        sim_dict["matlab_export"] = matlab_override
    if render_override is not None:
        sim_dict["render"] = render_override
    sim = SimulationSettings(**sim_dict)

    model = _build_system(params, init, sim, dataset)
    _solve(model, sim)
    tip_pos = model["mbs"].GetSensorValues(model["tip_sensor"])
    exu.Print(f"Tip position at t = s: {tip_pos}")
    if sim.matlab_export:
        export_path = Path(sim.matlab_export)
        _export_matlab_data(model, params, sim, model["mbs"], export_path)


def main() -> None:
    default_input = Path(__file__).resolve().with_name("flexible_pendulum_fem_data.txt")
    parser = argparse.ArgumentParser(description="Simulate the flexible double pendulum from exported FEM data")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to the JSON text dataset")
    parser.add_argument("--matlab-export", type=str, default=None, help="Override MATLAB export path")
    parser.add_argument("--no-matlab-export", action="store_true", help="Disable MATLAB export regardless of stored setting")
    parser.add_argument("--render", action="store_true", help="Enable renderer regardless of stored setting")
    parser.add_argument("--no-render", action="store_true", help="Disable renderer regardless of stored setting")
    args = parser.parse_args()

    render_override: Optional[bool] = None
    if args.render and args.no_render:
        raise ValueError("--render and --no-render cannot be used together")
    if args.render:
        render_override = True
    elif args.no_render:
        render_override = False

    run_simulation(
        args.input,
        matlab_override=args.matlab_export,
        disable_matlab=args.no_matlab_export,
        render_override=render_override,
    )



if __name__ == "__main__":
    main()
