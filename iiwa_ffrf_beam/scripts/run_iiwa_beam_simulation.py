"""Assemble and simulate the iiwa + flexible beam scenario in Exudyn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import exudyn as exu
from exudyn.itemInterface import GenericJoint, MarkerKinematicTreeRigid, MarkerSuperElementRigid, VGenericJoint
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink
from exudyn.robotics.utilities import GetURDFrobotData, LoadURDFrobot, HT2rotationMatrix, HT2translation
from exudyn.graphics import CheckerBoard
from exudyn.FEM import FEMinterface, ObjectFFRFreducedOrderInterface, HCBstaticModeSelection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG_PATH = PROJECT_ROOT / "data" / "simulation_config.json"

@dataclass
class TimeSeriesTrajectory:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    hold_final: bool = True

    @classmethod
    def from_array(cls, times: np.ndarray, positions: np.ndarray, hold_final: bool = True) -> "TimeSeriesTrajectory":
        if times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if positions.ndim != 2 or positions.shape[0] != times.shape[0]:
            raise ValueError("positions must be a 2D array with len(times) rows")
        velocities = cls._finite_difference(times, positions)
        accelerations = cls._finite_difference(times, velocities)
        return cls(times, positions, velocities, accelerations, hold_final)

    @staticmethod
    def _finite_difference(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values)
        dt = np.diff(times)
        dt[dt == 0.0] = 1e-12
        if len(times) > 1:
            result[0] = (values[1] - values[0]) / dt[0]
            result[-1] = (values[-1] - values[-2]) / dt[-1]
        if len(times) > 2:
            result[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])
        return result

    def _interp(self, array: np.ndarray, t: float) -> np.ndarray:
        left = array[0]
        right = array[-1] if self.hold_final else array[0]
        return np.array([
            np.interp(t, self.times, array[:, i], left=left[i], right=right[i])
            for i in range(array.shape[1])
        ])

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.hold_final:
            t = np.clip(t, self.times[0], self.times[-1])
        q = self._interp(self.positions, t)
        dq = self._interp(self.velocities, t)
        ddq = self._interp(self.accelerations, t)
        return q, dq, ddq



def resolve_project_path(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    for root in (PROJECT_ROOT, PROJECT_ROOT.parent):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (PROJECT_ROOT / candidate).resolve()

def load_json(path_str: str) -> Dict:
    path = resolve_project_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def load_trajectory(csv_path_str: str, hold_final: bool) -> TimeSeriesTrajectory:
    csv_path = resolve_project_path(csv_path_str)
    if not csv_path.exists():
        exu.Print(f"WARNING: trajectory file {csv_path} not found. Falling back to zero trajectory.")
        times = np.array([0.0, 1.0])
        positions = np.zeros((2, 7))
        return TimeSeriesTrajectory.from_array(times, positions, hold_final)

    raw = np.genfromtxt(csv_path, delimiter=",", comments="#")
    if raw.ndim == 1:
        raise ValueError("Trajectory file must contain at least two rows (time + positions)")
    times = raw[:, 0]
    positions = raw[:, 1:]
    return TimeSeriesTrajectory.from_array(times, positions, hold_final)



def get_interface_config(beam_cfg: Dict, name: str) -> Dict:
    for iface in beam_cfg.get("interfaces", []):
        if iface.get("name") == name:
            return iface
    raise KeyError(f"Interface '{name}' not defined in beam configuration")

def build_robot(mbs: exu.MainSystem, sim_cfg: Dict) -> Dict:
    urdf_cfg = sim_cfg["urdf"]
    base_path = resolve_project_path(urdf_cfg["base_path"])
    urdf_file = resolve_project_path(urdf_cfg["file"])
    try:
        load_result = LoadURDFrobot(str(urdf_file), str(base_path))
    except ImportError as exc:
        raise ImportError("Robotics Toolbox for Python is required. Install with 'pip install roboticstoolbox-python'.") from exc

    robot = load_result["robot"]
    urdf = load_result["urdf"]
    robot_data = GetURDFrobotData(robot, urdf, verbose=1)

    gravity = urdf_cfg.get("gravity", [0.0, 0.0, -9.81])

    mbs_robot = Robot(
        gravity=gravity,
        base=RobotBase(),
        tool=RobotTool(),
        referenceConfiguration=list(robot_data["staticJointValues"])
    )

    link_numbers = {"None": -1, "base_link": -1}
    for idx, link in enumerate(robot_data["linkList"]):
        link_numbers[link["name"]] = idx

    number_of_joints = robot_data["numberOfJoints"]
    p_control = np.concatenate([
        np.array([8e4, 6e4, 6e4, 2e4, 2e4, 1e4, 1e4]),
        5e3 * np.ones(max(0, number_of_joints - 7))
    ])
    d_control = np.concatenate([
        np.array([4e3, 3e3, 3e3, 1.2e3, 1.2e3, 600.0, 600.0]),
        300.0 * np.ones(max(0, number_of_joints - 7))
    ])

    for link_data in robot_data["linkList"]:
        vis = link_data.get("graphicsDataList", [])
        link_vis = VRobotLink(graphicsData=vis, showMBSjoint=False)
        parent_name = link_data.get("parentName", "None")
        parent_index = link_numbers.get(parent_name, -1)
        joint_number = link_data.get("jointNumber", None)
        if joint_number is not None and joint_number < len(p_control):
            pd_pair = (float(p_control[joint_number]), float(d_control[joint_number]))
        else:
            pd_pair = (0.0, 0.0)
        mbs_robot.AddLink(RobotLink(
            mass=link_data["mass"],
            parent=parent_index,
            COM=link_data["com"],
            inertia=link_data["inertiaCOM"],
            preHT=link_data["preHT"],
            jointType=link_data["jointType"],
            PDcontrol=pd_pair,
            visualization=link_vis
        ))

    robot_dict = mbs_robot.CreateKinematicTree(mbs)
    q_ref = np.array(robot_data["staticJointValues"])
    link_transforms = mbs_robot.LinkHT(q_ref)
    return {
        "robot": mbs_robot,
        "robot_data": robot_data,
        "kinematic_tree": robot_dict,
        "gravity": gravity,
        "link_transforms_ref": link_transforms,
        "reference_joint_values": q_ref
    }

def load_fem_with_cms(beam_cfg: Dict, sim_cfg: Dict) -> FEMinterface:
    fem = FEMinterface()
    cache_base = resolve_project_path(beam_cfg["fem_cache_basename"])
    cms_suffix = sim_cfg.get("fem_cache_suffix", "_cms")
    cms_path = Path(str(cache_base) + cms_suffix)
    cms_exists = cms_path.with_suffix(".npz").exists()
    load_path = cms_path if cms_exists else cache_base
    exu.Print(f"Loading FEM data from {load_path}.npz")
    fem.LoadFromFile(str(load_path))
    if not cms_exists:
        boundary_nodes = []
        boundary_weights = []
        for iface in beam_cfg.get("interfaces", []):
            nodes, weights = select_interface_nodes(fem, iface)
            boundary_nodes.append(nodes)
            boundary_weights.append(weights)
        n_eigen = beam_cfg.get("n_eigen_modes", 12)
        use_sparse = beam_cfg.get("use_sparse_solver", True)
        mode_name = beam_cfg.get("hcb_mode", "RBE2")
        hcb_mode = getattr(HCBstaticModeSelection, mode_name)
        exu.Print(f"Computing {n_eigen} CMS modes (mode={mode_name}) ...")
        fem.ComputeHurtyCraigBamptonModes(
            boundaryNodesList=boundary_nodes,
            nEigenModes=n_eigen,
            useSparseSolver=use_sparse,
            computationMode=hcb_mode
        )
        try:
            fem.SaveToFile(str(cms_path))
            exu.Print(f"CMS data saved to {cms_path}.npz")
        except Exception as exc:  # pylint: disable=broad-except
            exu.Print(f"WARNING: failed to save CMS data: {exc}")
    return fem

def select_interface_nodes(fem: FEMinterface, interface_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    selection = interface_cfg.get("selection_type", "plane").lower()
    name = interface_cfg.get("name", "interface")
    if selection == "plane":
        point = interface_cfg.get("point", [0.0, 0.0, 0.0])
        normal = interface_cfg.get("normal", [1.0, 0.0, 0.0])
        tolerance = interface_cfg.get("tolerance", 1e-4)
        nodes = fem.GetNodesInPlane(point, normal, tolerance=tolerance)
    elif selection == "node_list":
        nodes = interface_cfg.get("node_numbers", [])
    else:
        raise ValueError(f"Unsupported selection_type '{selection}' for interface '{name}'")
    if not nodes:
        raise RuntimeError(f"Interface '{name}' selection returned no nodes")
    weights_mode = interface_cfg.get("weighting", "uniform").lower()
    if weights_mode == "uniform":
        weights = np.ones(len(nodes)) / len(nodes)
    else:
        weights = np.array(interface_cfg.get("weights", []), dtype=float)
        if len(weights) != len(nodes):
            raise ValueError(f"Interface '{name}' weights length mismatch")
    return np.array(nodes, dtype=int), np.array(weights, dtype=float)

def build_beam_object(mbs: exu.MainSystem, fem: FEMinterface, beam_cfg: Dict, gravity: Tuple[float, float, float]) -> Dict:
    cms = ObjectFFRFreducedOrderInterface(fem)
    damping = beam_cfg.get("stiffness_proportional_damping", 0.0)
    color_rgba = beam_cfg.get("visualization_color", [0.3, 0.7, 1.0, 0.8])
    position_ref = beam_cfg.get("reference_position", [0.0, 0.0, 0.0])
    rotation_ref = beam_cfg.get("reference_rotation", [])
    ffrf_dict = cms.AddObjectFFRFreducedOrder(
        mbs,
        positionRef=position_ref,
        rotationMatrixRef=rotation_ref,
        initialVelocity=[0.0, 0.0, 0.0],
        initialAngularVelocity=[0.0, 0.0, 0.0],
        stiffnessProportionalDamping=damping,
        gravity=gravity,
        color=color_rgba
    )
    return ffrf_dict

def attach_beam_to_robot(mbs: exu.MainSystem, robot_info: Dict, beam_info: Dict, beam_cfg: Dict, sim_cfg: Dict, fem: FEMinterface) -> int:
    attach_cfg = sim_cfg["attachment"]
    link_index = attach_cfg.get("link_index", -1)
    kinematic_tree_obj = robot_info["kinematic_tree"]["objectKinematicTree"]
    robot_marker_local = attach_cfg.get("robot_marker_local_position", [0.0, 0.0, 0.0])

    marker_robot = mbs.AddMarker(MarkerKinematicTreeRigid(
        objectNumber=kinematic_tree_obj,
        linkNumber=link_index,
        localPosition=robot_marker_local
    ))

    interface_cfg = get_interface_config(beam_cfg, attach_cfg.get("beam_interface", "base"))
    nodes, weights = select_interface_nodes(fem, interface_cfg)
    beam_marker_offset = attach_cfg.get("beam_marker_offset", [0.0, 0.0, 0.0])
    marker_beam = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam_info["oFFRFreducedOrder"],
        meshNodeNumbers=nodes,
        weightingFactors=weights,
        offset=beam_marker_offset
    ))

    rotation0 = np.array(attach_cfg.get("rotation_marker0", np.eye(3)))
    rotation1 = np.array(attach_cfg.get("rotation_marker1", np.eye(3)))

    joint = mbs.AddObject(GenericJoint(
        markerNumbers=[marker_robot, marker_beam],
        constrainedAxes=[1, 1, 1, 1, 1, 1],
        rotationMarker0=rotation0,
        rotationMarker1=rotation1,
        visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)
    ))
    return joint

def configure_visualization(sc: exu.SystemContainer, sim_cfg: Dict) -> None:
    viz_cfg = sim_cfg.get("visualization", {})
    sc.visualizationSettings.connectors.showJointAxes = True
    sc.visualizationSettings.connectors.jointAxesLength = 0.05
    sc.visualizationSettings.connectors.jointAxesRadius = 0.004
    sc.visualizationSettings.markers.show = True
    sc.visualizationSettings.markers.defaultSize = 0.01
    if viz_cfg.get("show_beam_contour", True):
        variable_name = viz_cfg.get("contour_variable", "DisplacementLocal")
        try:
            output_var = getattr(exu.OutputVariableType, variable_name)
            sc.visualizationSettings.contour.outputVariable = output_var
            sc.visualizationSettings.contour.outputVariableComponent = -1
        except AttributeError:
            exu.Print(f"WARNING: unknown contour variable '{variable_name}', keeping defaults")

def configure_simulation_settings(sim_cfg: Dict) -> exu.SimulationSettings:
    sim_settings = exu.SimulationSettings()
    end_time = sim_cfg["simulation"].get("end_time", 2.0)
    step_size = sim_cfg["simulation"].get("step_size", 0.001)
    sim_settings.timeIntegration.numberOfSteps = int(end_time / step_size)
    sim_settings.timeIntegration.endTime = end_time
    sim_settings.solutionSettings.solutionWritePeriod = sim_cfg["simulation"].get("solution_write_period", 0.01)
    sim_settings.solutionSettings.sensorsWritePeriod = sim_cfg["simulation"].get("sensor_write_period", 0.01)
    solver_type = sim_cfg["simulation"].get("solver_type", "GeneralizedAlpha")
    sim_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
    sim_settings.timeIntegration.generalizedAlpha.spectralRadius = 0.9
    sim_settings.timeIntegration.newton.useModifiedNewton = True
    sim_settings.timeIntegration.newton.maxIterations = 15
    sim_settings.timeIntegration.newton.relativeTolerance = 1e-6
    sim_settings.timeIntegration.verboseMode = 1
    return sim_settings

def main() -> None:
    sim_cfg = load_json(SIM_CONFIG_PATH)
    beam_cfg = load_json(sim_cfg["beam_config_path"])

    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    mbs.CreateGround(graphicsDataList=[CheckerBoard(point=[0, 0, -0.001], size=6)])

    robot_info = build_robot(mbs, sim_cfg)
    attach_cfg = sim_cfg["attachment"]
    attach_index = attach_cfg.get("link_index", 0)
    attach_transform = robot_info["link_transforms_ref"][attach_index]
    if "reference_position" not in beam_cfg:
        beam_cfg["reference_position"] = HT2translation(attach_transform).tolist()
    if "reference_rotation" not in beam_cfg:
        beam_cfg["reference_rotation"] = HT2rotationMatrix(attach_transform).tolist()

    fem = load_fem_with_cms(beam_cfg, sim_cfg)
    beam_info = build_beam_object(mbs, fem, beam_cfg, robot_info["gravity"])

    attach_beam_to_robot(mbs, robot_info, beam_info, beam_cfg, sim_cfg, fem)

    traj = load_trajectory(sim_cfg["trajectory_csv"], sim_cfg.get("trajectory_hold_final", True))
    oKT = robot_info["kinematic_tree"]["objectKinematicTree"]
    nKT = robot_info["kinematic_tree"]["nodeGeneric"]
    q0, dq0, _ = traj.evaluate(0.0)

    mbs.SetNodeParameter(nKT, "initialCoordinates", q0.tolist())
    mbs.SetNodeParameter(nKT, "initialCoordinates_t", dq0.tolist())

    def prestep_callback(mbs_local: exu.MainSystem, time: float) -> bool:  # pylint: disable=unused-argument
        q_ref, dq_ref, _ = traj.evaluate(time)
        mbs_local.SetObjectParameter(oKT, "jointPositionOffsetVector", q_ref.tolist())
        mbs_local.SetObjectParameter(oKT, "jointVelocityOffsetVector", dq_ref.tolist())
        return True

    mbs.SetPreStepUserFunction(prestep_callback)

    static_settings = exu.SimulationSettings()
    static_settings.staticSolver.newton.useModifiedNewton = True
    static_settings.staticSolver.newton.relativeTolerance = 1e-6
    static_settings.staticSolver.newton.maxIterations = 20
    static_settings.staticSolver.verboseMode = 0
    try:
        mbs.SolveStatic(static_settings)
    except Exception as exc:
        exu.Print(f"WARNING: static solve failed: {exc}")

    sc.visualizationSettings.general.drawWorldBasis = True
    sc.visualizationSettings.window.renderWindowSize = [1600, 900]
    configure_visualization(sc, sim_cfg)

    mbs.Assemble()

    sim_settings = configure_simulation_settings(sim_cfg)

    gui_available = getattr(exu, 'GUIavailable', lambda: False)()
    if gui_available:
        sc.renderer.Start()
        if hasattr(sc.renderer, 'ZoomAll'):
            sc.renderer.ZoomAll()

    solver_name = sim_cfg["simulation"].get("solver_type", "GeneralizedAlpha")
    solver_enum = getattr(exu.DynamicSolverType, solver_name, exu.DynamicSolverType.TrapezoidalIndex2)
    mbs.SolveDynamic(sim_settings, solverType=solver_enum)

    if gui_available:
        sc.renderer.Stop()


if __name__ == "__main__":
    main()
