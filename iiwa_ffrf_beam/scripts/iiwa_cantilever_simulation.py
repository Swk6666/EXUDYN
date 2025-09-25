"""Full pipeline: build a cantilever beam via NGsolve, couple it to the iiwa-14 robot,
and drive joint 2 from 0 to 1 rad in 10 s using Exudyn."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import exudyn as exu
from exudyn.FEM import FEMinterface, HCBstaticModeSelection, ObjectFFRFreducedOrderInterface
from exudyn.graphics import CheckerBoard
from exudyn.itemInterface import GenericJoint, MarkerBodyRigid, MarkerKinematicTreeRigid, MarkerSuperElementRigid, VGenericJoint, SensorMarker
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink
from exudyn.robotics.utilities import GetURDFrobotData, LoadURDFrobot, HT2rotationMatrix, HT2translation
from exudyn.utilities import HTtranslate

BEAM_LENGTH = 2.0
BEAM_WIDTH = 0.05
BEAM_HEIGHT = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_FILE = PROJECT_ROOT / "iiwa_description" / "urdf" / "iiwa14.urdf"
URDF_BASE_PATH = PROJECT_ROOT / "iiwa_description"

# --------------------------- FEM preparation ---------------------------------

def build_cantilever_fem() -> Tuple[FEMinterface, np.ndarray, np.ndarray]:
    """Create a rectangular cantilever beam with NGsolve and compute CMS data."""
    fem = FEMinterface()

    import ngsolve as ngs
    from netgen.occ import Box, OCCGeometry

    length = 2
    width = 0.05
    height = 0.05
    mesh_order = 2
    mesh_size = width * 0.5

    box = Box((0.0, -0.5 * width, -0.5 * height), (length, 0.5 * width, 0.5 * height))
    geometry = OCCGeometry(box)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=mesh_size))

    density = 7800.0
    youngs = 2.1e10
    poisson = 0.3

    fem.ImportMeshFromNGsolve(mesh,
                              density=density,
                              youngsModulus=youngs,
                              poissonsRatio=poisson,
                              meshOrder=mesh_order)

    wall_nodes = fem.GetNodesInPlane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    if not wall_nodes:
        raise RuntimeError("No nodes found on cantilever base plane")
    wall_weights = np.ones(len(wall_nodes)) / len(wall_nodes)
    exu.Print(f"Cantilever wall interface uses {len(wall_nodes)} nodes")

    tip_nodes = fem.GetNodesInPlane([length, 0.0, 0.0], [1.0, 0.0, 0.0])
    if not tip_nodes:
        raise RuntimeError("No nodes found on cantilever tip plane")
    tip_weights = np.ones(len(tip_nodes)) / len(tip_nodes)

    n_modes = 8
    exu.Print(f"Computing {n_modes} Hurty-Craig-Bampton modes ...")
    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[wall_nodes, tip_nodes],
        nEigenModes=n_modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2
    )

    interface_data = {
        "wall": (np.array(wall_nodes, dtype=int), wall_weights),
        "tip": (np.array(tip_nodes, dtype=int), tip_weights)
    }

    return fem, interface_data

# --------------------------- Robot assembly ----------------------------------

def build_robot(mbs: exu.MainSystem, base_offset=None) -> Dict:
    try:
        load_result = LoadURDFrobot(str(URDF_FILE.resolve()), str(URDF_BASE_PATH.resolve()))
    except ImportError as exc:
        raise ImportError("Robotics Toolbox for Python is required (pip install roboticstoolbox-python).") from exc

    robot = load_result["robot"]
    urdf = load_result["urdf"]
    robot_data = GetURDFrobotData(robot, urdf, verbose=1)

    gravity = [0.0, 0.0, -9.81]

    if base_offset is None:
        base_offset = [0.0, 0.0, 0.0]

    robot_model = Robot(gravity=gravity,
                        base=RobotBase(HT=HTtranslate(base_offset)),
                        tool=RobotTool(),
                        referenceConfiguration=list(robot_data["staticJointValues"]))

    link_numbers = {"None": -1, "base_link": -1}
    for idx, link in enumerate(robot_data["linkList"]):
        link_numbers[link["name"]] = idx

    number_of_joints = robot_data["numberOfJoints"]
    p_control = np.concatenate([
        np.array([2e4, 1.5e4, 1.5e4, 8e3, 8e3, 4e3, 4e3]),
        2e3 * np.ones(max(0, number_of_joints - 7))
    ])
    d_control = np.concatenate([
        np.array([1500.0, 1200.0, 1200.0, 600.0, 600.0, 300.0, 300.0]),
        150.0 * np.ones(max(0, number_of_joints - 7))
    ])

    for link in robot_data["linkList"]:
        graphics = link.get("graphicsDataList", [])
        vis = VRobotLink(graphicsData=graphics, showMBSjoint=False)
        parent_name = link.get("parentName", "None")
        parent_index = link_numbers.get(parent_name, -1)
        joint_number = link.get("jointNumber", None)
        if joint_number is not None and joint_number < len(p_control):
            pd_pair = (float(p_control[joint_number]), float(d_control[joint_number]))
        else:
            pd_pair = (0.0, 0.0)
        robot_model.AddLink(RobotLink(mass=link["mass"],
                                      parent=parent_index,
                                      COM=link["com"],
                                      inertia=link["inertiaCOM"],
                                      preHT=link["preHT"],
                                      jointType=link["jointType"],
                                      PDcontrol=pd_pair,
                                      visualization=vis))

    robot_dict = robot_model.CreateKinematicTree(mbs)
    q_ref = np.array(robot_data["staticJointValues"])
    link_transforms = robot_model.LinkHT(q_ref)
    return {
        "robot": robot_model,
        "robot_data": robot_data,
        "kinematic_tree": robot_dict,
        "gravity": gravity,
        "link_transforms_ref": link_transforms,
        "reference_joint_values": q_ref
    }

# --------------------------- Trajectory --------------------------------------

def joint_trajectory(time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic profile taking joint 2 from 0 to 1 rad in 10 s with zero start/end velocity/acceleration."""
    q = np.zeros(7)
    dq = np.zeros(7)
    ddq = np.zeros(7)

    duration = 10.0
    target_angle = 1.0

    if time <= 0.0:
        return q, dq, ddq

    if time >= duration:
        q[1] = target_angle
        return q, dq, ddq

    s = time / duration
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s

    q_poly = 6.0 * s5 - 15.0 * s4 + 10.0 * s3
    dq_poly = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / duration
    ddq_poly = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (duration * duration)

    q[1] = target_angle * q_poly
    dq[1] = target_angle * dq_poly
    ddq[1] = target_angle * ddq_poly

    return q, dq, ddq

# --------------------------- System assembly ---------------------------------

def main() -> None:
    fem, interface_data = build_cantilever_fem()
    wall_nodes, wall_weights = interface_data["wall"]
    tip_nodes, tip_weights = interface_data["tip"]

    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    ground = mbs.CreateGround(graphicsDataList=[CheckerBoard(point=[0.0, 0.0, -0.001], size=4.0)])

    base_offset = [1.5, 0.0, 0.0]
    robot_info = build_robot(mbs, base_offset)
    base_anchor_local = -HT2translation(robot_info["robot_data"]["linkList"][0]["preHT"])

    cms = ObjectFFRFreducedOrderInterface(fem)
    beam_color = [0.1, 0.4, 0.8, 0.9]

    beam_position = [0.0, 0.0, 0.0]
    beam_rotation = np.eye(3)

    beam = cms.AddObjectFFRFreducedOrder(mbs,
                                         positionRef=beam_position,
                                         rotationMatrixRef=beam_rotation.tolist(),
                                         initialVelocity=[0.0, 0.0, 0.0],
                                         initialAngularVelocity=[0.0, 0.0, 0.0],
                                         gravity=robot_info["gravity"],
                                         color=beam_color,
                                         massProportionalDamping=5e-3,
                                         stiffnessProportionalDamping=0.05)

    marker_wall = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],
        meshNodeNumbers=wall_nodes,
        weightingFactors=wall_weights,
        offset=[0.0, 0.0, 0.0]
    ))

    marker_tip = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],
        meshNodeNumbers=tip_nodes,
        weightingFactors=tip_weights,
        offset=[0.0, 0.0, 0.0]
    ))

    marker_ground = mbs.AddMarker(MarkerBodyRigid(
        bodyNumber=ground,
        localPosition=[0.0, 0.0, 0.0]
    ))

    mbs.AddObject(GenericJoint(
        markerNumbers=[marker_ground, marker_wall],
        constrainedAxes=[1, 1, 1, 1, 1, 1],
        rotationMarker0=np.eye(3).tolist(),
        rotationMarker1=np.eye(3).tolist(),
        visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)
    ))

    attachment_link_base = 0
    marker_robot_base = mbs.AddMarker(MarkerKinematicTreeRigid(
        objectNumber=robot_info["kinematic_tree"]["objectKinematicTree"],
        linkNumber=attachment_link_base,
        localPosition=base_anchor_local.tolist()
    ))

    mbs.AddObject(GenericJoint(
        markerNumbers=[marker_robot_base, marker_tip],
        constrainedAxes=[1, 1, 1, 1, 1, 1],
        rotationMarker0=np.eye(3).tolist(),
        rotationMarker1=np.eye(3).tolist(),
        visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)
    ))

    attachment_link_tool = 6
    marker_tool = mbs.AddMarker(MarkerKinematicTreeRigid(
        objectNumber=robot_info["kinematic_tree"]["objectKinematicTree"],
        linkNumber=attachment_link_tool,
        localPosition=[0.0, 0.0, 0.0]
    ))

    sensor_tool = mbs.AddSensor(SensorMarker(
        markerNumber=marker_tool,
        storeInternal=True,
        outputVariableType=exu.OutputVariableType.Position
    ))

    kt_node = robot_info["kinematic_tree"]["nodeGeneric"]
    kt_object = robot_info["kinematic_tree"]["objectKinematicTree"]
    q0, dq0, _ = joint_trajectory(0.0)
    mbs.SetNodeParameter(kt_node, "initialCoordinates", q0.tolist())
    mbs.SetNodeParameter(kt_node, "initialCoordinates_t", dq0.tolist())
    mbs.SetObjectParameter(kt_object, "jointPositionOffsetVector", q0.tolist())
    mbs.SetObjectParameter(kt_object, "jointVelocityOffsetVector", dq0.tolist())

    def prestep(mbs_local: exu.MainSystem, time: float) -> bool:
        q_ref, dq_ref, _ = joint_trajectory(time)
        mbs_local.SetObjectParameter(kt_object, "jointPositionOffsetVector", q_ref.tolist())
        mbs_local.SetObjectParameter(kt_object, "jointVelocityOffsetVector", dq_ref.tolist())
        return True

    sc.visualizationSettings.general.drawWorldBasis = True
    sc.visualizationSettings.connectors.showJointAxes = True
    sc.visualizationSettings.connectors.jointAxesLength = 0.05
    sc.visualizationSettings.connectors.jointAxesRadius = 0.004
    sc.visualizationSettings.window.renderWindowSize = [1600, 900]
    sc.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
    sc.visualizationSettings.contour.outputVariableComponent = -1

    mbs.Assemble()

    mbs.SetPreStepUserFunction(prestep)

    static_settings = exu.SimulationSettings()
    static_settings.staticSolver.newton.useModifiedNewton = True
    static_settings.staticSolver.newton.relativeTolerance = 1e-6
    static_settings.staticSolver.newton.maxIterations = 20
    static_settings.staticSolver.verboseMode = 0
    try:
        mbs.SolveStatic(static_settings)
    except Exception as exc:
        exu.Print(f"WARNING: static solve failed: {exc}")

    sim_settings = exu.SimulationSettings()
    end_time = 10.0
    step = 0.0005
    sim_settings.timeIntegration.numberOfSteps = int(end_time / step)
    sim_settings.timeIntegration.endTime = end_time
    sim_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
    sim_settings.solutionSettings.solutionWritePeriod = 0.01
    sim_settings.solutionSettings.sensorsWritePeriod = 0.01
    sim_settings.timeIntegration.newton.useModifiedNewton = True
    sim_settings.timeIntegration.newton.maxIterations = 15
    sim_settings.timeIntegration.newton.relativeTolerance = 1e-6
    sim_settings.timeIntegration.generalizedAlpha.spectralRadius = 0.9
    sim_settings.timeIntegration.verboseMode = 1

    gui_available = False
    if gui_available:
        sc.renderer.Start()
        if hasattr(sc.renderer, 'ZoomAll'):
            sc.renderer.ZoomAll()

    mbs.SolveDynamic(sim_settings, solverType=exu.DynamicSolverType.GeneralizedAlpha)

    data = np.array(mbs.GetSensorStoredData(sensor_tool))
    if data.size > 0:
        times = data[:, 0]
        positions = data[:, 1:4]
        output_dir = PROJECT_ROOT / "iiwa_ffrf_beam" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 3.5))
        labels = ["x", "y", "z"]
        for idx, label in enumerate(labels):
            plt.plot(times, positions[:, idx], label=label)
        plt.xlabel("time [s]")
        plt.ylabel("end-effector position [m]")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(output_dir / "end_effector_position_vs_time.png", dpi=200)
        plt.close()

        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], lw=1.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("End-effector trajectory")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "end_effector_trajectory_3d.png", dpi=200)
        exu.Print(f'end-effector final position [m]: {positions[-1]}')
        plt.close(fig)

    if gui_available:
        sc.renderer.Stop()


if __name__ == "__main__":
    main()
