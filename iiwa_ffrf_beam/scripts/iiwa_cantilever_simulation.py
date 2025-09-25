"""Full pipeline: build a cantilever beam via NGsolve, couple it to the iiwa-14 robot,
and drive joint 2 from 0 to 1 rad in 10 s using Exudyn."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import exudyn as exu
from exudyn.FEM import FEMinterface, HCBstaticModeSelection, ObjectFFRFreducedOrderInterface
from exudyn.graphics import CheckerBoard
from exudyn.itemInterface import GenericJoint, MarkerKinematicTreeRigid, MarkerSuperElementRigid, VGenericJoint
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink
from exudyn.robotics.utilities import GetURDFrobotData, LoadURDFrobot, HT2rotationMatrix, HT2translation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_FILE = PROJECT_ROOT / "iiwa_description" / "urdf" / "iiwa14.urdf"
URDF_BASE_PATH = PROJECT_ROOT / "iiwa_description"

# --------------------------- FEM preparation ---------------------------------

def build_cantilever_fem() -> Tuple[FEMinterface, np.ndarray, np.ndarray]:
    """Create a rectangular cantilever beam with NGsolve and compute CMS data."""
    fem = FEMinterface()

    import ngsolve as ngs
    from netgen.occ import Box, OCCGeometry

    length = 0.6
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

    base_nodes = fem.GetNodesInPlane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    if not base_nodes:
        raise RuntimeError("No nodes found on cantilever base plane")

    weights = np.ones(len(base_nodes)) / len(base_nodes)
    exu.Print(f"Cantilever base uses {len(base_nodes)} interface nodes")

    n_modes = 12
    exu.Print(f"Computing {n_modes} Hurty-Craig-Bampton modes ...")
    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[base_nodes],
        nEigenModes=n_modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2
    )

    return fem, np.array(base_nodes, dtype=int), weights

# --------------------------- Robot assembly ----------------------------------

def build_robot(mbs: exu.MainSystem) -> Dict:
    try:
        load_result = LoadURDFrobot(str(URDF_FILE.resolve()), str(URDF_BASE_PATH.resolve()))
    except ImportError as exc:
        raise ImportError("Robotics Toolbox for Python is required (pip install roboticstoolbox-python).") from exc

    robot = load_result["robot"]
    urdf = load_result["urdf"]
    robot_data = GetURDFrobotData(robot, urdf, verbose=1)

    gravity = [0.0, 0.0, -9.81]

    robot_model = Robot(gravity=gravity,
                        base=RobotBase(),
                        tool=RobotTool(),
                        referenceConfiguration=list(robot_data["staticJointValues"]))

    link_numbers = {"None": -1, "base_link": -1}
    for idx, link in enumerate(robot_data["linkList"]):
        link_numbers[link["name"]] = idx

    for link in robot_data["linkList"]:
        graphics = link.get("graphicsDataList", [])
        vis = VRobotLink(graphicsData=graphics, showMBSjoint=False)
        parent_name = link.get("parentName", "None")
        parent_index = link_numbers.get(parent_name, -1)
        robot_model.AddLink(RobotLink(mass=link["mass"],
                                      parent=parent_index,
                                      COM=link["com"],
                                      inertia=link["inertiaCOM"],
                                      preHT=link["preHT"],
                                      jointType=link["jointType"],
                                      PDcontrol=(0.0, 0.0),
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
    fem, interface_nodes, interface_weights = build_cantilever_fem()

    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    mbs.CreateGround(graphicsDataList=[CheckerBoard(point=[0.0, 0.0, -0.001], size=4.0)])

    robot_info = build_robot(mbs)

    cms = ObjectFFRFreducedOrderInterface(fem)
    beam_color = [0.1, 0.4, 0.8, 0.9]

    attachment_link = 6
    attach_transform = robot_info["link_transforms_ref"][attachment_link]
    beam_position = HT2translation(attach_transform)
    beam_rotation = HT2rotationMatrix(attach_transform)

    beam = cms.AddObjectFFRFreducedOrder(mbs,
                                         positionRef=beam_position.tolist(),
                                         rotationMatrixRef=beam_rotation.tolist(),
                                         initialVelocity=[0.0, 0.0, 0.0],
                                         initialAngularVelocity=[0.0, 0.0, 0.0],
                                         gravity=robot_info["gravity"],
                                         color=beam_color,
                                         stiffnessProportionalDamping=0.01)

    marker_robot = mbs.AddMarker(MarkerKinematicTreeRigid(
        objectNumber=robot_info["kinematic_tree"]["objectKinematicTree"],
        linkNumber=attachment_link,
        localPosition=[0.0, 0.0, 0.0]
    ))

    marker_beam = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],
        meshNodeNumbers=interface_nodes,
        weightingFactors=interface_weights,
        offset=[0.0, 0.0, 0.0]
    ))

    mbs.AddObject(GenericJoint(markerNumbers=[marker_robot, marker_beam],
                               constrainedAxes=[1, 1, 1, 1, 1, 1],
                               visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)))

    kt_node = robot_info["kinematic_tree"]["nodeGeneric"]
    kt_object = robot_info["kinematic_tree"]["objectKinematicTree"]
    q0, dq0, _ = joint_trajectory(0.0)
    mbs.SetNodeParameter(kt_node, "initialCoordinates", q0.tolist())
    mbs.SetNodeParameter(kt_node, "initialCoordinates_t", dq0.tolist())

    def prestep(mbs_local: exu.MainSystem, time: float) -> bool:
        q_ref, dq_ref, _ = joint_trajectory(time)
        mbs_local.SetObjectParameter(kt_object, "jointPositionOffsetVector", q_ref.tolist())
        mbs_local.SetObjectParameter(kt_object, "jointVelocityOffsetVector", dq_ref.tolist())
        return True

    mbs.SetPreStepUserFunction(prestep)

    sc.visualizationSettings.general.drawWorldBasis = True
    sc.visualizationSettings.connectors.showJointAxes = True
    sc.visualizationSettings.connectors.jointAxesLength = 0.05
    sc.visualizationSettings.connectors.jointAxesRadius = 0.004
    sc.visualizationSettings.window.renderWindowSize = [1600, 900]
    sc.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
    sc.visualizationSettings.contour.outputVariableComponent = -1

    mbs.Assemble()

    sim_settings = exu.SimulationSettings()
    end_time = 10.0
    step = 0.002
    sim_settings.timeIntegration.numberOfSteps = int(end_time / step)
    sim_settings.timeIntegration.endTime = end_time
    sim_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
    sim_settings.solutionSettings.solutionWritePeriod = 0.02
    sim_settings.solutionSettings.sensorsWritePeriod = 0.02
    sim_settings.timeIntegration.verboseMode = 1

    gui_available = getattr(exu, 'GUIavailable', lambda: False)()
    if gui_available:
        sc.renderer.Start()
        sc.renderer.SetZoom3D(1.0)

    mbs.SolveDynamic(sim_settings, solverType=exu.DynamicSolverType.GeneralizedAlpha)

    if gui_available:
        sc.renderer.Stop()


if __name__ == "__main__":
    main()
