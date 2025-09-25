"""仅可视化：构建柔性梁 + 冗余坐标机器人 + 基座刚体的场景，打开 GUI，不做静/动力学仿真"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import exudyn as exu
from exudyn.FEM import FEMinterface, HCBstaticModeSelection, ObjectFFRFreducedOrderInterface
from exudyn.graphics import CheckerBoard
from exudyn.itemInterface import GenericJoint, MarkerBodyRigid, MarkerSuperElementRigid, VGenericJoint
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink
from exudyn.robotics.utilities import GetURDFrobotData, LoadURDFrobot, HT2rotationMatrix, HT2translation
from exudyn.utilities import HTtranslate
from exudyn.rigidBodyUtilities import InertiaCuboid

# 基本参数
BEAM_LENGTH = 2.0
BEAM_WIDTH = 0.05
BEAM_HEIGHT = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_FILE = PROJECT_ROOT / "iiwa_description" / "urdf" / "iiwa14.urdf"
URDF_BASE_PATH = PROJECT_ROOT / "iiwa_description"


def build_cantilever_fem() -> Tuple[FEMinterface, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """使用 NGsolve 创建矩形悬臂梁并计算 CMS 数据（长度沿 +Z）"""
    fem = FEMinterface()

    import ngsolve as ngs
    from netgen.occ import Box, OCCGeometry

    length = BEAM_LENGTH
    width = BEAM_WIDTH
    height = BEAM_HEIGHT
    mesh_order = 2
    mesh_size = width * 0.2

    box = Box((-0.5 * width, -0.5 * height, 0.0), (0.5 * width, 0.5 * height, length))
    geometry = OCCGeometry(box)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=mesh_size))

    density = 7800.0
    youngs = 2.1e9
    poisson = 0.3

    fem.ImportMeshFromNGsolve(mesh,
                              density=density,
                              youngsModulus=youngs,
                              poissonsRatio=poisson,
                              meshOrder=mesh_order)

    # 接口节点：z=0 固定端、z=length 自由端、z=0.75*length 中截面
    wall_nodes = fem.GetNodesInPlane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    tip_nodes = fem.GetNodesInPlane([0.0, 0.0, length], [0.0, 0.0, 1.0])
    mid_z = 0.75 * length
    mid_nodes = fem.GetNodesInPlane([0.0, 0.0, mid_z], [0.0, 0.0, 1.0])
    if not wall_nodes or not tip_nodes or not mid_nodes:
        raise RuntimeError("FEM 接口节点未找到：请检查网格/几何")

    wall_weights = np.ones(len(wall_nodes)) / len(wall_nodes)
    tip_weights = np.ones(len(tip_nodes)) / len(tip_nodes)
    mid_weights = np.ones(len(mid_nodes)) / len(mid_nodes)

    n_modes = 8
    exu.Print(f"正在计算 {n_modes} 个 Hurty-Craig-Bampton 模态...")
    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[wall_nodes, tip_nodes],
        nEigenModes=n_modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2
    )

    interface_data = {
        "wall": (np.array(wall_nodes, dtype=int), wall_weights),
        "tip": (np.array(tip_nodes, dtype=int), tip_weights),
        "mid": (np.array(mid_nodes, dtype=int), mid_weights),
    }
    return fem, interface_data


def build_robot(mbs: exu.MainSystem, base_marker: int, base_pose_HT: np.ndarray, pd_scale: float = 1e-3) -> Dict:
    """将 URDF 机器人以冗余坐标形式加入 MBS，附着到 base_marker；base_pose_HT 用于初始放置"""
    load_result = LoadURDFrobot(str(URDF_FILE.resolve()), str(URDF_BASE_PATH.resolve()))
    robot = load_result["robot"]
    urdf = load_result["urdf"]
    robot_data = GetURDFrobotData(robot, urdf, verbose=0)

    gravity = [0.0, 0.0, -9.81]

    robot_model = Robot(gravity=gravity,
                        base=RobotBase(HT=base_pose_HT.tolist()),
                        tool=RobotTool(),
                        referenceConfiguration=list(robot_data["staticJointValues"]))

    # 连杆编号映射
    link_numbers = {"None": -1, "base_link": -1}
    for idx, link in enumerate(robot_data["linkList"]):
        link_numbers[link["name"]] = idx

    # PD（统一缩放）
    number_of_joints = robot_data["numberOfJoints"]
    p_control = np.concatenate([
        np.array([2e4, 1.5e4, 1.5e4, 8e3, 8e3, 4e3, 4e3]) * pd_scale,
        2e3 * np.ones(max(0, number_of_joints - 7))
    ])
    d_control = np.concatenate([
        np.array([1500.0, 1200.0, 1200.0, 600.0, 600.0, 300.0, 300.0]) * pd_scale,
        150.0 * np.ones(max(0, number_of_joints - 7))
    ])

    # 添加连杆
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

    # 以冗余坐标加入系统
    dRCM = robot_model.CreateRedundantCoordinateMBS(mbs, baseMarker=base_marker, rotationMarkerBase=None)
    q_ref = np.array(robot_data["staticJointValues"])        # 参考关节角度
    link_transforms = robot_model.LinkHT(q_ref)
    return {
        "robot": robot_model,
        "robot_data": robot_data,
        "redundant_mbs": dRCM,
        "gravity": gravity,
        "link_transforms_ref": link_transforms,
        "reference_joint_values": q_ref,
    }


def main() -> None:
    fem, interface_data = build_cantilever_fem()
    wall_nodes, wall_weights = interface_data["wall"]
    tip_nodes, tip_weights = interface_data["tip"]
    mid_nodes, mid_weights = interface_data["mid"]

    sc = exu.SystemContainer()
    mbs = sc.AddSystem()

    # 地面 + 棋盘
    ground = mbs.CreateGround(graphicsDataList=[CheckerBoard(point=[0.0, 0.0, -0.001], size=4.0)])

    # 基座刚体：p=[0,0,1.5], R=绕Y轴+90° （[[0,0,1],[0,1,0],[-1,0,0]]）
    R_base = np.array([[0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0]], dtype=float)
    p_base = np.array([0.0, 0.0, 1.5])
    HT_base = np.eye(4)
    HT_base[:3, :3] = R_base
    HT_base[:3, 3] = p_base

    inertia_base = InertiaCuboid(density=500.0, sideLengths=[0.2, 0.2, 0.02])
    base_plate = mbs.CreateRigidBody(referencePosition=p_base.tolist(),
                                     referenceRotationMatrix=R_base.tolist(),
                                     inertia=inertia_base,
                                     gravity=[0.0, 0.0, -9.81],
                                     returnDict=True)
    marker_base_plate = mbs.AddMarker(MarkerBodyRigid(bodyNumber=base_plate['bodyNumber'], localPosition=[0.0, 0.0, 0.0]))

    # 机器人（冗余坐标）
    robot_info = build_robot(mbs, base_marker=marker_base_plate, base_pose_HT=HT_base, pd_scale=1e-3)

    # 柔性梁（FFRF 缩减）
    cms = ObjectFFRFreducedOrderInterface(fem)
    beam = cms.AddObjectFFRFreducedOrder(mbs,
                                         positionRef=[0.0, 0.0, 0.0],
                                         rotationMatrixRef=np.eye(3).tolist(),
                                         initialVelocity=[0.0, 0.0, 0.0],
                                         initialAngularVelocity=[0.0, 0.0, 0.0],
                                         gravity=robot_info["gravity"],
                                         color=[0.1, 0.4, 0.8, 0.9],
                                         massProportionalDamping=2e-2,
                                         stiffnessProportionalDamping=0.1)

    # 标记
    marker_wall = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=beam["oFFRFreducedOrder"],
                                                       meshNodeNumbers=wall_nodes,
                                                       weightingFactors=wall_weights,
                                                       offset=[0.0, 0.0, 0.0]))
    marker_tip = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=beam["oFFRFreducedOrder"],
                                                      meshNodeNumbers=tip_nodes,
                                                      weightingFactors=tip_weights,
                                                      offset=[0.0, 0.0, 0.0]))
    marker_mid = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=beam["oFFRFreducedOrder"],
                                                      meshNodeNumbers=mid_nodes,
                                                      weightingFactors=mid_weights,
                                                      offset=[0.0, 0.0, 0.0]))

    # 约束：墙端 ↔ 地面 6DoF 刚接；中截面 ↔ 基座刚体 6DoF 刚接
    marker_ground = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ground, localPosition=[0.0, 0.0, 0.0]))
    mbs.AddObject(GenericJoint(markerNumbers=[marker_ground, marker_wall],
                               constrainedAxes=[1, 1, 1, 1, 1, 1],
                               rotationMarker0=np.eye(3).tolist(),
                               rotationMarker1=np.eye(3).tolist(),
                               visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)))

    mbs.AddObject(GenericJoint(markerNumbers=[marker_base_plate, marker_mid],
                               constrainedAxes=[1, 1, 1, 1, 1, 1],
                               rotationMarker0=np.eye(3).tolist(),
                               rotationMarker1=np.eye(3).tolist(),
                               visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)))

    # 可视化设置
    sc.visualizationSettings.general.drawWorldBasis = True
    sc.visualizationSettings.connectors.showJointAxes = True
    sc.visualizationSettings.connectors.jointAxesLength = 0.05
    sc.visualizationSettings.connectors.jointAxesRadius = 0.004
    sc.visualizationSettings.window.renderWindowSize = [1600, 900]

    # 装配 + 打开 GUI（不做仿真）
    mbs.Assemble()

    sc.renderer.Start()
    if hasattr(sc.renderer, 'ZoomAll'):
        sc.renderer.ZoomAll()
    # 等待用户交互；关闭窗口后返回
    sc.renderer.DoIdleTasks()  # 默认 waitSeconds=-1，直到窗口关闭
    sc.renderer.Stop()


if __name__ == "__main__":
    main()

