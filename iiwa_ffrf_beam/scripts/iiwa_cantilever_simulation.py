"""完整流程：通过NGsolve构建悬臂梁，将其耦合到iiwa-14机器人上，
使用Exudyn在10秒内驱动关节2从0弧度运动到1弧度。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import exudyn as exu
from exudyn.FEM import FEMinterface, HCBstaticModeSelection, ObjectFFRFreducedOrderInterface
from exudyn.graphics import CheckerBoard
from exudyn.itemInterface import GenericJoint, MarkerBodyRigid, MarkerSuperElementRigid, VGenericJoint, SensorMarker
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink
from exudyn.robotics.utilities import GetURDFrobotData, LoadURDFrobot, HT2rotationMatrix, HT2translation
from exudyn.utilities import HTtranslate
from exudyn.rigidBodyUtilities import InertiaCuboid

# 梁的几何参数
BEAM_LENGTH = 2.0    # 梁长度 (m)
BEAM_WIDTH = 0.05    # 梁宽度 (m)
BEAM_HEIGHT = 0.05   # 梁高度 (m)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_FILE = PROJECT_ROOT / "iiwa_description" / "urdf" / "iiwa14.urdf"  # 机器人URDF文件路径
URDF_BASE_PATH = PROJECT_ROOT / "iiwa_description"

# --------------------------- FEM 准备工作 ---------------------------------

def build_cantilever_fem() -> Tuple[FEMinterface, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """使用NGsolve创建矩形悬臂梁并计算CMS数据"""
    fem = FEMinterface()

    import ngsolve as ngs
    from netgen.occ import Box, OCCGeometry

    # 几何参数
    length = 2          # 梁长 (m) —— 将沿 +Z 方向
    width = 0.05        # 截面在 X 方向的宽度 (m)
    height = 0.05       # 截面在 Y 方向的高度 (m)
    mesh_order = 2      # 网格阶次
    mesh_size = width * 0.2  # 网格大小（保持与原代码一致的细化程度）

    # 创建长方体几何和网格
    ''' 
        新的几何约定：
        - 梁长度沿 +Z 方向：Z 从 0 到 length（例如 2 米）
        - 截面为 XY 平面：
          X 从 -width/2 到 +width/2（宽度 0.05m，关于 X=0 对称）
          Y 从 -height/2 到 +height/2（高度 0.05m，关于 Y=0 对称）
    '''
    box = Box((-0.5 * width, -0.5 * height, 0.0), (0.5 * width, 0.5 * height, length))
    #    - 将长方体转换为OpenCASCADE几何对象
    # OpenCASCADE是一个专业的3D几何建模内核
    geometry = OCCGeometry(box)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=mesh_size))

    # 材料属性
    density = 7800.0    # 密度 (kg/m³)
    youngs = 2.1e9     # 杨氏模量 (Pa)
    poisson = 0.3       # 泊松比

    # 导入网格到FEM
    fem.ImportMeshFromNGsolve(mesh,
                              density=density,
                              youngsModulus=youngs,
                              poissonsRatio=poisson,
                              meshOrder=mesh_order)

    # 获取固定端（墙面）节点：Z=0 平面
    wall_nodes = fem.GetNodesInPlane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    if not wall_nodes:
        raise RuntimeError("在悬臂梁固定端平面上未找到节点")
    wall_weights = np.ones(len(wall_nodes)) / len(wall_nodes)  # 均匀权重
    exu.Print(f"悬臂梁固定端接口使用 {len(wall_nodes)} 个节点")

    # 获取自由端（梁端）节点：Z=length 平面
    tip_nodes = fem.GetNodesInPlane([0.0, 0.0, length], [0.0, 0.0, 1.0])
    if not tip_nodes:
        raise RuntimeError("在悬臂梁自由端平面上未找到节点")
    tip_weights = np.ones(len(tip_nodes)) / len(tip_nodes)  # 均匀权重

    # 获取中间截面（用于连接机器人基座）：Z=0.75*length 平面（此处即 1.5m）
    mid_z = 0.75 * length
    mid_nodes = fem.GetNodesInPlane([0.0, 0.0, mid_z], [0.0, 0.0, 1.0])
    if not mid_nodes:
        raise RuntimeError("在梁的 z=0.75*length 平面上未找到节点")
    mid_weights = np.ones(len(mid_nodes)) / len(mid_nodes)

    # 计算Hurty-Craig-Bampton模态
    n_modes = 8  # 模态数量
    exu.Print(f"正在计算 {n_modes} 个 Hurty-Craig-Bampton 模态...")
    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[wall_nodes, tip_nodes],  # 边界节点列表
        nEigenModes=n_modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2
    )

    # 返回接口数据
    interface_data = {
        "wall": (np.array(wall_nodes, dtype=int), wall_weights),  # 固定端接口 z=0
        "tip": (np.array(tip_nodes, dtype=int), tip_weights),     # 自由端接口 z=length
        "mid": (np.array(mid_nodes, dtype=int), mid_weights),     # 中间接口 z=0.75*length
    }

    return fem, interface_data

# --------------------------- 机器人装配 ----------------------------------

def build_robot(mbs: exu.MainSystem, base_marker: int, base_pose_HT: np.ndarray, pd_scale: float = 1.0) -> Dict:
    """构建机器人（冗余坐标MBS）并加载到多体系统中
    - base_marker: 机器人基座要附着的刚体标记（通常是基座刚体/地面带旋转的刚体）
    - base_pose_HT: 机器人基座希望的初始位姿（4x4 HT，影响刚体生成初始放置）
    - pd_scale: 关节PD增益缩放（例如 0.001 表示减小1000倍）
    """
    try:
        # 加载URDF机器人文件
        load_result = LoadURDFrobot(str(URDF_FILE.resolve()), str(URDF_BASE_PATH.resolve()))
    except ImportError as exc:
        raise ImportError("需要安装 Robotics Toolbox for Python (pip install roboticstoolbox-python)") from exc

    robot = load_result["robot"]
    urdf = load_result["urdf"]
    robot_data = GetURDFrobotData(robot, urdf, verbose=1)

    gravity = [0.0, 0.0, 0]  # 重力加速度

    # 创建机器人模型（基座含旋转+平移）；冗余坐标方式会使用 base.HT
    robot_model = Robot(gravity=gravity,
                        base=RobotBase(HT=base_pose_HT.tolist()),
                        tool=RobotTool(),
                        referenceConfiguration=list(robot_data["staticJointValues"]))

    # 建立连杆编号映射
    link_numbers = {"None": -1, "base_link": -1}
    for idx, link in enumerate(robot_data["linkList"]):
        link_numbers[link["name"]] = idx

    # 设置PD控制参数（支持整体缩放）
    number_of_joints = robot_data["numberOfJoints"]
    # P控制增益（比例控制参数）
    p_control = np.concatenate([
        np.array([2e4, 1.5e4, 1.5e4, 8e3, 8e3, 4e3, 4e3]) * pd_scale,
        2e3 * np.ones(max(0, number_of_joints - 7))
    ])
    # D控制增益（微分控制参数）
    d_control = np.concatenate([
        np.array([1500.0, 1200.0, 1200.0, 600.0, 600.0, 300.0, 300.0]) * pd_scale,
        150.0 * np.ones(max(0, number_of_joints - 7))
    ])

    # 添加机器人连杆
    for link in robot_data["linkList"]:
        graphics = link.get("graphicsDataList", [])
        vis = VRobotLink(graphicsData=graphics, showMBSjoint=False)
        parent_name = link.get("parentName", "None")
        parent_index = link_numbers.get(parent_name, -1)
        joint_number = link.get("jointNumber", None)
        # 设置该关节的PD控制参数
        if joint_number is not None and joint_number < len(p_control):
            pd_pair = (float(p_control[joint_number]), float(d_control[joint_number]))
        else:
            pd_pair = (0.0, 0.0)
        # 添加连杆到机器人模型
        robot_model.AddLink(RobotLink(mass=link["mass"],
                                      parent=parent_index,
                                      COM=link["com"],               # 质心位置
                                      inertia=link["inertiaCOM"],    # 惯性矩阵
                                      preHT=link["preHT"],           # 齐次变换矩阵
                                      jointType=link["jointType"],   # 关节类型
                                      PDcontrol=pd_pair,             # PD控制参数
                                      visualization=vis))

    # 以冗余坐标形式写入多体系统
    dRCM = robot_model.CreateRedundantCoordinateMBS(mbs, baseMarker=base_marker, rotationMarkerBase=None)
    q_ref = np.array(robot_data["staticJointValues"])        # 参考关节角度
    link_transforms = robot_model.LinkHT(q_ref)               # 连杆变换矩阵

    return {
        "robot": robot_model,
        "robot_data": robot_data,
        "redundant_mbs": dRCM,
        "gravity": gravity,
        "link_transforms_ref": link_transforms,
        "reference_joint_values": q_ref
    }

# --------------------------- 轨迹规划 --------------------------------------

def joint_trajectory(time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """五次多项式轨迹：在10秒内将关节2从0弧度运动到1弧度，起始和结束时速度和加速度均为零"""
    q = np.zeros(7)    # 关节位置
    dq = np.zeros(7)   # 关节速度
    ddq = np.zeros(7)  # 关节加速度

    duration = 10.0    # 运动持续时间 (s)
    target_angle = 1.0 # 目标角度 (rad)

    if time <= 0.0:
        return q, dq, ddq

    if time >= duration:
        q[1] = target_angle
        return q, dq, ddq

    # 归一化时间参数
    s = time / duration
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s

    # 五次多项式轨迹规划
    q_poly = 6.0 * s5 - 15.0 * s4 + 10.0 * s3                          # 位置多项式
    dq_poly = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / duration           # 速度多项式
    ddq_poly = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (duration * duration)  # 加速度多项式

    # 设置关节2的运动轨迹
    q[1] = target_angle * q_poly
    dq[1] = target_angle * dq_poly
    ddq[1] = target_angle * ddq_poly

    return q, dq, ddq

# --------------------------- 系统装配 ---------------------------------

def main() -> None:
    """主函数：构建完整的机器人-柔性梁耦合系统并进行仿真"""
    # 构建悬臂梁FEM模型
    fem, interface_data = build_cantilever_fem()
    wall_nodes, wall_weights = interface_data["wall"]   # 固定端节点信息 (z=0)
    tip_nodes, tip_weights = interface_data["tip"]      # 自由端节点信息 (z=length)
    mid_nodes, mid_weights = interface_data["mid"]      # 中间截面节点信息 (z=0.75*length=1.5)

    # 创建系统容器和多体系统
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    # 创建地面（带棋盘格图案）
    ground = mbs.CreateGround(graphicsDataList=[CheckerBoard(point=[0.0, 0.0, -0.001], size=4.0)])

    # 基座目标姿态 R（绕Y轴+90度）和位置 p=[0,0,1.5]
    R_base_init = np.array([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0]], dtype=float)
    p_base = np.array([0.0, 0.0, 1.5])
    HT_base = np.eye(4)
    HT_base[:3, :3] = R_base_init
    HT_base[:3, 3] = p_base

    # 创建“基座刚体”（薄板）作为机器人和梁的连接体
    inertia_base = InertiaCuboid(density=500.0, sideLengths=[0.2, 0.2, 0.02])
    base_plate = mbs.CreateRigidBody(referencePosition=p_base.tolist(),
                                     referenceRotationMatrix=R_base_init.tolist(),
                                     inertia=inertia_base,
                                     gravity=[0.0, 0.0, -9.81],
                                     returnDict=True)
    marker_base_plate = mbs.AddMarker(MarkerBodyRigid(bodyNumber=base_plate['bodyNumber'], localPosition=[0.0, 0.0, 0.0]))

    # 构建机器人并附着到基座刚体（冗余坐标MBS）。PD整体缩小（例如 1e-3）。
    robot_info = build_robot(mbs, base_marker=marker_base_plate, base_pose_HT=HT_base, pd_scale=1e-3)

    # 创建柔性体缩减模型接口
    cms = ObjectFFRFreducedOrderInterface(fem)
    beam_color = [0.1, 0.4, 0.8, 0.9]  # 梁的显示颜色 (RGBA)

    # 设置梁的初始位置和方向
    beam_position = [0.0, 0.0, 0.0]    # 梁的初始位置
    beam_rotation = np.eye(3)          # 梁的初始旋转矩阵（单位矩阵）

    # 添加柔性梁到多体系统
    beam = cms.AddObjectFFRFreducedOrder(mbs,
                                         positionRef=beam_position,
                                         rotationMatrixRef=beam_rotation.tolist(),
                                         initialVelocity=[0.0, 0.0, 0.0],           # 初始平移速度
                                         initialAngularVelocity=[0.0, 0.0, 0.0],    # 初始角速度
                                         gravity=robot_info["gravity"],              # 重力
                                         color=beam_color,                          # 显示颜色
                                         massProportionalDamping=2e-2,              # 质量比例阻尼（增大以提高数值稳定性）
                                         stiffnessProportionalDamping=0.1)          # 刚度比例阻尼（增大以提高数值稳定性）

    # 创建梁固定端标记点（用于约束到地面）
    marker_wall = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],   # 柔性体编号
        meshNodeNumbers=wall_nodes,             # 网格节点编号
        weightingFactors=wall_weights,          # 权重因子
        offset=[0.0, 0.0, 0.0]                 # 偏移量
    ))

    # 创建梁自由端标记点（用于输出/检查，不再用于与机器人连接）
    marker_tip = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],   # 柔性体编号
        meshNodeNumbers=tip_nodes,              # 网格节点编号
        weightingFactors=tip_weights,           # 权重因子
        offset=[0.0, 0.0, 0.0]                 # 偏移量
    ))

    # 创建梁中截面标记点（z=1.5m）：用于与机器人基座刚性连接
    marker_mid = mbs.AddMarker(MarkerSuperElementRigid(
        bodyNumber=beam["oFFRFreducedOrder"],
        meshNodeNumbers=mid_nodes,
        weightingFactors=mid_weights,
        offset=[0.0, 0.0, 0.0]
    ))

    #（装配前不做校准；装配后统一校准，见后续）

    # 创建地面标记点
    marker_ground = mbs.AddMarker(MarkerBodyRigid(
        bodyNumber=ground,                      # 地面物体编号
        localPosition=[0.0, 0.0, 0.0]          # 本地位置
    ))

    # 创建固定约束：将梁的固定端约束到地面（完全固定）
    mbs.AddObject(GenericJoint(
        markerNumbers=[marker_ground, marker_wall],      # 连接地面和梁固定端标记点
        constrainedAxes=[1, 1, 1, 1, 1, 1],             # 约束所有6个自由度（3平移+3旋转）
        rotationMarker0=np.eye(3).tolist(),             # 标记点0的旋转矩阵
        rotationMarker1=np.eye(3).tolist(),             # 标记点1的旋转矩阵
        visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)  # 可视化参数
    ))

    # 创建刚性连接：将梁的中截面 (z=1.5m) 连接到“基座刚体”
    mbs.AddObject(GenericJoint(
        markerNumbers=[marker_base_plate, marker_mid],   # 连接机器人基座刚体和梁中截面
        constrainedAxes=[1, 1, 1, 1, 1, 1],             # 约束所有6个自由度（刚性连接）
        rotationMarker0=np.eye(3).tolist(),             # 基座侧
        rotationMarker1=np.eye(3).tolist(),             # 梁侧
        visualization=VGenericJoint(axesRadius=0.02, axesLength=0.1)  # 可视化参数
    ))

    # 创建末端执行器标记点（冗余坐标MBS下，取最后一个刚体）和传感器
    end_body = robot_info["redundant_mbs"]["bodyList"][-1]
    marker_tool = mbs.AddMarker(MarkerBodyRigid(bodyNumber=end_body, localPosition=[0.0, 0.0, 0.0]))

    # 创建位置传感器，用于记录末端执行器轨迹
    sensor_tool = mbs.AddSensor(SensorMarker(
        markerNumber=marker_tool,                      # 要监测的标记点
        storeInternal=True,                            # 存储传感器数据
        outputVariableType=exu.OutputVariableType.Position  # 输出变量类型：位置
    ))

    # 设置机器人初始状态和轨迹跟踪
    # 冗余坐标MBS暂不进行轨迹跟踪/PD偏置更新（需要扭矩加载路径），此处预步为空
    def prestep(mbs_local: exu.MainSystem, time: float) -> bool:
        return True

    # 设置可视化参数
    sc.visualizationSettings.general.drawWorldBasis = True              # 显示世界坐标系
    sc.visualizationSettings.connectors.showJointAxes = True            # 显示关节轴
    sc.visualizationSettings.connectors.jointAxesLength = 0.05          # 关节轴长度
    sc.visualizationSettings.connectors.jointAxesRadius = 0.004         # 关节轴半径
    sc.visualizationSettings.window.renderWindowSize = [1600, 900]      # 渲染窗口大小
    sc.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal  # 等高线输出变量
    sc.visualizationSettings.contour.outputVariableComponent = -1       # 输出变量分量（-1表示所有分量）

    # 装配多体系统
    mbs.Assemble()

    # 在装配后对梁的标记进行一次几何校准，使中截面、自由端精确落在目标位置
    try:
        tip_target = np.array([0.0, 0.0, BEAM_LENGTH])
        tip_pos0 = mbs.GetMarkerOutput(marker_tip, exu.OutputVariableType.Position,
                                       configuration=exu.ConfigurationType.Initial)
        delta_tip = tip_target - np.array(tip_pos0, dtype=float)
        if np.linalg.norm(delta_tip) > 1e-12:
            mbs.SetMarkerParameter(marker_tip, "offset", delta_tip.tolist())
    except Exception as exc:
        exu.Print(f"警告: 梁自由端装配后校准失败: {exc}")

    try:
        mid_target = np.array([0.0, 0.0, 0.75 * BEAM_LENGTH])
        mid_pos0 = mbs.GetMarkerOutput(marker_mid, exu.OutputVariableType.Position,
                                       configuration=exu.ConfigurationType.Initial)
        delta_mid = mid_target - np.array(mid_pos0, dtype=float)
        if np.linalg.norm(delta_mid) > 1e-12:
            mbs.SetMarkerParameter(marker_mid, "offset", delta_mid.tolist())
    except Exception as exc:
        exu.Print(f"警告: 梁中截面装配后校准失败: {exc}")

    # 输出目录（用于截图和曲线）
    output_dir = PROJECT_ROOT / "iiwa_ffrf_beam" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 简单的截图函数（使用 OpenGL 渲染器保存图像）
    def save_scene_screenshot(file_stem: str) -> None:
        try:
            sc.visualizationSettings.exportImages.saveImageFileName = str(output_dir / file_stem)
            sc.visualizationSettings.exportImages.saveImageFormat = "PNG"
            sc.visualizationSettings.exportImages.saveImageSingleFile = True
            # 启动渲染器（如未启动）并截图
            if not sc.renderer.IsActive():
                sc.renderer.Start()
            if hasattr(sc.renderer, 'ZoomAll'):
                sc.renderer.ZoomAll()
            # 设定视角：XZ 平面（通过将模型绕 X 轴 +90° 旋转）
            try:
                rs = sc.renderer.GetState()
                # 绕 X 轴 +90 度的旋转矩阵
                R_x90 = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 0.0, -1.0],
                                  [0.0, 1.0, 0.0]], dtype=float)
                rs['modelRotation'] = R_x90.tolist()
                sc.renderer.SetState(rs, waitForRendererFullStartup=False)
                sc.renderer.SendRedrawSignal()
            except Exception as exc:
                exu.Print(f"警告: 设置XZ视角失败: {exc}")
            sc.renderer.RedrawAndSaveImage()
        except Exception as exc:
            exu.Print(f"警告: 截图 '{file_stem}' 失败: {exc}")
        finally:
            # 关闭渲染器，避免阻塞
            if sc.renderer.IsActive():
                sc.renderer.Stop()

    # 在装配后保存场景初始截图（包含梁与机械臂的形状）
    save_scene_screenshot("scene_initial")

    # 设置预步函数（用于轨迹跟踪）
    mbs.SetPreStepUserFunction(prestep)

    # 静力学求解设置
    static_settings = exu.SimulationSettings()
    static_settings.staticSolver.newton.useModifiedNewton = True        # 使用修改的牛顿法
    static_settings.staticSolver.newton.relativeTolerance = 1e-6        # 相对收敛容差
    static_settings.staticSolver.newton.maxIterations = 20              # 最大迭代次数
    static_settings.staticSolver.verboseMode = 0                        # 详细输出模式

    # 静力学：分步自适应，以提高收敛性
    static_settings.staticSolver.numberOfLoadSteps = 20
    static_settings.staticSolver.adaptiveStep = True
    static_settings.linearSolverType = exu.LinearSolverType.EigenDense
    static_settings.linearSolverSettings.ignoreSingularJacobian = True
    static_settings.staticSolver.newton.absoluteTolerance = 1e-5
    try:
        # 进行静力学求解（找到系统的初始平衡状态）
        mbs.SolveStatic(static_settings, showHints=True)
    except Exception as exc:
        exu.Print(f"警告: 静力学求解失败: {exc}")
    finally:
        pass

    # 在静力学平衡后再次截图（若想看到平衡后的形状）
    save_scene_screenshot("scene_after_static")

    # ==== 输出仿真开始（t=0）时末端执行器的 SE(3) 位姿 ====
    # 说明：Initial 配置对应装配后的初始状态（未进行静力平衡迭代），
    #       Current 配置在静力学求解成功时等于平衡状态（作为动力学起点）。
    def _print_se3(p_vec, R_mat, title_prefix=""):
        p = np.array(p_vec, dtype=float).reshape(3)
        R = np.array(R_mat, dtype=float).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        exu.Print(f"{title_prefix}平移 p [m] = {np.array2string(p, precision=6)}")
        exu.Print(f"{title_prefix}旋转矩阵 R =\n{np.array2string(R, precision=6)}")
        exu.Print(f"{title_prefix}齐次变换 T =\n{np.array2string(T, precision=6)}")

    try:
        p_init = mbs.GetMarkerOutput(marker_tool, exu.OutputVariableType.Position,
                                     configuration=exu.ConfigurationType.Initial)
        R_init = mbs.GetMarkerOutput(marker_tool, exu.OutputVariableType.RotationMatrix,
                                     configuration=exu.ConfigurationType.Initial)
        _print_se3(p_init, R_init, title_prefix="初始配置：")
    except Exception as exc:
        exu.Print(f"警告: 获取初始配置 SE(3) 失败: {exc}")

    try:
        p_start = mbs.GetMarkerOutput(marker_tool, exu.OutputVariableType.Position,
                                      configuration=exu.ConfigurationType.Current)
        R_start = mbs.GetMarkerOutput(marker_tool, exu.OutputVariableType.RotationMatrix,
                                      configuration=exu.ConfigurationType.Current)
        _print_se3(p_start, R_start, title_prefix="仿真开始：")
    except Exception as exc:
        exu.Print(f"警告: 获取仿真开始 SE(3) 失败: {exc}")

    # ==== 输出梁自由端（tip）的初始 SE(3) 位姿 ====
    try:
        p_tip_init = mbs.GetMarkerOutput(marker_tip, exu.OutputVariableType.Position,
                                         configuration=exu.ConfigurationType.Initial)
        R_tip_init = mbs.GetMarkerOutput(marker_tip, exu.OutputVariableType.RotationMatrix,
                                         configuration=exu.ConfigurationType.Initial)
        _print_se3(p_tip_init, R_tip_init, title_prefix="梁自由端 初始配置：")
    except Exception as exc:
        exu.Print(f"警告: 获取梁自由端初始 SE(3) 失败: {exc}")

    # ==== 输出机械臂基座刚体（锚点）的初始位姿（SE(3)）====
    try:
        p_base_init = mbs.GetMarkerOutput(marker_base_plate, exu.OutputVariableType.Position,
                                          configuration=exu.ConfigurationType.Initial)
        R_base_init_out = mbs.GetMarkerOutput(marker_base_plate, exu.OutputVariableType.RotationMatrix,
                                          configuration=exu.ConfigurationType.Initial)
        _print_se3(p_base_init, R_base_init_out, title_prefix="机器人基座锚点 初始配置：")
    except Exception as exc:
        exu.Print(f"警告: 获取机器人基座初始 SE(3) 失败: {exc}")

    # 同时输出仿真开始（Current=静力学后）的基座位姿，便于核对
    try:
        p_base_cur = mbs.GetMarkerOutput(marker_base_plate, exu.OutputVariableType.Position,
                                         configuration=exu.ConfigurationType.Current)
        R_base_cur = mbs.GetMarkerOutput(marker_base_plate, exu.OutputVariableType.RotationMatrix,
                                         configuration=exu.ConfigurationType.Current)
        _print_se3(p_base_cur, R_base_cur, title_prefix="机器人基座锚点 仿真开始：")
    except Exception as exc:
        exu.Print(f"警告: 获取机器人基座仿真开始 SE(3) 失败: {exc}")

    # 额外输出：机器人基座“原点”的位姿（由 base_offset/R_base_init 给定，与锚点不同）
    try:
        p_base_origin = np.array(p_base, dtype=float)
        R_base_origin = np.array(R_base_init, dtype=float)
        _print_se3(p_base_origin, R_base_origin, title_prefix="机器人基座原点 初始配置：")
    except Exception as exc:
        exu.Print(f"警告: 获取机器人基座原点位姿失败: {exc}")

    # 动力学仿真设置
    sim_settings = exu.SimulationSettings()
    end_time = 10.0   # 仿真结束时间 (s)
    step = 0.0002     # 时间步长 (s)（减小步长提升稳定性）
    sim_settings.timeIntegration.numberOfSteps = int(end_time / step)   # 仿真步数
    sim_settings.timeIntegration.endTime = end_time                     # 结束时间
    sim_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True  # 计算初始加速度
    # 使用速度级约束需要 Newmark；开启 Newmark 并使用 index-2 约束
    sim_settings.timeIntegration.generalizedAlpha.useNewmark = True
    sim_settings.timeIntegration.generalizedAlpha.useIndex2Constraints = True
    sim_settings.linearSolverType = exu.LinearSolverType.EigenDense                  # 稠密线性求解
    sim_settings.linearSolverSettings.ignoreSingularJacobian = True                  # 冗余约束时放宽
    sim_settings.timeIntegration.newton.absoluteTolerance = 1e-5                     # 绝对公差放宽
    sim_settings.solutionSettings.solutionWritePeriod = 0.01            # 解输出周期
    sim_settings.solutionSettings.sensorsWritePeriod = 0.01             # 传感器输出周期
    sim_settings.timeIntegration.newton.useModifiedNewton = True        # 使用修改的牛顿法
    sim_settings.timeIntegration.newton.maxIterations = 40              # 最大迭代次数（增大提高鲁棒性）
    sim_settings.timeIntegration.newton.relativeTolerance = 1e-7        # 相对收敛容差（更严格）
    sim_settings.timeIntegration.generalizedAlpha.spectralRadius = 0.7  # 谱半径（更多数值阻尼）
    sim_settings.timeIntegration.verboseMode = 1                        # 详细输出模式

    # GUI设置（如果可用）
    gui_available = False  # 设置为True可启用图形界面
    if gui_available:
        sc.renderer.Start()
        if hasattr(sc.renderer, 'ZoomAll'):
            sc.renderer.ZoomAll()

    # 执行动力学仿真
    mbs.SolveDynamic(sim_settings, solverType=exu.DynamicSolverType.GeneralizedAlpha, showHints=True)

    # 处理传感器数据并绘制结果
    data = np.array(mbs.GetSensorStoredData(sensor_tool))
    if data.size > 0:
        times = data[:, 0]          # 时间数据
        positions = data[:, 1:4]    # 位置数据 (x, y, z)
        # 创建输出目录
        output_dir = PROJECT_ROOT / "iiwa_ffrf_beam" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 绘制末端执行器位置随时间变化的曲线
        plt.figure(figsize=(6, 3.5))
        labels = ["x", "y", "z"]
        for idx, label in enumerate(labels):
            plt.plot(times, positions[:, idx], label=label)
        plt.xlabel("时间 [s]")
        plt.ylabel("末端执行器位置 [m]")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(output_dir / "end_effector_position_vs_time.png", dpi=200)
        plt.close()

        # 绘制末端执行器3D轨迹
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], lw=1.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("末端执行器轨迹")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "end_effector_trajectory_3d.png", dpi=200)
        exu.Print(f'末端执行器最终位置 [m]: {positions[-1]}')
        plt.close(fig)

    # 停止图形界面（如果启用）
    if gui_available:
        sc.renderer.Stop()


if __name__ == "__main__":
    main()
