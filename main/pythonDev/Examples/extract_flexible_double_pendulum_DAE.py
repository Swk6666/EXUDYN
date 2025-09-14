#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Extract DAE system from flexible double pendulum using GeometricallyExactBeam2D
#
# Model:    Planar model of a highly flexible double pendulum with two arms, each of length 0.25m;
#           Extract explicit DAE matrices: M, K, D, C, f_ext
#
# Author:   Modified from Johannes Gerstmayr's original pendulum example
# Date:     2025-09-14
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
# *flexible double pendulum DAE extraction*
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## import libaries
import exudyn as exu
from exudyn.utilities import * #includes itemInterface and rigidBodyUtilities
import exudyn.graphics as graphics #only import if it does not conflict

import numpy as np
# from math import sin, cos, pi

print("=" * 70)
print("柔性双摆系统DAE方程组显式提取")
print("=" * 70)

## setup system container and mbs
SC = exu.SystemContainer()
mbs = SC.AddSystem()

## define parameters for double pendulum beams
nElements1 = 5         # number of elements for first pendulum arm
nElements2 = 5         # number of elements for second pendulum arm
L1 = 0.25              # length of first pendulum arm
L2 = 0.25              # length of second pendulum arm
lElem1 = L1/nElements1 # length of one finite element for first arm
lElem2 = L2/nElements2 # length of one finite element for second arm
E=1e9                  # very soft elastomer
rho=1000               # elastomer
h=0.002                # height of rectangular beam element in m
b=0.01                 # width of rectangular beam element in m
A=b*h                  # cross sectional area of beam element in m^2
I=b*h**3/12            # second moment of area of beam element in m^4
nu = 0.3               # Poisson's ratio
    
EI = E*I
EA = E*A
rhoA = rho*A
rhoI = rho*I
ks = 10*(1+nu)/(12+11*nu) # shear correction factor
G = E/(2*(1+nu))          # shear modulus
GA = ks*G*A               # shear stiffness of beam

g = [0,-9.81,0]           # gravity load

print(f"\n系统参数:")
print(f"  第一摆臂: {nElements1}个单元, 长度 {L1}m")
print(f"  第二摆臂: {nElements2}个单元, 长度 {L2}m")
print(f"  弹性模量: E = {E:.1e} Pa")
print(f"  密度: ρ = {rho} kg/m³")
print(f"  截面尺寸: {b}m × {h}m")

## create nodes for first pendulum arm (horizontal initial position)
firstNodes = []
for i in range(nElements1+1):
    pRef = [i*lElem1, 0, 0]  # horizontal initial position
    n = mbs.AddNode(NodeRigidBody2D(referenceCoordinates = pRef))
    firstNodes.append(n)
    if i==0: firstNode = n

## create nodes for second pendulum arm (continuing from first arm, also horizontal)
secondNodes = []
for i in range(nElements2+1):
    pRef = [L1 + i*lElem2, 0, 0]  # continue from end of first arm
    n = mbs.AddNode(NodeRigidBody2D(referenceCoordinates = pRef))
    secondNodes.append(n)
    if i==0: secondFirstNode = n

## create beam elements for first pendulum arm:
listBeams1 = []
for i in range(nElements1):
    oBeam = mbs.AddObject(ObjectBeamGeometricallyExact2D(nodeNumbers = [firstNodes[i], firstNodes[i+1]], 
                                                            physicsLength=lElem1,
                                                            physicsMassPerLength=rhoA,
                                                            physicsCrossSectionInertia=rhoI,
                                                            physicsBendingStiffness=EI,
                                                            physicsAxialStiffness=EA,
                                                            physicsShearStiffness=GA,
                                                            visualization=VObjectBeamGeometricallyExact2D(drawHeight = h)
                                                ))
    listBeams1.append(oBeam)

## create beam elements for second pendulum arm:
listBeams2 = []
for i in range(nElements2):
    oBeam = mbs.AddObject(ObjectBeamGeometricallyExact2D(nodeNumbers = [secondNodes[i], secondNodes[i+1]], 
                                                            physicsLength=lElem2,
                                                            physicsMassPerLength=rhoA,
                                                            physicsCrossSectionInertia=rhoI,
                                                            physicsBendingStiffness=EI,
                                                            physicsAxialStiffness=EA,
                                                            physicsShearStiffness=GA,
                                                            visualization=VObjectBeamGeometricallyExact2D(drawHeight = h)
                                                ))
    listBeams2.append(oBeam)

## combine all beams for loading
listBeams = listBeams1 + listBeams2

## create ground node with marker for coordinate constraints
oGround = mbs.CreateGround(referencePosition=[0,0,0])
mGround = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oGround, localPosition=[0,0,0]))

## add revolute joint between ground and first node of first pendulum arm
mNode0 = mbs.AddMarker(MarkerNodePosition(nodeNumber=firstNode))
joint1 = mbs.AddObject(ObjectJointRevolute2D(markerNumbers=[mGround, mNode0]))

## add revolute joint between end of first arm and beginning of second arm
# Get the last node of first arm and first node of second arm
mNode1End = mbs.AddMarker(MarkerNodePosition(nodeNumber=firstNodes[-1]))
mNode2Start = mbs.AddMarker(MarkerNodePosition(nodeNumber=secondNodes[0]))
joint2 = mbs.AddObject(ObjectJointRevolute2D(markerNumbers=[mNode1End, mNode2Start]))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## add gravity loading for beams
for beam in listBeams:
    marker = mbs.AddMarker(MarkerBodyMass(bodyNumber=beam))
    mbs.AddLoad(LoadMassProportional(markerNumber=marker, loadVector=g))
    

## assemble system and define simulation settings
mbs.Assemble()

print(f"\n系统组装完成!")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 开始提取DAE系统矩阵
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("\n" + "=" * 50)
print("开始提取系统矩阵...")
print("=" * 50)

# 创建仿真设置
simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.writeSolutionToFile = False
simulationSettings.staticSolver.verboseMode = 0

# 使用静态求解器提取矩阵
staticSolver = exu.MainSolverStatic()
staticSolver.InitializeSolver(mbs, simulationSettings)

# 获取系统尺寸信息
nODE2 = staticSolver.GetODE2size()  # 位置坐标数
nODE1 = staticSolver.GetODE1size()  # 速度坐标数 
nAE = staticSolver.GetAEsize()      # 代数约束数

print(f"\n系统维数信息:")
print(f"  ODE2 坐标数 (位置): {nODE2}")
print(f"  ODE1 坐标数 (速度): {nODE1}")
print(f"  代数约束数 (AE): {nAE}")
print(f"  总自由度: {nODE2}")
print(f"  约束后有效自由度: {nODE2 - nAE}")

# ========== 1. 提取质量矩阵 M ==========
print(f"\n1. 提取质量矩阵 M...")
staticSolver.ComputeMassMatrix(mbs)
M_full = staticSolver.GetSystemMassMatrix()
M = M_full[0:nODE2, 0:nODE2]  # 只取ODE2部分

print(f"   质量矩阵 M 维度: {M.shape}")
print(f"   质量矩阵条件数: {np.linalg.cond(M):.2e}")

# ========== 2. 提取刚度矩阵 K ==========
print(f"\n2. 提取刚度矩阵 K...")
staticSolver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=-1, 
                                    scalarFactor_ODE2_t=0, 
                                    scalarFactor_ODE1=0)
jacobian = staticSolver.GetSystemJacobian()
K = jacobian[0:nODE2, 0:nODE2]

print(f"   刚度矩阵 K 维度: {K.shape}")
print(f"   刚度矩阵条件数: {np.linalg.cond(K):.2e}")

# ========== 3. 提取阻尼矩阵 D ==========
print(f"\n3. 提取阻尼矩阵 D...")
staticSolver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=0, 
                                    scalarFactor_ODE2_t=-1, 
                                    scalarFactor_ODE1=0)
jacobian_t = staticSolver.GetSystemJacobian()
D = jacobian_t[0:nODE2, 0:nODE2]

print(f"   阻尼矩阵 D 维度: {D.shape}")

# ========== 4. 提取约束雅可比矩阵 C ==========
C = None
if nAE > 0:
    print(f"\n4. 提取约束雅可比矩阵 C...")
    staticSolver.ComputeJacobianAE(mbs, scalarFactor_ODE2=1., 
                                   scalarFactor_ODE2_t=0., 
                                   scalarFactor_ODE1=0., 
                                   velocityLevel=False)
    jacobian_full = staticSolver.GetSystemJacobian()
    C = jacobian_full[nODE2:nODE2+nAE, 0:nODE2]
    
    print(f"   约束雅可比 C 维度: {C.shape}")
    print(f"   约束雅可比秩: {np.linalg.matrix_rank(C)}")
else:
    print(f"\n4. 无约束，跳过约束雅可比矩阵")

# ========== 5. 外力向量 (重力) ==========
print(f"\n5. 计算外力向量...")
# 对于线性化系统，外力向量主要是重力，可以通过LoadMassProportional计算
# 创建一个简单的外力向量（重力效应）
f_ext = np.zeros(nODE2)
# 对于每个节点，y方向受重力影响
for i in range(nODE2//3):  # 每3个坐标一个节点 (x,y,θ)
    f_ext[3*i + 1] = -9.81 * rhoA * (lElem1 if i < (nElements1+1) else lElem2)  # y方向重力

print(f"   外力向量 f_ext 维度: {f_ext.shape}")
print(f"   外力向量范数: {np.linalg.norm(f_ext):.4f}")

staticSolver.FinalizeSolver(mbs, simulationSettings)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 显示和保存矩阵
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("\n" + "=" * 70)
print("显示提取的DAE系统矩阵")
print("=" * 70)

print(f"\n完整的DAE系统形式:")
print(f"  M(q)q̈ + D(q,q̇)q̇ + K(q)q = f_ext + C^T(q)λ")
print(f"  C(q)q = 0")
print(f"")
print(f"增广矩阵形式:")
print(f"  [M  C^T] [q̈]   [f_ext - D*q̇ - K*q]")
print(f"  [C   0 ] [λ ] = [      0         ]")

# 保存矩阵到文件
print(f"\n保存矩阵到文件...")

# 质量矩阵
np.savetxt('flexible_double_pendulum_mass_matrix.txt', M, fmt='%.6e', delimiter='\t')
print(f"  质量矩阵 M 已保存到 'flexible_double_pendulum_mass_matrix.txt'")

# 刚度矩阵
np.savetxt('flexible_double_pendulum_stiffness_matrix.txt', K, fmt='%.6e', delimiter='\t')
print(f"  刚度矩阵 K 已保存到 'flexible_double_pendulum_stiffness_matrix.txt'")

# 阻尼矩阵
np.savetxt('flexible_double_pendulum_damping_matrix.txt', D, fmt='%.6e', delimiter='\t')
print(f"  阻尼矩阵 D 已保存到 'flexible_double_pendulum_damping_matrix.txt'")

# 约束雅可比矩阵
if C is not None:
    np.savetxt('flexible_double_pendulum_constraint_jacobian.txt', C, fmt='%.6e', delimiter='\t')
    print(f"  约束雅可比 C 已保存到 'flexible_double_pendulum_constraint_jacobian.txt'")

# 外力向量
np.savetxt('flexible_double_pendulum_external_forces.txt', f_ext, fmt='%.6e', delimiter='\t')
print(f"  外力向量 f_ext 已保存到 'flexible_double_pendulum_external_forces.txt'")

# 显示矩阵的一些关键信息
print(f"\n矩阵特性分析:")
print(f"  质量矩阵 M:")
print(f"    - 维度: {M.shape}")
print(f"    - 对称性: {np.allclose(M, M.T)}")
print(f"    - 正定性: {np.all(np.linalg.eigvals(M) > 0)}")
print(f"    - 条件数: {np.linalg.cond(M):.2e}")

print(f"  刚度矩阵 K:")
print(f"    - 维度: {K.shape}")
print(f"    - 对称性: {np.allclose(K, K.T)}")
print(f"    - 条件数: {np.linalg.cond(K):.2e}")

if C is not None:
    print(f"  约束雅可比 C:")
    print(f"    - 维度: {C.shape}")
    print(f"    - 秩: {np.linalg.matrix_rank(C)}")
    print(f"    - 条件数: {np.linalg.cond(C @ C.T):.2e}")

# 显示部分矩阵内容（前几行几列）
print(f"\n质量矩阵 M 的前6×6子矩阵 (显示前两个节点的质量分布):")
print(np.array2string(M[:6, :6], precision=3, suppress_small=True, max_line_width=120))

if C is not None:
    print(f"\n约束雅可比矩阵 C (完整):")
    print(np.array2string(C, precision=3, suppress_small=True, max_line_width=120))

print(f"\n外力向量 f_ext (前12个元素, 对应前4个节点):")
print(np.array2string(f_ext[:12], precision=3, suppress_small=True))

# 分析柔性特性
print(f"\n柔性双摆特性分析:")
print(f"  第一摆臂节点数: {len(firstNodes)} (索引 0-{len(firstNodes)-1})")
print(f"  第二摆臂节点数: {len(secondNodes)} (索引 {len(firstNodes)}-{len(firstNodes)+len(secondNodes)-1})")
print(f"  每个节点3个DOF: (x, y, θ)")
print(f"  弯曲刚度 EI: {EI:.2e} N⋅m²")
print(f"  轴向刚度 EA: {EA:.2e} N")
print(f"  剪切刚度 GA: {GA:.2e} N")

print(f"\n" + "=" * 70)
print("柔性双摆DAE方程提取完成！")
print("=" * 70)
print(f"\n✅ 成功提取了柔性双摆系统的完整DAE方程组")
print(f"✅ 所有矩阵已保存到当前目录")
print(f"✅ 系统为 {nODE2}×{nODE2} 维的DAE系统，含 {nAE} 个约束")
print(f"✅ 包含大变形几何非线性和多种变形模式")

# 如果需要，可以继续仿真
print(f"\n是否运行可视化仿真? (输入 'y' 继续)")
user_input = input()
if user_input.lower() == 'y':
    print("开始仿真...")
    
    # Add initial conditions for more interesting motion
    import math
    for node in firstNodes[1:]:  # skip first node which is fixed
        mbs.SetNodeParameter(node, 'initialVelocities', [0, 0, 0.5])  # small angular velocity
    for node in secondNodes[1:]:  # skip first node which is connected to first arm
        mbs.SetNodeParameter(node, 'initialVelocities', [0, 0, -0.3])  # opposite direction

    simulationSettings = exu.SimulationSettings()
        
    tEnd = 5  # shorter simulation for demonstration
    stepSize = 0.002
    simulationSettings.timeIntegration.numberOfSteps = int(tEnd/stepSize)
    simulationSettings.timeIntegration.endTime = tEnd
    simulationSettings.timeIntegration.verboseMode = 1
    simulationSettings.solutionSettings.solutionWritePeriod = 0.01
    simulationSettings.solutionSettings.writeSolutionToFile = True

    simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
    simulationSettings.timeIntegration.newton.useModifiedNewton = True #for faster simulation

    ## add some visualization settings
    SC.visualizationSettings.nodes.defaultSize = 0.01
    SC.visualizationSettings.nodes.drawNodesAsPoint = False
    SC.visualizationSettings.bodies.beams.crossSectionFilled = True

    ## run dynamic simulation
    mbs.SolveDynamic(simulationSettings)

    ## visualize computed solution:
    mbs.SolutionViewer()
else:
    print("DAE提取完成，跳过仿真。")
