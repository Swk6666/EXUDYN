"""Flexible double pendulum using Craig–Bampton reduced-order models.

Each pendulum link is exported as a floating-frame (FFRF) flexible body built
from a Netgen/NGSolve mesh and reduced with the Hurty–Craig–Bampton method.
The two bodies are coupled with revolute joints realized via super-element
markers while initial angles/velocities mimic a classical rigid double pendulum.

Prerequisites: NGsolve/Netgen must be installed and importable from Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

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

    length_first: float = 0.5          # [m]
    length_second: float = 0.4         # [m]
    width: float = 0.012               # [m]
    thickness: float = 0.003           # [m]
    youngs_modulus: float = 2.1e9      # [Pa]
    density: float = 1180.0            # [kg/m^3]
    poisson: float = 0.34
    modes_first: int = 3
    modes_second: int = 3
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
    render: bool = True


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _planar_velocity(omega: float, position: np.ndarray) -> np.ndarray:
    return np.cross(np.array([0.0, 0.0, omega]), position)


def _build_cms_beam(
    length: float,
    params: FlexibleLinkParameters,
    n_modes: int,
    label: str,
) -> Dict[str, np.ndarray | FEMinterface]:
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

    return {
        "fem": fem,
        "nodes_root": np.array(nodes_root, dtype=int),
        "nodes_tip": np.array(nodes_tip, dtype=int),
        "weights_root": weights_root,
        "weights_tip": weights_tip,
        "root_mean": root_mean,
        "tip_mean": tip_mean,
    }


def _create_superelement_marker(
    mbs: exu.MainSystem,
    body_number: int,
    mesh_nodes: np.ndarray,
    weights: np.ndarray,
    offset: Iterable[float] | None = None,
    show: bool = False,
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


def build_system(params: FlexibleLinkParameters, init: InitialState) -> Dict[str, object]:
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
    tip_mean_2 = beam_second["tip_mean"]

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

    return {
        "SC": SC,
        "mbs": mbs,
        "obj_first": obj_first,
        "obj_second": obj_second,
        "tip_sensor": tip_sensor,
        "hinge_pos": hinge_pos,
    }


def run_simulation(
    params: FlexibleLinkParameters = FlexibleLinkParameters(),
    init: InitialState = InitialState(),
    sim: SimulationSettings = SimulationSettings(),
) -> None:
    model = build_system(params, init)
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
    settings.solutionSettings.sensorsWritePeriod = 0.01
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


if __name__ == "__main__":
    params = FlexibleLinkParameters()
    init = InitialState()
    sim = SimulationSettings()
    run_simulation(params, init, sim)
