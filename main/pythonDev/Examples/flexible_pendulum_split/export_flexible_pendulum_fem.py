"""Export NGsolve-based FEM data for the flexible double pendulum to a text file.

The script rebuilds the Craig–Bampton reduced-order models for both pendulum
links and stores all information required to run the Exudyn simulation later on
without touching NGsolve again. The resulting ``.txt`` file contains JSON data
so it remains human-readable while preserving numerical precision.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse import csr_matrix

import exudyn as exu
from exudyn.FEM import (
    FEMinterface,
    HCBstaticModeSelection,
)


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
    render: bool = False
    frames_per_second: int = 60
    store_trajectory: bool = True
    matlab_export: Optional[str] = "output/flexible_pendulum_sim.mat"
    adaptive_step: bool = False
    spectral_radius: float = 0.85
    modified_newton: bool = True
    newton_tol: float = 1e-8
    newton_max_iter: int = 12


def _build_beam(length: float, modes: int, params: FlexibleLinkParameters, label: str) -> Dict[str, Any]:
    try:
        import ngsolve as ngs
        from netgen.csg import CSGeometry, OrthoBrick, Pnt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "NGsolve/Netgen is required to export the flexible pendulum data."
        ) from exc

    geo = CSGeometry()
    brick = OrthoBrick(
        Pnt(0.0, -params.thickness * 0.5, -params.width * 0.5),
        Pnt(length, params.thickness * 0.5, params.width * 0.5),
    )
    geo.Add(brick)

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
    if not nodes_root or not nodes_tip:
        raise RuntimeError(f"Failed to identify boundary nodes for {label} link")

    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=[nodes_root, nodes_tip],
        nEigenModes=modes,
        useSparseSolver=True,
        computationMode=HCBstaticModeSelection.RBE2,
    )

    if not fem.surface:
        fem.VolumeToSurfaceElements()

    positions = fem.GetNodePositionsAsArray()
    surface_trigs = np.asarray(fem.GetSurfaceTriangles(), dtype=int)

    root_mean = positions[nodes_root].mean(axis=0)
    tip_mean = positions[nodes_tip].mean(axis=0)

    weights_root = (np.ones(len(nodes_root)) / len(nodes_root)).tolist()
    weights_tip = (np.ones(len(nodes_tip)) / len(nodes_tip)).tolist()

    return {
        "fem": fem,
        "nodes_root": list(map(int, nodes_root)),
        "nodes_tip": list(map(int, nodes_tip)),
        "weights_root": weights_root,
        "weights_tip": weights_tip,
        "root_mean": root_mean.tolist(),
        "tip_mean": tip_mean.tolist(),
        "surface_trigs": surface_trigs.tolist() if surface_trigs.size else [],
    }


def _csr_to_dict(matrix: csr_matrix | None) -> Dict[str, Any] | None:
    if matrix is None:
        return None
    csr = csr_matrix(matrix)
    return {
        "type": "csr",
        "data": csr.data.tolist(),
        "indices": csr.indices.tolist(),
        "indptr": csr.indptr.tolist(),
        "shape": list(csr.shape),
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return str(value)


def _serialize_fem(fem: FEMinterface) -> Dict[str, Any]:
    data = fem.GetDictionary()
    data["massMatrix"] = _csr_to_dict(data.get("massMatrix"))
    data["stiffnessMatrix"] = _csr_to_dict(data.get("stiffnessMatrix"))
    return _to_builtin(data)


def export_dataset(output_path: Path) -> None:
    params = FlexibleLinkParameters()
    init = InitialState()
    sim = SimulationSettings()

    beam_first = _build_beam(params.length_first, params.modes_first, params, "first")
    beam_second = _build_beam(params.length_second, params.modes_second, params, "second")

    beam_first_payload = {key: value for key, value in beam_first.items() if key != "fem"}
    beam_second_payload = {key: value for key, value in beam_second.items() if key != "fem"}
    beam_first_payload["fem"] = _serialize_fem(beam_first["fem"])
    beam_second_payload["fem"] = _serialize_fem(beam_second["fem"])

    payload = {
        "parameters": asdict(params),
        "initial_state": asdict(init),
        "simulation": asdict(sim),
        "beam_first": beam_first_payload,
        "beam_second": beam_second_payload,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    exu.Print(f"Exported flexible pendulum FEM data to {output_path}")


def main() -> None:
    default_output = Path(__file__).resolve().with_name("flexible_pendulum_fem_data.txt")
    parser = argparse.ArgumentParser(description="Export FEM data for the flexible double pendulum")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination text file (JSON formatted)",
    )
    args = parser.parse_args()
    export_dataset(args.output)


if __name__ == "__main__":
    main()
