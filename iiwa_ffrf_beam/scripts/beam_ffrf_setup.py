"""Utility script to prepare a reduced-order flexible beam (FFRF + HCB) for the iiwa scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import exudyn as exu
from exudyn.FEM import FEMinterface, HCBstaticModeSelection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "data" / "fem" / "beam_config.json"
DEFAULT_CMS_SUFFIX = "_cms"

def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Beam configuration not found: {path}")
    with path.open("r", encoding="utf-8") as cfg_file:
        return json.load(cfg_file)

def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def import_mesh_from_ngsolve(fem: FEMinterface, config: Dict) -> None:
    geometry = config.get("geometry", {})
    mesh_order = config.get("mesh_order", 1)
    material = config.get("material", {})

    try:
        import ngsolve as ngs
        from netgen.occ import Box
        from netgen.occ import OCCGeometry
    except ImportError as exc:
        raise ImportError("NGsolve/netgen not available. Install ngsolve to generate meshes or set generate_with_ngsolve=false.") from exc

    geom_type = geometry.get("type", "box")
    if geom_type != "box":
        raise ValueError(f"Unsupported geometry type '{geom_type}'. Extend import_mesh_from_ngsolve accordingly.")

    length = geometry.get("length", 0.6)
    width = geometry.get("width", 0.05)
    height = geometry.get("height", 0.05)
    mesh_size = geometry.get("mesh_size", min(width, height) * 0.5)

    box = Box((0.0, -0.5 * width, -0.5 * height), (length, 0.5 * width, 0.5 * height))
    occ_geo = OCCGeometry(box)
    mesh = ngs.Mesh(occ_geo.GenerateMesh(maxh=mesh_size))

    density = material.get("density", 7800.0)
    youngs = material.get("youngs_modulus", 2.1e11)
    nu = material.get("poissons_ratio", 0.3)

    fem.ImportMeshFromNGsolve(
        mesh,
        density=density,
        youngsModulus=youngs,
        poissonsRatio=nu,
        meshOrder=mesh_order,
    )


def load_or_generate_fem(config: Dict) -> FEMinterface:
    fem = FEMinterface()
    cache_basename = config.get("fem_cache_basename")
    if cache_basename is None:
        raise KeyError("'fem_cache_basename' missing in configuration")

    cache_path = (PROJECT_ROOT / cache_basename).resolve()
    ensure_parent_directory(cache_path)

    if config.get("generate_with_ngsolve", False):
        exu.Print("Generating FEM mesh with NGsolve ...")
        import_mesh_from_ngsolve(fem, config)
        fem.SaveToFile(str(cache_path))
        exu.Print(f"Mesh saved to {cache_path}.npz")
    else:
        load_target = cache_path
        cms_suffix = config.get("cms_save_suffix", DEFAULT_CMS_SUFFIX)
        if config.get("load_cms_directly", False):
            load_target = Path(str(cache_path) + cms_suffix)
        exu.Print(f"Loading FEM data from {load_target}.npz ...")
        fem.LoadFromFile(str(load_target))

    return fem

def build_interface_sets(fem: FEMinterface, interface_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    selection = interface_cfg.get("selection_type", "plane").lower()
    weighting_mode = interface_cfg.get("weighting", "uniform").lower()

    if selection == "plane":
        point = interface_cfg.get("point", [0.0, 0.0, 0.0])
        normal = interface_cfg.get("normal", [1.0, 0.0, 0.0])
        tolerance = interface_cfg.get("tolerance", 1e-4)
        nodes = fem.GetNodesInPlane(point, normal, tolerance=tolerance)
    elif selection == "node_list":
        nodes = interface_cfg.get("node_numbers", [])
    else:
        raise ValueError(f"Unsupported selection_type '{selection}'")

    if len(nodes) == 0:
        raise RuntimeError(f"Interface '{interface_cfg.get('name', 'unnamed')}' returned no nodes. Check configuration or tolerance.")

    nodes = np.array(nodes, dtype=int)
    if weighting_mode == "uniform":
        weights = np.ones(len(nodes)) / len(nodes)
    elif weighting_mode == "custom":
        raw_weights = np.array(interface_cfg.get("weights", []), dtype=float)
        if len(raw_weights) != len(nodes):
            raise ValueError("Custom weights length must match number of selected nodes")
        weights = raw_weights
    else:
        raise ValueError(f"Unsupported weighting mode '{weighting_mode}'")

    return nodes, weights

def compute_cms(fem: FEMinterface, config: Dict) -> Dict:
    interfaces_cfg = config.get("interfaces", [])
    if not interfaces_cfg:
        raise ValueError("No interfaces defined in beam configuration")

    boundary_nodes: List[np.ndarray] = []
    boundary_weights: List[np.ndarray] = []
    interface_report: List[Dict] = []

    for iface in interfaces_cfg:
        nodes, weights = build_interface_sets(fem, iface)
        boundary_nodes.append(nodes)
        boundary_weights.append(weights)
        interface_report.append({
            "name": iface.get("name", "interface"),
            "selection_type": iface.get("selection_type", "plane"),
            "node_count": int(len(nodes))
        })
        exu.Print(f"Interface '{iface.get('name', 'interface')}' uses {len(nodes)} nodes")

    n_eigen = config.get("n_eigen_modes", 20)
    use_sparse = config.get("use_sparse_solver", True)
    hcb_mode_name = config.get("hcb_mode", "RBE2")
    hcb_mode = getattr(HCBstaticModeSelection, hcb_mode_name)

    exu.Print(f"Computing {n_eigen} HCB modes (mode={hcb_mode_name}, sparse={use_sparse}) ...")
    fem.ComputeHurtyCraigBamptonModes(
        boundaryNodesList=boundary_nodes,
        nEigenModes=n_eigen,
        useSparseSolver=use_sparse,
        computationMode=hcb_mode,
    )

    damping = config.get("stiffness_proportional_damping", 0.0)
    return {
        "interfaces": interface_report,
        "n_eigen_modes": n_eigen,
        "stiffness_proportional_damping": damping
    }

def maybe_compute_post_processing_modes(fem: FEMinterface, config: Dict) -> None:
    if not config.get("compute_stress_modes", False):
        return
    exu.Print("Computing stress post-processing modes ...")
    try:
        fem.ComputePostProcessingModesNGsolve(None)
    except Exception as exc:  # pylint: disable=broad-except
        exu.Print(f"WARNING: stress mode computation failed: {exc}")


def save_fem(fem: FEMinterface, config: Dict) -> Path:
    cache_basename = config.get("fem_cache_basename")
    cache_path = (PROJECT_ROOT / cache_basename).resolve()
    cms_suffix = config.get("cms_save_suffix", DEFAULT_CMS_SUFFIX)
    target = Path(str(cache_path) + cms_suffix)
    ensure_parent_directory(target)
    fem.SaveToFile(str(target))
    exu.Print(f"CMS data saved to {target}.npz")
    return target

def save_report(report: Dict, config_path: Path, cms_path: Path) -> None:
    report_path = cms_path.parent / "beam_setup_report.json"
    payload = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "cms_file": str(cms_path.relative_to(PROJECT_ROOT)) + ".npz",
        **report,
    }
    with report_path.open("w", encoding="utf-8") as rep_file:
        json.dump(payload, rep_file, indent=2)
    exu.Print(f"Report written to {report_path}")


def main() -> None:
    config = load_config(CONFIG_PATH)
    fem = load_or_generate_fem(config)
    report = compute_cms(fem, config)
    maybe_compute_post_processing_modes(fem, config)
    cms_path = save_fem(fem, config)
    save_report(report, CONFIG_PATH, cms_path)


if __name__ == "__main__":
    main()
