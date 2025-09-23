"""Utility to dump Craig–Bampton data for the flexible double pendulum.

The generated dataset feeds the standalone C++ implementation that mirrors the
Exudyn example in `flexible_pendulum.py`.
"""

from __future__ import annotations

from pathlib import Path
import sys

# ensure local exudyn package is importable without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

from flexible_pendulum import (
    FlexibleLinkParameters,
    InitialState,
    SimulationSettings,
    export_flexible_pendulum_dataset,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_output = repo_root / "customFlexiblePendulum" / "data"

    params = FlexibleLinkParameters()
    init = InitialState()
    sim = SimulationSettings(store_trajectory=False, matlab_export=None)

    export_flexible_pendulum_dataset(default_output, params, init, sim)
    print(f"Exported flexible pendulum dataset to {default_output}")


if __name__ == "__main__":
    main()
