"""CLI entry point:  python -m lax9 <phase> [options]

Phases: intake frames overlaps calibrate stitch scale detect map plan
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from .config import load_config

PHASES = ["intake", "frames", "overlaps", "calibrate", "stitch",
          "scale", "detect", "map", "plan", "grid"]

GATES = {
    "scale": "Phase 5 GATE: a handful of measured floor points/distances (laser meter). "
             "Without them the map is correct in shape but not in metres.",
    "detect": "Phase 6 GATE: the pallet instance-segmentation model (weights or endpoint) "
              "+ how to call it. A manual-annotation fallback is allowed for testing.",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lax9", description="LAX-9 Stage 1 pipeline")
    p.add_argument("phase", choices=PHASES, help="pipeline phase to run")
    p.add_argument("--config", default=None, help="path to config.yaml")
    # plan options
    p.add_argument("--pallet", default=None, help="(plan) pallet id to move")
    p.add_argument("--to", nargs=2, type=float, metavar=("X", "Y"),
                   help="(plan) destination in metres")
    p.add_argument("--selfcal", action="store_true",
                   help="(calibrate) self-calibrate from footage (no checkerboard)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    phase = args.phase
    if phase == "intake":
        from . import intake
        intake.run(cfg, args)
    elif phase == "frames":
        from . import frames
        frames.run(cfg, args)
    elif phase == "overlaps":
        from . import overlaps
        overlaps.run(cfg, args)
    elif phase == "calibrate":
        from . import calibrate
        calibrate.run(cfg, args)
    elif phase == "grid":
        from . import grid
        grid.run(cfg, args)
    else:
        # Not-yet-built / gated phases: be explicit rather than silently failing.
        print(f"\nPhase '{phase}' is not implemented yet in this build.")
        if phase in GATES:
            print(f"  {GATES[phase]}")
        else:
            order = {p: i for i, p in enumerate(PHASES)}
            print(f"  It comes after the phases already built "
                  f"(intake, frames, overlaps). Build order: {' -> '.join(PHASES)}.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
