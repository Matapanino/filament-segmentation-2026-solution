"""One argparse definition of the decision rule, shared by every script.

The rule is what this competition is scored on (F-009/F-010), so its flags live
in one place: a submission, an OOF safety report and a screen arm are then the
same object built the same way — the drift that a hand-copied flag list invites
is what this module exists to prevent.
"""
from __future__ import annotations

import argparse

from .instancing import Rule


def add_rule_args(
    ap: argparse.ArgumentParser, min_area_required: bool = False
) -> argparse.ArgumentParser:
    """Add one flag per ``Rule`` stage, with the Rule's own defaults."""
    d = Rule()
    g = ap.add_argument_group("decision rule (filseg.instancing.Rule)")
    g.add_argument("--thr", type=float, default=d.thr,
                   help="threshold; the LOW threshold when --hys-hi is set")
    g.add_argument("--min-area", type=int, required=min_area_required,
                   default=d.min_area, help="minimum component area in px")
    g.add_argument("--close-px", type=int, default=d.close_px,
                   help="isotropic closing kernel (R2 ship mechanism: 21)")
    g.add_argument("--hys-hi", type=float, default=d.hys_hi,
                   help="hysteresis core gate; 0 disables hysteresis")
    g.add_argument("--line-px", type=int, default=d.line_px,
                   help="oriented (line-SE) closing span; 0 disables")
    g.add_argument("--n-angles", type=int, default=d.n_angles)
    g.add_argument("--link-gap", type=int, default=d.link_gap,
                   help="collinear tip linking max gap in px; 0 disables")
    g.add_argument("--link-angle", type=float, default=d.link_angle)
    g.add_argument("--link-window", type=int, default=d.link_window)
    g.add_argument("--link-min-area", type=int, default=d.link_min_area)
    g.add_argument("--fill", action="store_true", help="fill interior holes")
    g.add_argument("--accept-tau", type=float, default=d.accept_tau,
                   help="acceptance-model p(match) cut; needs --acceptance-model")
    return ap


def rule_from_args(args: argparse.Namespace) -> Rule:
    """Build the ``Rule`` from parsed args (unknown extra args are ignored)."""
    fields = Rule.__dataclass_fields__
    return Rule(**{k: getattr(args, k) for k in fields if hasattr(args, k)})
