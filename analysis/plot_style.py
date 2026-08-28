#!/usr/bin/env python3
"""
Unified plotting style for all paper figures.
Import and call apply_style() at the top of each plotting script.
Also exposes a shared COLORS palette.
"""
from __future__ import annotations
import matplotlib as mpl

# Color palette (color-blind friendly, muted high-contrast)
COLORS = {
    # Methods
    'Vanilla_FL': '#6b8ec1',      # muted blue
    'Trimmed_Mean': '#7fb77e',    # muted green
    # Metrics (RQ4)
    'participation': '#6b8ec1',   # reuse muted blue
    'attack': '#c97f7f',          # muted red
    'accuracy': '#7fb77e',        # reuse muted green
    # Accents
    'accent1': '#9B59B6',         # purple
}

_DEF_PARAMS = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Font sizes
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    # Lines & markers
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    # Grid aesthetics (use per-axes grid(...))
}

def apply_style(extra: dict | None = None) -> None:
    params = dict(_DEF_PARAMS)
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)

if __name__ == '__main__':
    # Smoke smoke check
    apply_style()
    print('Plot style applied with params:', {k: mpl.rcParams[k] for k in _DEF_PARAMS})

