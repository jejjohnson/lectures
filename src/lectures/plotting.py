"""Consistent figure styling for lecture notes and slides.

A single source of truth for plot aesthetics so that figures embedded in the
deep ``topics/`` notes and the curated ``courses/`` slides look identical.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Palette tuned to read well on both white (notes) and dark (slides) backgrounds.
PALETTE: dict[str, str] = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
}

# Named semantic colors used throughout the lecture figures.
INK = "#22223b"  # main series / "the truth"
COOL = "#3d5a80"  # calm / well-behaved case
WARM = "#e07a5f"  # danger / overconfident model
BAND = "#98c1d9"  # uncertainty band
CRASH = "#bc4749"  # tail events / crashes
GRID = "#d9d9e3"  # annotation box border
PANEL = "#f4f4f8"  # annotation box fill


def set_style(context: str = "notes") -> None:
    """Apply the shared Matplotlib style.

    Args:
        context: ``"notes"`` for the website/book (light) or ``"slides"`` for
            reveal.js decks (larger fonts, transparent background).
    """
    base = {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE.values())),
        "font.size": 11,
    }
    if context == "slides":
        base.update(
            {
                "font.size": 16,
                "figure.figsize": (8, 4.5),
                "figure.dpi": 200,  # crisper rasters when projected/scaled on slides
                "savefig.dpi": 200,
                "savefig.transparent": True,
                "axes.facecolor": "none",
                "figure.facecolor": "none",
            }
        )
    plt.rcParams.update(base)


def finalize(ax: plt.Axes, title: str | None = None) -> plt.Axes:
    """Apply common finishing touches to an axis."""
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return ax


def despine(ax: plt.Axes) -> None:
    """Remove the top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    ha: str = "center",
    va: str = "top",
    fontsize: int = 10,
) -> None:
    """Place a rounded annotation box at axes-fraction coordinates (x, y)."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=GRID),
    )


def mark_event(
    ax: plt.Axes,
    x: object,
    label: str,
    *,
    color: str = CRASH,
    top: float = 0.95,
    fontsize: int = 9,
) -> None:
    """Mark a dated event on a time-series axis with a line and a bubble.

    Draws a faint vertical rule at data coordinate ``x`` (e.g. a ``Timestamp``)
    and a rounded label bubble near the top of the plot, so figures can point at
    nameable market events (crashes, votes) without overlapping the data.

    Args:
        ax: Target axis (its x-axis is in data units, e.g. dates).
        x: Where to place the line, in *data* coordinates.
        label: Short text for the bubble.
        color: Line and bubble-border colour.
        top: Vertical position of the bubble, in axes fraction (0 = bottom).
        fontsize: Bubble font size.
    """
    ax.axvline(x, color=color, lw=1.2, ls="--", alpha=0.7, zorder=1)
    trans = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    ax.text(
        x,
        top,
        label,
        transform=trans,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=color, alpha=0.95),
        zorder=5,
    )
