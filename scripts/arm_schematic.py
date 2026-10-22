"""Arm schematic for the report (FIGURE_PLAN #1): links, angles, torques, payload, gravity.

Geometry and conventions are read from ``src.dynamics.ArmParams`` so the figure cannot drift from the
model: ``q1`` from the +x (horizontal) axis, ``q2`` relative to link 1, both counter-clockwise positive,
gravity along -y, payload ``m_p`` a point mass at the tip of link 2. The faded arm along the x-axis is
the fixed initial condition ``x0 = 0`` (horizontal, at rest).

Run: ``python -m scripts.arm_schematic`` -> ``results/arm_schematic.png`` (and ``.pdf``).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

from src.dynamics import ArmParams

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "arm_schematic"

LINK1_C, LINK2_C = "#0072B2", "#E69F00"     # Okabe-Ito, as in docs/FIGURE_PLAN.md
PAYLOAD_C, TORQUE_C, GRAV_C = "#D55E00", "#CC79A7", "#555555"
INK = "#222222"


def torque_arrow(ax, centre, radius, start_deg, end_deg, label, label_deg):
    """Curved counter-clockwise arrow around a joint (positive torque direction)."""
    a = np.radians(np.linspace(start_deg, end_deg, 60))
    xs, ys = centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)
    ax.plot(xs[:-3], ys[:-3], color=TORQUE_C, lw=1.8, zorder=2)
    ax.add_patch(FancyArrowPatch((xs[-4], ys[-4]), (xs[-1], ys[-1]), arrowstyle="-|>",
                                 mutation_scale=14, color=TORQUE_C, lw=1.8, zorder=2))
    la = np.radians(label_deg)
    ax.text(centre[0] + 1.32 * radius * np.cos(la), centre[1] + 1.32 * radius * np.sin(la), label,
            color=TORQUE_C, fontsize=13, ha="center", va="center")


def dimension(ax, p0, p1, offset, label, color):
    """Dimension line parallel to p0 -> p1, shifted sideways by ``offset``."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * offset
    a, b = p0 + n, p1 + n
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=9, color=color, lw=0.9))
    for p in (p0, p1):
        ax.plot([p[0], p[0] + 1.15 * n[0]], [p[1], p[1] + 1.15 * n[1]], color=color, lw=0.6, ls=":")
    m = (a + b) / 2 + 0.55 * n / np.linalg.norm(n) * 0.09
    ax.text(*m, label, color=color, fontsize=11, ha="center", va="center",
            rotation=np.degrees(np.arctan2(d[1], d[0])), rotation_mode="anchor")


def main(q1_deg: float = 32.0, q2_deg: float = 48.0) -> None:
    p = ArmParams()
    q1, q2 = np.radians(q1_deg), np.radians(q2_deg)
    j1 = np.array([0.0, 0.0])
    j2 = j1 + p.l1 * np.array([np.cos(q1), np.sin(q1)])
    tip = j2 + p.l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    c1 = j1 + p.lc1 * np.array([np.cos(q1), np.sin(q1)])
    c2 = j2 + p.lc2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")

    # ground and axes
    ax.fill_between([-0.16, 0.16], -0.075, -0.045, color="#cccccc", hatch="////", edgecolor="#888888", lw=0)
    ax.plot([-0.16, 0.16], [-0.045, -0.045], color="#888888", lw=1)
    ax.plot([0, 0], [-0.045, 0], color="#888888", lw=2)
    for vec, name in (((1.0, 0), "x"), ((0, 1.0), "y")):
        end = np.array(vec) * (0.98 if name == "x" else 0.62)
        ax.add_patch(FancyArrowPatch((0, 0), end, arrowstyle="-|>", mutation_scale=11, color="#999999", lw=0.8))
        ax.text(*(end + (0.03, -0.035) if name == "x" else end + (-0.04, 0.0)), name, color="#777777", fontsize=11)

    # initial condition x0 = 0: horizontal arm at rest (faded)
    ax.plot([0, p.l1, p.l1 + p.l2], [0, 0, 0], color="#bbbbbb", lw=5, solid_capstyle="round", zorder=0, alpha=0.6)
    ax.text(p.l1 + p.l2 + 0.03, -0.05, "x(0) = 0:\nhorizontal, at rest", color="#999999",
            fontsize=8.5, ha="left", va="top")

    # links
    ax.plot(*zip(j1, j2), color=LINK1_C, lw=9, solid_capstyle="round", zorder=3)
    ax.plot(*zip(j2, tip), color=LINK2_C, lw=9, solid_capstyle="round", zorder=3)
    ax.text(*(c1 + np.array([0.10, -0.035])), "link 1\n$m_1$, $I_1$", color=LINK1_C, fontsize=10, ha="left", va="top")
    ax.text(*(c2 + np.array([0.05, 0.02])), "link 2\n$m_2$, $I_2$", color=LINK2_C, fontsize=10, ha="left", va="center")

    # joints and centres of mass
    for j in (j1, j2):
        ax.add_patch(Circle(j, 0.022, facecolor="white", edgecolor=INK, lw=1.6, zorder=5))
    for c in (c1, c2):
        ax.add_patch(Circle(c, 0.014, facecolor="white", edgecolor=INK, lw=1.0, zorder=5))
        ax.add_patch(Circle(c, 0.006, facecolor=INK, zorder=6))

    # payload at the tip
    ax.add_patch(Circle(tip, 0.035, facecolor=PAYLOAD_C, edgecolor=INK, lw=1.0, zorder=6))
    ax.text(*(tip + np.array([0.055, 0.0])), "payload $m_p$\n(point mass at tip)", color=PAYLOAD_C,
            fontsize=10, ha="left", va="center")

    # angle q1 from the horizontal
    ax.add_patch(Arc(j1, 0.30, 0.30, theta1=0, theta2=q1_deg, color=INK, lw=1.2))
    a = np.radians(q1_deg / 2)
    ax.text(0.19 * np.cos(a) + 0.015, 0.19 * np.sin(a) - 0.005, "$q_1$", fontsize=13, va="center")

    # q2 relative to the extension of link 1
    ext = j2 + 0.20 * np.array([np.cos(q1), np.sin(q1)])
    ax.plot(*zip(j2, ext), color=INK, lw=0.9, ls="--")
    ax.add_patch(Arc(j2, 0.26, 0.26, theta1=q1_deg, theta2=q1_deg + q2_deg, color=INK, lw=1.2))
    a = np.radians(q1_deg + q2_deg / 2)
    ax.text(*(j2 + 0.17 * np.array([np.cos(a), np.sin(a)]) + np.array([0.012, 0.0])), "$q_2$", fontsize=13, va="center")

    # torques (counter-clockwise positive), with viscous friction at each joint
    torque_arrow(ax, j1, 0.10, 45, 165, r"$\tau_1$", 128)
    torque_arrow(ax, j2, 0.075, 80, 335, r"$\tau_2$", 4)
    ax.text(0.0, -0.115, r"friction $-b\,\dot q_1$", fontsize=8.5, color="#777777", ha="center")
    ax.text(*(j2 + np.array([0.0, -0.14])), r"friction $-b\,\dot q_2$", fontsize=8.5, color="#777777", ha="center")

    # dimensions
    dimension(ax, j1, j2, 0.075, f"$l_1$ = {p.l1} m", LINK1_C)
    dimension(ax, j2, tip, 0.07, f"$l_2$ = {p.l2} m", LINK2_C)

    # gravity
    g0 = np.array([-0.30, 0.62])
    ax.add_patch(FancyArrowPatch(g0, g0 + (0, -0.20), arrowstyle="-|>", mutation_scale=16, color=GRAV_C, lw=2))
    ax.text(*(g0 + (0.03, -0.10)), "$g$ = 9.81 m/s$^2$", color=GRAV_C, fontsize=10, va="center")

    # legend box with the fixed constants
    txt = (f"$m_1$ = {p.m1} kg,  $m_2$ = {p.m2} kg\n"
           r"$l_{c,i} = l_i/2$,   $I_i = m_i l_i^2/12$" "\n"
           r"inputs: $m_p$, $b$, $\tau_1(t)$, $\tau_2(t)$" "\n"
           r"output: $x = [q_1, q_2, \dot q_1, \dot q_2]$, $t \in [0, 3]$ s")
    ax.text(-0.34, 0.92, txt, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f6f6f6", edgecolor="#cccccc"))

    ax.set_xlim(-0.38, 1.25)
    ax.set_ylim(-0.17, 0.95)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    for ext_ in (".png", ".pdf"):
        fig.savefig(OUT.with_suffix(ext_), dpi=200, bbox_inches="tight")
        print("saved", OUT.with_suffix(ext_).relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
