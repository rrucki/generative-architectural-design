"""
generate_graphical_abstract.py
------------------------------
Generates a publication-grade scientific Graphical Abstract (2.5:1 aspect ratio,
3000x1200 px at 300 DPI) typeset in Charis SIL + STIX mathematics, compliant with
Elsevier / Applied Soft Computing guidelines.

All parcel boundaries (P_plot), buildable setback envelopes (E_design), and building
footprints (P) are strictly 4-edge quadrilaterals (verified via Shapely assertions).
Exports PNG (300 DPI), vector PDF, and LZW-compressed TIFF (300 DPI) into manuscript/figures/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    Polygon as MplPolygon,
    FancyArrowPatch,
    Rectangle,
)
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Muted academic color palette (colorblind-friendly, high contrast in grayscale)
C_INK = "#111827"
C_SUBTLE = "#4B5563"
C_RULE = "#374151"
C_NAVY = "#1D3557"
C_BLUE = "#2B5B84"
C_TEAL = "#0F6E56"
C_GREEN = "#1B7837"
C_PURPLE = "#6A51A3"
C_PURPLE_FILL = "#EFEAF7"
C_CRIMSON = "#B2182B"
C_AMBER = "#B45309"


def make_oriented_rect(cx, cy, w, h, angle_deg):
    """Creates an exact 90-degree 4-vertex rectangle in isotropic figure coordinates."""
    rad = np.radians(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[ca, -sa], [sa, ca]])
    return local @ rot.T + np.array([cx, cy])


def add_panel_header(ax, x, y, w, label, title):
    """Classic scientific figure panel header with crisp booktabs-style underline."""
    ax.text(
        x,
        y + 0.017,
        f"{label}  {title}",
        fontsize=8.9,
        fontweight="bold",
        color=C_INK,
        ha="left",
        va="bottom",
        zorder=5,
    )
    ax.plot([x, x + w], [y + 0.004, y + 0.004], color=C_RULE, lw=0.95, zorder=5)


def add_sharp_box(ax, x, y, w, h, fc="#FFFFFF", ec="#9CA3AF", lw=0.80, ls="-", radius=0.005, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linestyle=ls,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def add_arrow(ax, x0, y0, x1, y1, color=C_RULE, lw=1.05, hl=4.8, hw=3.3, style="-|>", z=5):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=f"{style},head_length={hl},head_width={hw}",
        color=color,
        linewidth=lw,
        zorder=z,
        mutation_scale=1.0,
    )
    ax.add_patch(arr)


def draw_right_angle_marker(ax, p_corner, p_prev, p_next, size=0.022, color=C_NAVY, lw=0.9, z=7):
    """Draws a crisp geometric right-angle square indicator at a footprint corner."""
    u = p_prev - p_corner
    v = p_next - p_corner
    u = u / np.linalg.norm(u) * size
    v = v / np.linalg.norm(v) * size
    pts = np.array([p_corner + u, p_corner + u + v, p_corner + v])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z)


def draw_parcel_scene(
    ax,
    ox,
    oy,
    sx,
    sy,
    footprints,
    show_annotations=False,
    show_vertices=False,
):
    """Draws a strict 4-edge irregular quadrilateral parcel P_plot and 4-edge asymmetrically
    offset setback envelope E_design with 4-edge footprint(s). Verified with Shapely assertions."""
    plot_norm = np.array([[0.04, 0.12], [0.95, 0.06], [0.89, 0.94], [0.13, 0.80]])
    env_norm = np.array([[0.14, 0.20], [0.75, 0.15], [0.71, 0.83], [0.21, 0.72]])
    assert len(plot_norm) == 4 and len(env_norm) == 4

    def tf(pts):
        return np.column_stack([ox + pts[:, 0] * sx, oy + pts[:, 1] * sy])

    plot_xy = tf(plot_norm)
    env_xy = tf(env_norm)
    plot_poly = ShapelyPolygon(plot_xy)
    env_poly = ShapelyPolygon(env_xy)
    assert env_poly.within(plot_poly), "Envelope E_design must lie strictly inside P_plot"

    ax.add_patch(
        MplPolygon(
            plot_xy,
            closed=True,
            facecolor="#FFFFFF",
            edgecolor="#6B7280",
            linestyle="--",
            linewidth=1.05,
            zorder=2,
        )
    )
    ax.add_patch(
        MplPolygon(
            env_xy,
            closed=True,
            facecolor=C_PURPLE_FILL,
            edgecolor=C_PURPLE,
            linestyle="-",
            linewidth=1.1,
            alpha=0.85,
            zorder=3,
        )
    )

    drawn_fps = []
    for fp in footprints:
        if "rect" in fp:
            rx, ry, rw, rh, ang = fp["rect"]
            pts = make_oriented_rect(ox + rx * sx, oy + ry * sy, rw * sx, rh * sy, ang)
        else:
            pts = tf(np.array(fp["pts"]))
        assert len(pts) == 4, "Every footprint must have strictly 4 edges"
        if fp.get("must_be_inside", False):
            assert ShapelyPolygon(pts).within(env_poly), f"Footprint violates E_design: {fp}"
        ax.add_patch(
            MplPolygon(
                pts,
                closed=True,
                facecolor=fp.get("fc", "none"),
                edgecolor=fp.get("ec", C_GREEN),
                linestyle=fp.get("ls", "-"),
                linewidth=fp.get("lw", 1.6),
                alpha=fp.get("alpha", 1.0),
                zorder=fp.get("z", 4),
            )
        )
        if show_vertices or fp.get("markers", False):
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=fp.get("ms", 12),
                color=fp.get("ec", C_GREEN),
                edgecolors="#FFFFFF",
                linewidths=0.4,
                zorder=fp.get("z", 4) + 2,
            )
        drawn_fps.append(pts)

    if show_annotations and drawn_fps:
        fp_pts = drawn_fps[0]
        ax.text(
            ox + 0.07 * sx,
            oy + 0.87 * sy,
            r"$P_{\mathrm{plot}}$",
            fontsize=8.4,
            color="#4B5563",
            zorder=7,
        )
        ax.text(
            ox + 0.45 * sx,
            oy + 0.72 * sy,
            r"$\mathcal{E}_{\mathrm{design}}$",
            fontsize=8.4,
            color=C_PURPLE,
            zorder=7,
        )
        draw_right_angle_marker(ax, fp_pts[0], fp_pts[1], fp_pts[3], size=0.024, color=C_NAVY, lw=0.95)
        ax.text(
            fp_pts[0, 0] + 0.030,
            fp_pts[0, 1] + 0.014,
            r"$\alpha_i$",
            fontsize=7.4,
            color=C_NAVY,
            ha="left",
            va="bottom",
            zorder=7,
        )
        mid_bot = 0.55 * fp_pts[0] + 0.45 * fp_pts[1]
        mid_top = 0.55 * fp_pts[3] + 0.45 * fp_pts[2]
        add_arrow(
            ax,
            mid_bot[0],
            mid_bot[1] + 0.005,
            mid_top[0],
            mid_top[1] - 0.005,
            color=C_NAVY,
            lw=0.9,
            hl=4.0,
            hw=2.8,
            style="<|-|>",
            z=7,
        )
        center_w = 0.5 * (mid_bot + mid_top)
        ax.text(
            center_w[0] + 0.012,
            center_w[1],
            r"$W(P)$",
            fontsize=7.4,
            color=C_NAVY,
            ha="left",
            va="center",
            zorder=7,
        )
        p_env_r = 0.5 * (env_xy[1] + env_xy[2])
        p_plt_r = 0.5 * (plot_xy[1] + plot_xy[2])
        add_arrow(
            ax,
            p_env_r[0] + 0.004,
            p_env_r[1],
            p_plt_r[0] - 0.004,
            p_plt_r[1],
            color="#6B7280",
            lw=0.8,
            hl=3.5,
            hw=2.5,
            style="<|-|>",
            z=7,
        )
        ax.text(
            0.5 * (p_env_r[0] + p_plt_r[0]),
            0.5 * (p_env_r[1] + p_plt_r[1]) + 0.024,
            r"$\Delta_{\mathrm{shift}}$",
            fontsize=7.0,
            color="#4B5563",
            ha="center",
            va="bottom",
            zorder=7,
        )

    return plot_xy, env_xy, drawn_fps


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Charis SIL", "Charter", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": C_RULE,
            "axes.linewidth": 0.75,
        }
    )

    # Exact 2.5:1 ratio: 10.0 x 4.0 inches at 300 DPI = 3000 x 1200 px
    fig = plt.figure(figsize=(10.0, 4.0), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Symmetric margins (0.030 left & right) and four balanced columns
    w1, w2, w3, w4 = 0.555, 0.585, 0.560, 0.632
    gap = 0.036
    x1 = 0.030
    x2 = x1 + w1 + gap
    x3 = x2 + w2 + gap
    x4 = x3 + w3 + gap
    # Right edge check: x4 + w4 = 2.470 (0.030 right margin)
    hdr_y = 0.895

    # Subtle vertical hairline rules separating the 4 conceptual stages
    for x_sep in [x1 + w1 + gap / 2, x2 + w2 + gap / 2, x3 + w3 + gap / 2]:
        ax.plot([x_sep, x_sep], [0.045, 0.945], color="#E5E7EB", lw=0.75, ls="-", zorder=1)

    # =========================================================================
    # PANEL (a): Constrained Domain (N = 4)
    # =========================================================================
    add_panel_header(ax, x1, hdr_y, w1, "(a)", "Constrained Domain ($N=4$)")

    draw_parcel_scene(
        ax,
        x1 + 0.025,
        0.445,
        0.505,
        0.425,
        footprints=[
            {
                "rect": (0.44, 0.44, 0.36, 0.36, -4.0),
                "must_be_inside": True,
                "fc": "#DCE8F2",
                "ec": C_NAVY,
                "ls": "-",
                "lw": 1.45,
                "alpha": 0.78,
                "markers": True,
                "ms": 11,
            }
        ],
        show_annotations=True,
    )

    # Formal mathematical specification block
    ax.text(
        x1 + 0.006,
        0.408,
        r"Feasible manifold $\mathcal{F}(\xi)$:",
        fontsize=7.7,
        fontweight="bold",
        color=C_INK,
        ha="left",
        va="top",
    )
    constraints_tex = [
        (r"$\bullet\; P \subset \mathcal{E}_{\mathrm{design}},\; C_{\mathrm{simple}}(P)$", "containment"),
        (r"$\bullet\; \mathrm{Area}(P) \in [A_{\min}, A_{\max}]$", "coverage"),
        (r"$\bullet\; W(P) \in [W_{\min}, W_{\max}]$", "tract span"),
        (r"$\bullet\; |\alpha_i - 90^\circ| \leq 5^\circ\; (\forall i)$", "orthogonality"),
    ]
    for i, (eq, desc) in enumerate(constraints_tex):
        cy = 0.354 - i * 0.048
        ax.text(x1 + 0.008, cy, eq, fontsize=7.2, color=C_INK, ha="left", va="center")
        ax.text(x1 + w1 - 0.004, cy, desc, fontsize=6.2, style="italic", color=C_SUBTLE, ha="right", va="center")

    # Proposition 2 mathematical callout (classic theorem bar on left)
    add_sharp_box(ax, x1 + 0.004, 0.048, w1 - 0.008, 0.114, fc="#FDF7F7", ec="#E5C5C5", lw=0.75, radius=0.004)
    ax.add_patch(Rectangle((x1 + 0.004, 0.048), 0.008, 0.114, facecolor=C_CRIMSON, edgecolor="none", zorder=3))
    ax.text(
        x1 + 0.022,
        0.136,
        "Proposition 2 (Analytical Bound):",
        fontsize=6.9,
        fontweight="bold",
        color=C_CRIMSON,
        ha="left",
        va="center",
        zorder=4,
    )
    ax.text(
        x1 + 0.022,
        0.099,
        r"$W_{\min}^2 \sin 85^\circ > A_{\max} \;\Rightarrow\; \mathcal{F}(\xi) = \emptyset$",
        fontsize=7.3,
        color=C_INK,
        ha="left",
        va="center",
        zorder=4,
    )
    ax.text(
        x1 + 0.022,
        0.066,
        r"Proves $17.7\%$ of draws infeasible pre-solve.",
        fontsize=6.1,
        color=C_SUBTLE,
        ha="left",
        va="center",
        zorder=4,
    )

    # =========================================================================
    # PANEL (b): Four-Stage Curriculum RL (CL-PPO)
    # =========================================================================
    add_panel_header(ax, x2, hdr_y, w2, "(b)", "Four-Stage Curriculum RL")

    ax.text(
        x2 + w2 / 2,
        0.865,
        r"Cumulative Sub-Manifolds ($\mathcal{F}_1 \supset \mathcal{F}_2 \supset \mathcal{F}_3 \supset \mathcal{F}_4$)",
        fontsize=6.5,
        style="italic",
        color=C_SUBTLE,
        ha="center",
        va="center",
    )

    stages = [
        ("Stage 1", "Containment", "MDP vertex moves; PPO clip", r"$D_{\mathrm{place}} = 0$", "#F3F6FA", C_BLUE),
        ("Stage 2", "+ Footprint Area", "Warm-start + VecNormalize", r"$D_{\mathrm{area}} = 0$", "#E6EEF6", C_BLUE),
        ("Stage 3", "+ Tract Width", "Warm-start + VecNormalize", r"$D_{\mathrm{width}} = 0$", "#D6E4F0", C_NAVY),
        ("Stage 4", "+ Orthogonality", "Warm-start + VecNormalize", r"$D_{\mathrm{angle}} = 0$", "#C6DAEC", C_NAVY),
    ]
    sw = w2 - 0.016
    sh = 0.088
    sy_top = 0.742
    for idx, (s_id, s_title, s_sub, s_math, bg, border) in enumerate(stages):
        sy_pos = sy_top - idx * (sh + 0.032)
        add_sharp_box(ax, x2 + 0.008, sy_pos, sw, sh, fc=bg, ec=border, lw=0.80, radius=0.005, z=2)
        ax.add_patch(Rectangle((x2 + 0.008, sy_pos), 0.009, sh, facecolor=border, edgecolor="none", zorder=3))
        ax.text(
            x2 + 0.028,
            sy_pos + sh * 0.65,
            f"{s_id}: {s_title}",
            fontsize=7.4,
            fontweight="bold",
            color=C_INK,
            ha="left",
            va="center",
            zorder=4,
        )
        ax.text(
            x2 + 0.028,
            sy_pos + sh * 0.27,
            s_sub,
            fontsize=6.0,
            style="italic",
            color=C_SUBTLE,
            ha="left",
            va="center",
            zorder=4,
        )
        ax.text(
            x2 + sw - 0.004,
            sy_pos + sh * 0.50,
            s_math,
            fontsize=7.2,
            color=C_NAVY,
            ha="right",
            va="center",
            zorder=4,
        )
        if idx < 3:
            add_arrow(
                ax,
                x2 + w2 / 2,
                sy_pos - 0.002,
                x2 + w2 / 2,
                sy_pos - 0.030,
                color=C_NAVY,
                lw=0.95,
                hl=4.4,
                hw=3.1,
            )

    # Shaping equation box
    add_sharp_box(ax, x2 + 0.008, 0.258, sw, 0.095, fc="#FFFFFF", ec="#9CA3AF", lw=0.75, radius=0.004)
    ax.text(
        x2 + w2 / 2,
        0.330,
        "Potential-Difference Progress Shaping:",
        fontsize=6.8,
        fontweight="bold",
        color=C_INK,
        ha="center",
        va="center",
    )
    ax.text(
        x2 + w2 / 2,
        0.288,
        r"$r_{c,t} = [\Phi_c(s_t) - \Phi_c(s_{t-1})]\cdot\mathbb{I}(D_{c,0}>0),\;\; R^{(c)} = \sum_{j=1}^c R_j$",
        fontsize=6.9,
        color=C_NAVY,
        ha="center",
        va="center",
    )

    # Empirical Ablation Mini-Table (booktabs style — zero overlap)
    ax.text(
        x2 + 0.010,
        0.230,
        "Curriculum Ablation (Solv. / Raw Feas.):",
        fontsize=6.8,
        fontweight="bold",
        color=C_INK,
        ha="left",
        va="center",
    )
    ax.plot([x2 + 0.008, x2 + w2 - 0.008], [0.210, 0.210], color=C_INK, lw=0.85)
    ablation_rows = [
        ("4-Stage CL-PPO", "80.8% / 66.0%", "67.3 steps", C_NAVY, True),
        ("w/o Ortho. (Stage 3)", "— / 0.1%", "non-ortho.", C_SUBTLE, False),
        ("1-Stage PPO (No-CL)", "0.1% / 0.1%", "trap (64M)", C_CRIMSON, False),
    ]
    for i, (m_lbl, m_val, m_note, col, bold) in enumerate(ablation_rows):
        ry = 0.182 - i * 0.044
        ax.text(x2 + 0.010, ry, m_lbl, fontsize=6.2, fontweight="bold" if bold else "normal", color=col, ha="left", va="center")
        ax.text(x2 + 0.352, ry, m_val, fontsize=6.3, fontweight="bold" if bold else "normal", color=col, ha="center", va="center")
        ax.text(x2 + w2 - 0.010, ry, m_note, fontsize=6.0, style="italic", color=C_SUBTLE, ha="right", va="center")
    ax.plot([x2 + 0.008, x2 + w2 - 0.008], [0.070, 0.070], color=C_INK, lw=0.85)
    ax.text(
        x2 + w2 / 2,
        0.049,
        r"$90\%$ conv. at $168.8\mathrm{M}$ steps (${\sim}0.05\,\mathrm{s}$ neural pass).",
        fontsize=6.0,
        color=C_SUBTLE,
        ha="center",
        va="center",
    )

    # =========================================================================
    # PANEL (c): Neuro-Numerical Refinement (Algorithm 1)
    # =========================================================================
    add_panel_header(ax, x3, hdr_y, w3, "(c)", "Neuro-Numerical Refinement")

    mid_x3 = x3 + w3 / 2

    # Node 1: Rollout
    add_sharp_box(ax, x3 + 0.008, 0.775, w3 - 0.016, 0.088, fc="#F8FAFC", ec=C_NAVY, lw=0.80)
    ax.text(
        mid_x3,
        0.835,
        r"Phase 1: Frozen CL-PPO Rollout ($T \leq 256$)",
        fontsize=7.0,
        fontweight="bold",
        color=C_INK,
        ha="center",
        va="center",
    )
    ax.text(
        mid_x3,
        0.798,
        r"Track lowest-cost simple state $x_{\mathrm{RL}}^*$",
        fontsize=6.3,
        color=C_SUBTLE,
        ha="center",
        va="center",
    )

    left_bx = x3 + 0.008
    bw_c = (w3 - 0.030) / 2.0
    right_bx = left_bx + bw_c + 0.014

    add_arrow(ax, left_bx + bw_c / 2, 0.773, left_bx + bw_c / 2, 0.728, color=C_GREEN, lw=0.95, hl=4.4, hw=3.1)
    add_arrow(ax, right_bx + bw_c / 2, 0.773, right_bx + bw_c / 2, 0.728, color=C_TEAL, lw=0.95, hl=4.4, hw=3.1)

    # Branch A: Direct Feasible (80.8%)
    add_sharp_box(ax, left_bx, 0.620, bw_c, 0.105, fc="#F2F9F5", ec=C_GREEN, lw=0.85)
    ax.text(left_bx + bw_c / 2, 0.697, r"$P_t \in \mathcal{F}(\xi)$ (80.8%)", fontsize=6.9, fontweight="bold", color=C_GREEN, ha="center", va="center")
    ax.text(left_bx + bw_c / 2, 0.661, "Return immediately", fontsize=6.3, color=C_INK, ha="center", va="center")
    ax.text(left_bx + bw_c / 2, 0.635, "(0 SLSQP calls)", fontsize=6.0, style="italic", color=C_SUBTLE, ha="center", va="center")

    # Branch B: Truncated Cat. B (19.2%)
    add_sharp_box(ax, right_bx, 0.620, bw_c, 0.105, fc="#F3F8F7", ec=C_TEAL, lw=0.85)
    ax.text(right_bx + bw_c / 2, 0.701, "Truncated (19.2%)", fontsize=6.9, fontweight="bold", color=C_TEAL, ha="center", va="center")
    ax.text(right_bx + bw_c / 2, 0.668, r"SLSQP on $J_{\mathrm{smooth}}(x)$", fontsize=6.3, color=C_INK, ha="center", va="center")
    ax.text(right_bx + bw_c / 2, 0.633, r"start $x_{\mathrm{RL}}^* \to \mathrm{round}$", fontsize=6.0, style="italic", color=C_SUBTLE, ha="center", va="center")

    add_arrow(ax, right_bx + bw_c / 2, 0.618, right_bx + bw_c / 2, 0.576, color=C_TEAL, lw=0.95, hl=4.4, hw=3.1)

    # Geometry diagram showing sub-grid truncation -> continuous refinement
    add_sharp_box(ax, x3 + 0.008, 0.140, w3 - 0.016, 0.434, fc="#FFFFFF", ec="#D1D5DB", lw=0.75)
    ax.text(
        mid_x3,
        0.551,
        r"Sub-Grid Residual Resolution ($N=4$)",
        fontsize=6.9,
        fontweight="bold",
        color=C_INK,
        ha="center",
        va="center",
    )

    _, _, fps_c = draw_parcel_scene(
        ax,
        x3 + 0.042,
        0.205,
        0.476,
        0.320,
        footprints=[
            {
                "pts": [[0.23, 0.23], [0.77, 0.18], [0.79, 0.60], [0.29, 0.65]],
                "must_be_inside": False,
                "fc": "none",
                "ec": C_CRIMSON,
                "ls": "--",
                "lw": 1.35,
                "markers": True,
                "ms": 10,
                "z": 4,
            },
            {
                "rect": (0.47, 0.44, 0.38, 0.38, -4.0),
                "must_be_inside": True,
                "fc": "#D9EFE2",
                "ec": C_GREEN,
                "ls": "-",
                "lw": 1.75,
                "alpha": 0.78,
                "markers": True,
                "ms": 12,
                "z": 5,
            },
        ],
        show_annotations=False,
    )
    red_pts, grn_pts = fps_c[0], fps_c[1]
    for k in [1, 2]:
        add_arrow(
            ax,
            red_pts[k, 0],
            red_pts[k, 1],
            grn_pts[k, 0],
            grn_pts[k, 1],
            color=C_INK,
            lw=0.75,
            hl=3.5,
            hw=2.4,
            z=8,
        )
    draw_right_angle_marker(ax, grn_pts[0], grn_pts[1], grn_pts[3], size=0.020, color=C_GREEN, lw=0.9)

    # Clean well-spaced legend inside geometry box
    ax.plot([x3 + 0.036, x3 + 0.080], [0.168, 0.168], color=C_CRIMSON, ls="--", lw=1.35, zorder=5)
    ax.text(x3 + 0.088, 0.168, r"$x_{\mathrm{RL}}^*$ (grid)", fontsize=6.4, color=C_CRIMSON, va="center")
    ax.plot([x3 + 0.278, x3 + 0.322], [0.168, 0.168], color=C_GREEN, ls="-", lw=1.75, zorder=5)
    ax.text(x3 + 0.330, 0.168, r"Refined ($P^* \in \mathcal{F}$)", fontsize=6.4, fontweight="bold", color=C_GREEN, va="center")

    # Summary footnote at bottom of Panel (c)
    ax.text(
        mid_x3,
        0.104,
        r"98.1% recovery (154/157; 90.9% from $x_{\mathrm{RL}}^*$)",
        fontsize=6.4,
        fontweight="bold",
        color=C_TEAL,
        ha="center",
        va="center",
    )
    ax.text(
        mid_x3,
        0.064,
        r"Smooth $\mathrm{float64}$ margins $\delta$ + strict discrete check.",
        fontsize=6.0,
        style="italic",
        color=C_SUBTLE,
        ha="center",
        va="center",
    )

    # =========================================================================
    # PANEL (d): Benchmark & Design Diversity
    # =========================================================================
    add_panel_header(ax, x4, hdr_y, w4, "(d)", "Benchmark & Design Diversity")

    # Top: True quantitative Pareto plot (Solvable Feasibility [%] vs Mean Evals, log scale)
    inset_x = (x4 + 0.082) / 2.5
    inset_y = 0.518
    inset_w = (w4 - 0.100) / 2.5
    inset_h = 0.340
    ax_p = fig.add_axes([inset_x, inset_y, inset_w, inset_h])
    ax_p.set_facecolor("#FFFFFF")

    # Subtle optimal-region highlight (upper-left corner)
    ax_p.axvspan(35, 220, ymin=0.77, ymax=1.0, facecolor="#ECFDF5", alpha=0.75, zorder=0)

    # Exact Table 2 benchmark metrics (1,000 unseen parcels; 817 solvable)
    pts_data = [
        ("CL-PPO + Ref.", 123.9, 99.6, C_GREEN, "*", 105, -2.0, 9.5, "center"),
        ("CL-PPO", 67.3, 80.8, C_NAVY, "o", 30, 0.0, -10.5, "center"),
        ("5x SLSQP", 356.9, 99.6, "#374151", "s", 24, 5.5, 6.5, "left"),
        ("CMA-ES", 367.6, 98.8, C_BLUE, "^", 28, 6.0, -8.5, "left"),
        ("Diff-Evol.", 9150.8, 100.0, "#6B7280", "D", 22, -2.0, -10.5, "center"),
        ("COBYLA", 4312.6, 46.9, "#9CA3AF", "v", 24, 0.0, -10.0, "center"),
        ("MC-Global", 34577.2, 58.3, "#9CA3AF", "P", 26, -3.0, 9.0, "right"),
        ("PPO No-CL", 79.0, 0.1, C_CRIMSON, "X", 30, 6.0, 2.5, "left"),
    ]

    ax_p.set_xscale("log")
    ax_p.set_xlim(32, 78000)
    ax_p.set_ylim(-10, 122)
    ax_p.grid(True, which="major", color="#E5E7EB", lw=0.55, ls=":")
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)

    for name, ev, feas, col, mk, sz, dx, dy, ha in pts_data:
        is_main = name == "CL-PPO + Ref."
        ax_p.scatter(
            ev,
            feas,
            c=col,
            marker=mk,
            s=sz,
            edgecolors="#FFFFFF" if not is_main else "#064E3B",
            linewidths=0.45 if not is_main else 0.65,
            zorder=6 if is_main else 4,
        )
        ax_p.annotate(
            name,
            xy=(ev, feas),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.6 if is_main else 5.3,
            fontweight="bold" if is_main else "normal",
            color=C_GREEN if is_main else (C_CRIMSON if "No-CL" in name else C_INK),
            ha=ha,
            va="center",
            zorder=7,
        )

    # Crisp thin arrow between 5x SLSQP and CL-PPO + Ref. (properly scaled)
    ax_p.annotate(
        "",
        xy=(148, 99.6),
        xytext=(318, 99.6),
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=0.85, mutation_scale=7.5),
        zorder=5,
    )
    ax_p.text(225, 89.0, r"$2.9\times$", fontsize=5.4, fontweight="bold", color=C_GREEN, ha="center", va="top")

    ax_p.set_xticks([100, 1000, 10000])
    ax_p.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$"], fontsize=6.0, color=C_SUBTLE)
    ax_p.set_yticks([0, 50, 100])
    ax_p.set_yticklabels(["0%", "50%", "100%"], fontsize=6.0, color=C_SUBTLE)
    ax_p.tick_params(axis="both", length=2.4, width=0.65, pad=1.5)
    ax_p.set_xlabel("Mean Steps / Evaluations (log scale)", fontsize=6.2, color=C_INK, labelpad=1.5)
    ax_p.set_ylabel("Solvable Feas.", fontsize=6.2, color=C_INK, labelpad=2.0)
    ax_p.set_title("1,000 Unseen Parcels (Paired Tests)", fontsize=6.7, fontweight="bold", color=C_INK, pad=2.5)

    # Compact statistical summary line under Pareto chart
    ax.text(
        x4 + w4 / 2,
        0.415,
        "99.6% feas.  |  0.0% area err  |  0.247 s  |  98–100% OOD",
        fontsize=5.9,
        fontweight="bold",
        color=C_GREEN,
        ha="center",
        va="center",
    )

    # Bottom: Stochastic Policy Multi-Option Synthesis (strictly 4-edge geometry)
    ax.plot([x4 + 0.008, x4 + w4 - 0.008], [0.388, 0.388], color="#E5E7EB", lw=0.75)
    ax.text(
        x4 + w4 / 2,
        0.362,
        r"Stochastic Sampling ($K=20$ Rollouts, Zero Retraining)",
        fontsize=6.5,
        fontweight="bold",
        color=C_INK,
        ha="center",
        va="center",
    )

    draw_parcel_scene(
        ax,
        x4 + 0.068,
        0.122,
        0.495,
        0.222,
        footprints=[
            {"rect": (0.37, 0.43, 0.28, 0.38, -4.0), "must_be_inside": True, "fc": C_NAVY, "ec": C_NAVY, "lw": 1.15, "alpha": 0.22, "z": 4},
            {"rect": (0.55, 0.44, 0.28, 0.40, -3.0), "must_be_inside": True, "fc": C_TEAL, "ec": C_TEAL, "lw": 1.15, "alpha": 0.22, "z": 5},
            {"rect": (0.45, 0.36, 0.42, 0.27, -3.5), "must_be_inside": True, "fc": C_AMBER, "ec": C_AMBER, "lw": 1.15, "alpha": 0.22, "z": 6},
            {"rect": (0.52, 0.54, 0.30, 0.36, 4.0), "must_be_inside": True, "fc": C_PURPLE, "ec": C_PURPLE, "lw": 1.15, "alpha": 0.22, "z": 7},
        ],
        show_annotations=False,
    )

    ax.text(
        x4 + w4 / 2,
        0.092,
        "9.9–18.8 spatially distinct layouts (19.0 unique tuples)",
        fontsize=6.2,
        fontweight="bold",
        color=C_INK,
        ha="center",
        va="center",
    )
    ax.text(
        x4 + w4 / 2,
        0.056,
        r"$94.8\%$ valid  |  $1 - \mathrm{IoU} = 0.335$  |  Shift $53.7\,\mathrm{px}$",
        fontsize=5.9,
        style="italic",
        color=C_SUBTLE,
        ha="center",
        va="center",
    )

    # Subtle inter-panel flow chevrons centered in the vertical gutters
    for x_arr in [x1 + w1 + gap / 2, x2 + w2 + gap / 2, x3 + w3 + gap / 2]:
        add_arrow(ax, x_arr - 0.010, 0.515, x_arr + 0.012, 0.515, color=C_RULE, lw=1.25, hl=4.8, hw=3.4, z=6)

    # Save PNG (3000 x 1200 px, 300 DPI), PDF (vector), and TIFF (300 DPI, LZW)
    png_path = OUTPUT_DIR / "graphical_abstract.png"
    pdf_path = OUTPUT_DIR / "graphical_abstract.pdf"
    tif_path = OUTPUT_DIR / "graphical_abstract.tif"

    fig.savefig(png_path, dpi=300, facecolor="#FFFFFF")
    fig.savefig(pdf_path, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)

    with Image.open(png_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(tif_path, compression="tiff_lzw", dpi=(300, 300))
        print(f"Saved Scientific Graphical Abstract: {rgb_img.size[0]}x{rgb_img.size[1]} px at 300 DPI")
        print(f"  PNG: {png_path}")
        print(f"  PDF: {pdf_path}")
        print(f"  TIF: {tif_path}")


if __name__ == "__main__":
    main()
