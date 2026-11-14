"""
objective.py
------------
Smooth, continuous objective/cost function for classical numerical optimization baselines
(CMA-ES, SLSQP, COBYLA, Differential Evolution) and Hybrid CL-PPO + Local Polish.

Evaluates candidate 4-vertex footprint proposals against parcel constraints
defined in UrbanPlannerEnv (OutlineAssessment).
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from shapely.geometry import Polygon, Point, LineString

from urban_planner.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    TARGET_ANGLE_DEGREES,
    ANGLE_TOLERANCE_DEGREES,
)
from urban_planner.utils.geometry_utils import (
    calculate_rectangular_width,
    calculate_polygon_angles,
    distance_to_polygon,
)


def order_points_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Orders a list of 2D points counter-clockwise around their centroid.
    Ensures consistent vertex ordering for polygon calculations.
    """
    if len(points) < 3:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    return sorted(points, key=angle)


def _pt_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Fast pure-float distance from point (px, py) to segment (ax, ay)-(bx, by)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    if t <= 0.0:
        return math.hypot(px - ax, py - ay)
    if t >= 1.0:
        return math.hypot(px - bx, py - by)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _fast_rectangular_width(coords: List[Tuple[float, float]]) -> float:
    """Fast pure-float implementation of calculate_rectangular_width."""
    p0, p1, p2, p3 = coords
    # Opposite edges: edge0=(p0,p1) vs edge2=(p2,p3); edge1=(p1,p2) vs edge3=(p3,p0)
    avg1 = 0.5 * (
        _pt_seg_dist(p0[0], p0[1], p2[0], p2[1], p3[0], p3[1])
        + _pt_seg_dist(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    )
    avg2 = 0.5 * (
        _pt_seg_dist(p2[0], p2[1], p0[0], p0[1], p1[0], p1[1])
        + _pt_seg_dist(p3[0], p3[1], p0[0], p0[1], p1[0], p1[1])
    )
    first_min = avg1 if avg1 < avg2 else avg2

    avg3 = 0.5 * (
        _pt_seg_dist(p1[0], p1[1], p3[0], p3[1], p0[0], p0[1])
        + _pt_seg_dist(p2[0], p2[1], p3[0], p3[1], p0[0], p0[1])
    )
    avg4 = 0.5 * (
        _pt_seg_dist(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1])
        + _pt_seg_dist(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
    )
    second_min = avg3 if avg3 < avg4 else avg4
    return first_min if first_min < second_min else second_min


def _segments_intersect(
    A: Tuple[float, float], B: Tuple[float, float],
    C: Tuple[float, float], D: Tuple[float, float],
) -> bool:
    """Fast pure-float segment intersection test for non-adjacent edges."""
    def ccw(P1, P2, P3):
        return (P3[1] - P1[1]) * (P2[0] - P1[0]) > (P2[1] - P1[1]) * (P3[0] - P1[0])

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def check_self_intersection_smooth(coords: List[Tuple[float, float]]) -> float:
    """
    Computes a continuous penalty for self-intersecting or degenerate quadrilaterals.
    Returns 0.0 if the polygon is strictly simple and non-degenerate.
    """
    if len(coords) < 4:
        return 5000.0

    p0, p1, p2, p3 = coords
    l0 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    l1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    l2 = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    l3 = math.hypot(p0[0] - p3[0], p0[1] - p3[1])
    min_edge_len = min(l0, l1, l2, l3)

    penalty = 0.0
    if min_edge_len < 5.0:
        penalty += 500.0 * (5.0 - min_edge_len)

    # Opposite edges (0-1 with 2-3) and (1-2 with 3-0)
    if _segments_intersect(p0, p1, p2, p3):
        penalty += 1000.0 + l0 + l2

    if _segments_intersect(p1, p2, p3, p0):
        penalty += 1000.0 + l1 + l3

    return penalty


def get_signed_area(coords: List[Tuple[float, float]]) -> float:
    """Computes signed polygon area using the Shoelace formula (positive for CCW)."""
    if len(coords) < 3:
        return 0.0
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return 0.5 * area


def calculate_footprint_cost(
    x: np.ndarray,
    env_unwrapped: Any,
    weights: Optional[Dict[str, float]] = None,
    reorder_ccw: bool = False,
) -> float:
    """
    Evaluates candidate coordinate vector x in continuous floating-point space
    and returns a smooth scalar cost >= 0.0.
    
    Avoids integer rounding inside the cost evaluation so that finite-difference
    gradients (e.g. in SLSQP) remain non-zero and smooth across all constraints.
    Includes a small interior margin (~1 px / 0.5 deg) so that continuous optimum (cost=0)
    remains strictly feasible after integer coordinate rounding in OutlineAssessment.

    Args:
        x: Array of shape (8,) containing [x0, y0, x1, y1, x2, y2, x3, y3]
        env_unwrapped: The unwrapped UrbanPlannerEnv instance containing plot and assessment
        weights: Optional dictionary overriding default constraint weights
        reorder_ccw: Whether to sort vertices CCW dynamically (False preserves variable index identity)

    Returns:
        float: Total scalar cost >= 0 (0.0 represents a fully compliant interior solution).
    """
    if weights is None:
        weights = {
            "placement": 15.0,
            "area": 0.05,
            "width": 8.0,
            "angle": 4.0,
            "intersection": 2.0,
        }

    # 1. Soft penalty if outside window bounds, plus clip
    out_of_window_penalty = 0.0
    for i in range(0, 8, 2):
        if x[i] < 0.0:
            out_of_window_penalty += -x[i]
        elif x[i] > WINDOW_WIDTH:
            out_of_window_penalty += x[i] - WINDOW_WIDTH
        if x[i + 1] < 0.0:
            out_of_window_penalty += -x[i + 1]
        elif x[i + 1] > WINDOW_HEIGHT:
            out_of_window_penalty += x[i + 1] - WINDOW_HEIGHT

    x_clipped = np.clip(x[0::2], 0.0, float(WINDOW_WIDTH))
    y_clipped = np.clip(x[1::2], 0.0, float(WINDOW_HEIGHT))
    coords = [(float(px), float(py)) for px, py in zip(x_clipped, y_clipped)]

    if reorder_ccw:
        coords = order_points_ccw(coords)

    # 2. Self-intersection check (smooth)
    intersection_penalty = check_self_intersection_smooth(coords)
    signed_area = get_signed_area(coords)
    area = abs(signed_area)

    if intersection_penalty > 0:
        return float(
            5000.0
            + weights["intersection"] * intersection_penalty
            + out_of_window_penalty * 10.0
        )

    # 3. Placement (distance to buildable envelope + 1.0 px interior margin against int rounding)
    permitted_polygon = env_unwrapped.plot.permitted_outline
    ext_ring = permitted_polygon.exterior
    placement_cost = 0.0
    interior_margin_px = 1.0
    for pt in coords:
        pt_geom = Point(pt)
        b_dist = ext_ring.distance(pt_geom)
        if not permitted_polygon.contains(pt_geom):
            placement_cost += b_dist + interior_margin_px
        elif b_dist < interior_margin_px:
            placement_cost += (interior_margin_px - b_dist)

    # 4. Area difference (in float, with a small interior margin for integer rounding stability)
    min_area, max_area = env_unwrapped.plot.area_range
    area_margin = min(100.0, 0.05 * (max_area - min_area))
    eff_min_area = min_area + area_margin
    eff_max_area = max_area - area_margin
    if area < eff_min_area:
        area_diff = eff_min_area - area
    elif area > eff_max_area:
        area_diff = area - eff_max_area
    else:
        area_diff = 0.0

    # 5. Tract width difference (in float! Do NOT round coords to int here)
    width = _fast_rectangular_width(coords)
    min_width, max_width = env_unwrapped.plot.width_range
    width_margin = min(1.5, 0.05 * max(1.0, max_width - min_width))
    eff_min_width = min_width + width_margin
    eff_max_width = max_width - width_margin
    if width < eff_min_width:
        width_diff = eff_min_width - width
    elif width > eff_max_width:
        width_diff = width - eff_max_width
    else:
        width_diff = 0.0

    # 6. Corner angles (in float! Smooth orthogonality to 90 degrees with interior tolerance)
    angles = calculate_polygon_angles(coords)
    eff_angle_tol = max(1.0, ANGLE_TOLERANCE_DEGREES - 0.5)
    angle_cost = 0.0
    for angle in angles:
        dev = abs(angle - TARGET_ANGLE_DEGREES)
        if dev > eff_angle_tol:
            angle_cost += (dev - eff_angle_tol)

    total_cost = (
        weights["placement"] * placement_cost
        + weights["area"] * area_diff
        + weights["width"] * width_diff
        + weights["angle"] * angle_cost
        + out_of_window_penalty * 10.0
    )

    return float(total_cost)


def evaluate_final_solution(
    x: np.ndarray,
    env_unwrapped: Any,
    reorder_ccw: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a candidate solution coordinate vector on the environment's discrete
    OutlineAssessment state, returning benchmark metrics.

    Args:
        x: Array of shape (8,) containing [x0, y0, x1, y1, x2, y2, x3, y3]
        env_unwrapped: The unwrapped UrbanPlannerEnv instance
        reorder_ccw: Whether to reorder vertices CCW before evaluation

    Returns:
        Dict containing solved flag, area_error, width_feasible, ortho_deviation, etc.
    """
    x_clipped = np.clip(x[0::2], 0.0, float(WINDOW_WIDTH))
    y_clipped = np.clip(x[1::2], 0.0, float(WINDOW_HEIGHT))
    raw_coords = list(zip(x_clipped, y_clipped))
    if reorder_ccw:
        ordered_coords = order_points_ccw(raw_coords)
    else:
        ordered_coords = raw_coords
    int_coords = [(int(round(c[0])), int(round(c[1]))) for c in ordered_coords]

    # Update actual env state
    env_unwrapped.outline.coords = int_coords
    env_unwrapped.outline.update_geometry()
    env_unwrapped.assessment.update(int_coords)

    assessment = env_unwrapped.assessment
    is_self_intersecting = check_self_intersection_smooth(int_coords) > 0.0
    is_solution = all(assessment.checklist.values()) and not is_self_intersecting

    target_mean = (env_unwrapped.plot.area_range[0] + env_unwrapped.plot.area_range[1]) / 2.0
    area_error_pct = (
        (abs(assessment.area_difference) / target_mean) * 100.0
        if assessment.area_difference != 0
        else 0.0
    )
    width_feasible = assessment.width_difference == 0
    angles = assessment.current_angles
    ortho_dev = (
        sum(abs(a - 90.0) for a in angles) / len(angles) if len(angles) > 0 else 90.0
    )

    outside = any(dist > 0 for dist in assessment.outside_shape_distances)
    violation_rate = 100.0 if (outside or is_self_intersecting) else 0.0

    return {
        "solved": is_solution,
        "area_error": area_error_pct,
        "violation_rate": violation_rate,
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev,
        "assessment_checklist": assessment.checklist.copy(),
        "final_coords": int_coords,
    }


if __name__ == "__main__":
    import gymnasium as gym
    import urban_planner

    print("--- Running Smooth Objective Function & Gradient Smoke Test on 5 Parcels ---")
    raw_env = gym.make("UrbanPlanner-v0", stage=4)

    for seed in range(1, 6):
        obs, info = raw_env.reset(seed=seed)
        unwrapped = raw_env.unwrapped

        # Generate a synthetic non-orthogonal quadrilateral inside the permitted bounds
        poly = unwrapped.plot.permitted_outline
        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        w = min(120.0, (maxx - minx) * 0.3)
        h = min(120.0, (maxy - miny) * 0.3)
        test_x = np.array(
            [
                cx - w / 2, cy - h / 2,
                cx + w / 2, cy - h / 2 + 15.0,
                cx + w / 2 - 10.0, cy + h / 2,
                cx - w / 2, cy + h / 2,
            ],
            dtype=float,
        )

        cost_val = calculate_footprint_cost(test_x, unwrapped)
        eval_res = evaluate_final_solution(test_x, unwrapped)

        # Verify finite-difference gradient is non-zero (crucial for SLSQP)
        eps = 1e-6
        grad = np.zeros(8)
        for idx in range(8):
            x_plus = test_x.copy()
            x_plus[idx] += eps
            grad[idx] = (calculate_footprint_cost(x_plus, unwrapped) - cost_val) / eps

        grad_norm = float(np.linalg.norm(grad))
        print(
            f"Seed {seed}: Cost = {cost_val:.2f} | Grad Norm = {grad_norm:.4f} | "
            f"Ortho Dev = {eval_res['ortho_deviation']:.1f}° | Solved = {eval_res['solved']}"
        )
        assert cost_val >= 0.0, "Cost must be non-negative"
        assert grad_norm > 0.0, "Finite-difference gradient must be non-zero for non-optimal polygon"

    print("Smoke test passed! Smooth objective function and gradients verified.")
