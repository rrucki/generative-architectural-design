"""
run_hybrid_ppo.py
-----------------
Amortized Neural Warm-Start + Continuous Local Polish (CL-PPO + Polish).

Operationalizes the amortized neural warm-start + local refinement concept (Section 4.6):
1. Executes the trained CL-PPO policy for up to T=256 steps to rapidly navigate
   the non-convex parcel geometry into the feasible basin of attraction.
2. If the discrete action space of CL-PPO leaves residual constraint violations
   at truncation, extracts the best trajectory state and applies a fast continuous
   local refinement (SLSQP) to resolve sub-grid discretization residuals.
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
from scipy.optimize import minimize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner
from urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT, INTERNAL_MAX_STEPS
from scripts.baselines.objective import (
    calculate_footprint_cost,
    evaluate_final_solution,
    check_self_intersection_smooth,
    _segments_intersect,
)


def _fast_assessment_score(assessment: Any) -> float:
    """O(1) proxy score using already-computed OutlineAssessment attributes."""
    return (
        15.0 * sum(assessment.outside_shape_distances)
        + 0.05 * abs(assessment.area_difference)
        + 8.0 * assessment.width_difference
        + 4.0 * sum(assessment.angle_differences)
    )


def _is_self_intersecting_fast(coords: List[Tuple[int, int]]) -> bool:
    if len(coords) < 4:
        return True
    p0, p1, p2, p3 = coords
    return _segments_intersect(p0, p1, p2, p3) or _segments_intersect(p1, p2, p3, p0)


def run_hybrid_ppo(
    model: Any,
    vec_env: Any,
    raw_env: Any,
    seed: int,
    max_ppo_steps: int = INTERNAL_MAX_STEPS,
    max_polish_evals: int = 1600,
) -> Dict[str, Any]:
    """
    Runs CL-PPO followed by conditional continuous local refinement (SLSQP)
    when the discrete policy truncates near the feasibility boundary.
    """
    random.seed(seed)
    np.random.seed(seed)

    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    start_time = time.perf_counter()

    total_ppo_steps = 0
    violating_steps = 0
    solved_by_ppo = False

    best_traj_score = float("inf")
    best_traj_coords = list(unwrapped.outline.coords)

    for step in range(max_ppo_steps):
        batched_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        norm_obs = vec_env.normalize_obs(batched_obs)

        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action[0])
        total_ppo_steps += 1

        a = unwrapped.assessment
        coords = a.current_outline_coords
        outside = any(dist > 0 for dist in a.outside_shape_distances)
        invalid = _is_self_intersecting_fast(coords)
        if outside or invalid:
            violating_steps += 1

        if not invalid:
            score = _fast_assessment_score(a)
            if score < best_traj_score:
                best_traj_score = score
                best_traj_coords = list(coords)

        if terminated and info.get("is_solution", False) and not invalid:
            solved_by_ppo = True
            best_traj_coords = list(coords)
            break

        if terminated or truncated:
            break

    if solved_by_ppo:
        wall_clock_seconds = time.perf_counter() - start_time
        a = unwrapped.assessment
        angles = a.current_angles
        ortho_dev = sum(abs(ang - 90.0) for ang in angles) / len(angles) if angles else 90.0
        return {
            "solved": True,
            "steps": total_ppo_steps,
            "ppo_steps": total_ppo_steps,
            "polish_evals": 0,
            "area_error": 0.0,
            "violation_rate": (violating_steps / total_ppo_steps) * 100.0 if total_ppo_steps > 0 else 0.0,
            "width_feasible": True,
            "ortho_deviation": ortho_dev,
            "wall_clock_seconds": wall_clock_seconds,
            "assessment_checklist": a.checklist.copy(),
            "final_coords": best_traj_coords,
        }

    polish_evals = 0
    best_x = np.array([float(c) for pt in best_traj_coords for c in pt], dtype=float)
    final_eval = evaluate_final_solution(best_x, unwrapped)

    # Necessary geometric condition for orthogonal quadrilateral feasibility:
    # Since width = min(d1, d2) and area ~ d1 * d2 >= width^2 * sin(85 deg),
    # if w_min^2 * sin(85 deg) > a_max, the random generator drew incompatible targets.
    w_min, w_max = unwrapped.plot.width_range
    a_min, a_max = unwrapped.plot.area_range
    geometrically_compatible = (w_min ** 2) * math.sin(math.radians(85.0)) <= a_max
    bounds = [(0.0, float(WINDOW_WIDTH)), (0.0, float(WINDOW_HEIGHT))] * 4

    def smooth_cost(x: np.ndarray) -> float:
        nonlocal polish_evals
        polish_evals += 1
        return calculate_footprint_cost(x, unwrapped)

    if not final_eval["solved"] and geometrically_compatible:
        # Anchor candidate initializations at CL-PPO's best trajectory state and envelope interior
        cx = float(np.mean(best_x[0::2]))
        cy = float(np.mean(best_x[1::2]))
        env_poly = unwrapped.plot.permitted_outline
        minx, miny, maxx, maxy = env_poly.bounds
        env_cx, env_cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
        rep_pt = env_poly.representative_point()

        target_mid = 0.5 * (a_min + a_max)
        max_valid_w = min(float(w_max), math.sqrt(float(a_max)))
        w0 = 0.5 * (float(w_min) + max_valid_w)
        h0 = max(w0, target_mid / max(50.0, w0))

        starts = [best_x]
        centers = [
            (cx, cy),
            (rep_pt.x, rep_pt.y),
            (env_cx, env_cy),
            (0.5 * (cx + rep_pt.x), 0.5 * (cy + rep_pt.y)),
        ]
        for center_x, center_y in centers:
            for angle_deg in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 65.0):
                rad = math.radians(angle_deg)
                ca, sa = math.cos(rad), math.sin(rad)
                dx_w, dy_w = (w0 / 2.0) * ca, (w0 / 2.0) * sa
                dx_h, dy_h = -(h0 / 2.0) * sa, (h0 / 2.0) * ca
                rect_guess = np.array(
                    [
                        center_x - dx_w - dx_h, center_y - dy_w - dy_h,
                        center_x + dx_w - dx_h, center_y + dy_w - dy_h,
                        center_x + dx_w + dx_h, center_y + dy_w + dy_h,
                        center_x - dx_w + dx_h, center_y - dy_w + dy_h,
                    ],
                    dtype=float,
                )
                starts.append(rect_guess)

        best_polish_cost = calculate_footprint_cost(best_x, unwrapped)
        for x0 in starts:
            if polish_evals >= max_polish_evals:
                break
            try:
                res = minimize(
                    smooth_cost,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": 35, "ftol": 1e-4},
                )
                ev = evaluate_final_solution(res.x, unwrapped)
                if ev["solved"]:
                    final_eval = ev
                    break
                if res.fun < best_polish_cost:
                    best_polish_cost = float(res.fun)
                    final_eval = ev
            except Exception:
                continue
    elif not final_eval["solved"]:
        # Over-constrained parcel (w_min^2 > a_max): perform a single fast 12-iteration
        # local polish from CL-PPO's best trajectory state to minimize residual error.
        try:
            res = minimize(
                smooth_cost,
                best_x,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 12, "ftol": 1e-3},
            )
            ev = evaluate_final_solution(res.x, unwrapped)
            if ev["area_error"] + ev["ortho_deviation"] < final_eval["area_error"] + final_eval["ortho_deviation"]:
                final_eval = ev
        except Exception:
            pass

    wall_clock_seconds = time.perf_counter() - start_time
    final_eval["steps"] = total_ppo_steps + polish_evals
    final_eval["ppo_steps"] = total_ppo_steps
    final_eval["polish_evals"] = polish_evals
    final_eval["violation_rate"] = (
        (violating_steps / total_ppo_steps) * 100.0 if total_ppo_steps > 0 else 0.0
    )
    final_eval["wall_clock_seconds"] = wall_clock_seconds
    return final_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Hybrid CL-PPO + Local Polish")
    parser.add_argument("--test-parcels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(1)

    raw_env = gym.make("UrbanPlanner-v0", stage=4)
    model_path = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model.zip"
    stats_path = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model_vec_normalize.pkl"

    dummy = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_env = VecNormalize.load(str(stats_path), dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")

    solved = 0
    times = []
    area_errs = []
    ortho_devs = []
    steps_list = []

    for i in range(args.test_parcels):
        s = args.seed + i
        res = run_hybrid_ppo(model, vec_env, raw_env, seed=s)
        if res["solved"]:
            solved += 1
            steps_list.append(res["steps"])
        times.append(res["wall_clock_seconds"])
        area_errs.append(res["area_error"])
        ortho_devs.append(res["ortho_deviation"])

    print(
        f"Hybrid CL-PPO + Polish ({args.test_parcels} parcels): "
        f"Feasibility = {solved / args.test_parcels * 100:.1f}% | "
        f"Mean Time = {np.mean(times):.4f}s | "
        f"Mean Steps = {np.mean(steps_list):.1f} | "
        f"Mean Area Err = {np.mean(area_errs):.2f}% | "
        f"Mean Ortho Dev = {np.mean(ortho_devs):.2f}°"
    )
