"""
run-comparative-eval.py
-----------------------
Comprehensive Parallelized Comparative Evaluation Benchmark.
Evaluates CL-PPO and Hybrid (CL-PPO + Polish) against ablation and classical
numerical optimization baselines (PPO No-CL, CMA-ES, SLSQP, COBYLA,
Differential Evolution, MC-Local, MC-Global) across N identical parcel seeds.

Usage:
    ./.venv/bin/python scripts/run-comparative-eval.py --n-episodes 10 --workers 2
    ./.venv/bin/python scripts/run-comparative-eval.py --n-episodes 1000 --workers 8 --load-cached-heavy
"""

import argparse
import json
import logging
import math
import multiprocessing
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import gymnasium as gym
import numpy as np
import shapely.vectorized
from shapely.geometry import Polygon, Point
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner
from urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT, INTERNAL_MAX_STEPS
from urban_planner.logger_config import setup_logging
from urban_planner.types import Stage

from scripts.baselines.run_cma_es import run_cma_es
from scripts.baselines.run_constrained_opt import run_slsqp, run_cobyla
from scripts.baselines.run_de import run_de
from scripts.baselines.run_hybrid_ppo import run_hybrid_ppo

logger = logging.getLogger("comparative_eval")

# Global variables for worker processes
raw_env = None
vec_env_cl = None
model_cl = None
vec_env_nocl = None
model_nocl = None


def init_worker():
    global raw_env, vec_env_cl, model_cl, vec_env_nocl, model_nocl

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    raw_env = gym.make("UrbanPlanner-v0", stage=4)

    model_path_cl = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model.zip"
    stats_path_cl = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model_vec_normalize.pkl"

    if model_path_cl.exists() and stats_path_cl.exists():
        dummy_cl = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
        vec_env_cl = VecNormalize.load(str(stats_path_cl), dummy_cl)
        vec_env_cl.training = False
        vec_env_cl.norm_reward = False
        model_cl = PPO.load(model_path_cl, env=vec_env_cl, device="cpu")
    else:
        model_cl = None
        vec_env_cl = None

    model_path_nocl = PROJECT_ROOT / "models" / "ppo_no_cl_1783091653" / "best_model.zip"
    stats_path_nocl = PROJECT_ROOT / "models" / "ppo_no_cl_1783091653" / "best_model_vec_normalize.pkl"

    if model_path_nocl.exists() and stats_path_nocl.exists():
        dummy_nocl = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
        vec_env_nocl = VecNormalize.load(str(stats_path_nocl), dummy_nocl)
        vec_env_nocl.training = False
        vec_env_nocl.norm_reward = False
        model_nocl = PPO.load(model_path_nocl, env=vec_env_nocl, device="cpu")
    else:
        model_nocl = None
        vec_env_nocl = None


def get_target_area_mean(plot) -> float:
    return (plot.area_range[0] + plot.area_range[1]) / 2.0


def check_self_intersection_fast(coords: List[Tuple[int, int]]) -> bool:
    if len(coords) < 4:
        return False
    p0, p1, p2, p3 = coords

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    return intersect(p0, p1, p2, p3) or intersect(p1, p2, p3, p0)


def get_area_fast(coords: List[Tuple[int, int]]) -> float:
    p0, p1, p2, p3 = coords
    return 0.5 * abs(
        p0[0] * p1[1] + p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p0[1]
        - (p0[1] * p1[0] + p1[1] * p2[0] + p2[1] * p3[0] + p3[1] * p0[0])
    )


def order_points_ccw(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(points) < 3:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


class PolygonPointSampler:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
        self.cache = []

    def get_points(self, num_points: int) -> List[Tuple[int, int]]:
        while len(self.cache) < num_points:
            batch_size = max(50000, num_points)
            xs = np.random.randint(int(self.min_x), int(self.max_x) + 1, size=batch_size)
            ys = np.random.randint(int(self.min_y), int(self.max_y) + 1, size=batch_size)
            mask = shapely.vectorized.contains(self.polygon, xs, ys)
            valid_xs = xs[mask]
            valid_ys = ys[mask]
            for x, y in zip(valid_xs, valid_ys):
                self.cache.append((int(x), int(y)))
        result = self.cache[:num_points]
        self.cache = self.cache[num_points:]
        return result


def check_angles_orthogonal_fast(coords: List[Tuple[int, int]], tolerance: float = 5.0) -> bool:
    for i in range(4):
        p_prev = coords[i - 1]
        p_curr = coords[i]
        p_next = coords[(i + 1) % 4]
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        len1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        len2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if len1 == 0 or len2 == 0:
            return False
        cos_val = max(-1.0, min(1.0, dot / (len1 * len2)))
        angle_deg = math.degrees(math.acos(cos_val))
        if abs(angle_deg - 90.0) > tolerance:
            return False
    return True


def run_ppo_agent(
    model,
    vec_env,
    raw_env,
    seed: int,
    max_steps: int = INTERNAL_MAX_STEPS,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    start_time = time.perf_counter()

    total_steps = 0
    violating_steps = 0
    solved = False

    for step in range(max_steps):
        batched_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        norm_obs = vec_env.normalize_obs(batched_obs)

        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action[0])
        total_steps += 1

        outside = any(dist > 0 for dist in unwrapped.assessment.outside_shape_distances)
        invalid = check_self_intersection_fast(unwrapped.assessment.current_outline_coords)
        if outside or invalid:
            violating_steps += 1

        if terminated and info.get("is_solution", False) and not invalid:
            solved = True
            break

        if terminated or truncated:
            break

    wall_clock_seconds = time.perf_counter() - start_time

    assessment = unwrapped.assessment
    target_mean = get_target_area_mean(unwrapped.plot)
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

    return {
        "solved": solved,
        "steps": total_steps,
        "area_error": area_error_pct,
        "violation_rate": (violating_steps / total_steps) * 100.0 if total_steps > 0 else 0.0,
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev,
        "wall_clock_seconds": wall_clock_seconds,
    }


def run_mc_local(
    raw_env,
    seed: int,
    max_steps: int = INTERNAL_MAX_STEPS,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    start_time = time.perf_counter()

    # Re-seed action space RNG for deterministic reproducibility per parcel seed
    raw_env.action_space.seed(seed)

    total_steps = 0
    violating_steps = 0
    solved = False

    for step in range(max_steps):
        action = raw_env.action_space.sample()
        obs, reward, terminated, truncated, info = raw_env.step(action)
        total_steps += 1

        outside = any(dist > 0 for dist in unwrapped.assessment.outside_shape_distances)
        invalid = check_self_intersection_fast(unwrapped.assessment.current_outline_coords)
        if outside or invalid:
            violating_steps += 1

        if terminated and info.get("is_solution", False) and not invalid:
            solved = True
            break

        if terminated or truncated:
            break

    wall_clock_seconds = time.perf_counter() - start_time

    assessment = unwrapped.assessment
    target_mean = get_target_area_mean(unwrapped.plot)
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

    return {
        "solved": solved,
        "steps": total_steps,
        "area_error": area_error_pct,
        "violation_rate": (violating_steps / total_steps) * 100.0 if total_steps > 0 else 0.0,
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev,
        "wall_clock_seconds": wall_clock_seconds,
    }


def run_mc_global(
    raw_env,
    seed: int,
    max_proposals: int = 100000,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    start_time = time.perf_counter()
    envelope = unwrapped.plot.permitted_outline
    sampler = PolygonPointSampler(envelope)

    total_proposals = 0
    solved = False

    points_pool = sampler.get_points(4 * max_proposals)
    min_area, max_area = unwrapped.plot.area_range

    for proposal in range(max_proposals):
        total_proposals += 1
        idx = proposal * 4
        points = points_pool[idx: idx + 4]
        if len(points) < 4:
            break

        final_coords = order_points_ccw(points)
        if check_self_intersection_fast(final_coords):
            continue

        area = int(get_area_fast(final_coords))
        if not (min_area <= area <= max_area):
            continue

        if not check_angles_orthogonal_fast(final_coords):
            continue

        unwrapped.outline.coords = final_coords
        unwrapped.outline.update_geometry()
        unwrapped.assessment.update(final_coords)

        if all(unwrapped.assessment.checklist.values()):
            solved = True
            break

    wall_clock_seconds = time.perf_counter() - start_time

    assessment = unwrapped.assessment
    target_mean = get_target_area_mean(unwrapped.plot)
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

    return {
        "solved": solved,
        "steps": total_proposals,
        "area_error": area_error_pct,
        "violation_rate": 0.0,
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev,
        "wall_clock_seconds": wall_clock_seconds,
    }


def run_episode_task(seed: int, methods: List[str]) -> Dict[str, Any]:
    global raw_env, vec_env_cl, model_cl, vec_env_nocl, model_nocl

    res_dict = {"seed": seed}

    if "cl_ppo" in methods:
        res_dict["CL-PPO"] = run_ppo_agent(model_cl, vec_env_cl, raw_env, seed=seed)

    if "hybrid_ppo" in methods:
        res_dict["CL-PPO + Polish"] = run_hybrid_ppo(model_cl, vec_env_cl, raw_env, seed=seed)

    if "ppo_no_cl" in methods:
        res_dict["PPO (No-CL)"] = run_ppo_agent(model_nocl, vec_env_nocl, raw_env, seed=seed)

    if "cma_es" in methods:
        res_dict["CMA-ES"] = run_cma_es(raw_env, seed=seed)

    if "slsqp" in methods:
        res_dict["SLSQP"] = run_slsqp(raw_env, seed=seed, num_starts=5)

    if "cobyla" in methods:
        res_dict["COBYLA"] = run_cobyla(raw_env, seed=seed, num_starts=10)

    if "de" in methods:
        res_dict["Diff-Evolution"] = run_de(raw_env, seed=seed)

    if "mc_local" in methods:
        res_dict["MC-Local"] = run_mc_local(raw_env, seed=seed)

    if "mc_global" in methods:
        res_dict["MC-Global"] = run_mc_global(raw_env, seed=seed, max_proposals=100000)

    return res_dict


def aggregate_results(
    results: List[Dict[str, Any]],
    solvable_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    n_episodes = len(results)
    if n_episodes == 0:
        return {}

    solved_episodes = [r for r in results if r["solved"]]
    success_rate = (len(solved_episodes) / n_episodes) * 100.0

    if solvable_mask is not None and sum(solvable_mask) > 0:
        solvable_results = [r for r, is_solv in zip(results, solvable_mask) if is_solv]
        solvable_solved = sum(1 for r in solvable_results if r["solved"])
        solvable_success_rate = (solvable_solved / len(solvable_results)) * 100.0
    else:
        solvable_success_rate = success_rate

    mean_steps_solved = float(np.mean([r["steps"] for r in solved_episodes])) if len(solved_episodes) > 0 else np.nan
    std_steps_solved = float(np.std([r["steps"] for r in solved_episodes])) if len(solved_episodes) > 0 else np.nan

    mean_time = float(np.mean([r.get("wall_clock_seconds", 0.0) for r in results]))
    std_time = float(np.std([r.get("wall_clock_seconds", 0.0) for r in results]))

    mean_area_error = float(np.mean([float(r["area_error"]) for r in results]))
    std_area_error = float(np.std([float(r["area_error"]) for r in results]))

    violation_rate = float(np.mean([float(r["violation_rate"]) for r in results]))
    width_feasibility = (sum(1 for r in results if r["width_feasible"]) / n_episodes) * 100.0

    mean_ortho_dev = float(np.mean([float(r["ortho_deviation"]) for r in results]))
    std_ortho_dev = float(np.std([float(r["ortho_deviation"]) for r in results]))

    return {
        "success_rate": success_rate,
        "solvable_success_rate": solvable_success_rate,
        "mean_steps_solved": mean_steps_solved,
        "std_steps_solved": std_steps_solved,
        "mean_time_seconds": mean_time,
        "std_time_seconds": std_time,
        "mean_area_error": mean_area_error,
        "std_area_error": std_area_error,
        "violation_rate": violation_rate,
        "width_feasibility": width_feasibility,
        "mean_ortho_dev": mean_ortho_dev,
        "std_ortho_dev": std_ortho_dev,
    }


def format_latex_table_v2(aggregated: Dict[str, Dict[str, Any]], n_episodes: int, n_solvable: int) -> str:
    methods = list(aggregated.keys())
    header_cols = " & ".join([f"\\textbf{{{m}}}" for m in methods])
    col_format = "l" + "c" * len(methods)

    def row_val(field: str, is_pct: bool = True, precision: int = 1) -> str:
        vals = []
        for m in methods:
            v = aggregated[m].get(field, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append("N/A")
            elif is_pct:
                vals.append(f"{v:.{precision}f}\\%")
            else:
                vals.append(f"{v:.{precision}f}")
        return " & ".join(vals)

    def row_steps() -> str:
        vals = []
        for m in methods:
            mean = aggregated[m].get("mean_steps_solved", None)
            std = aggregated[m].get("std_steps_solved", None)
            if mean is None or np.isnan(mean):
                vals.append("N/A")
            else:
                vals.append(f"{mean:.1f} $\\pm$ {std:.1f}")
        return " & ".join(vals)

    def row_time() -> str:
        vals = []
        for m in methods:
            t = aggregated[m].get("mean_time_seconds", 0.0)
            vals.append(f"{t:.3f} s")
        return " & ".join(vals)

    return (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{Comparative engineering and optimization performance metrics across {n_episodes} procedural parcel instances ({n_solvable} geometrically compatible).}}\n"
        "    \\label{tab:comparative_benchmarks_v2}\n"
        f"    \\begin{{tabular}}{{{col_format}}}\n"
        "        \\hline\n"
        f"        \\textbf{{Performance Metric}} & {header_cols} \\\\\n"
        "        \\hline\n"
        f"        \\textbf{{Feasibility Ratio (All $N={n_episodes}$) [\\%]}} & {row_val('success_rate')} \\\\\n"
        f"        \\textbf{{Feasibility Ratio (Solvable $N={n_solvable}$) [\\%]}} & {row_val('solvable_success_rate')} \\\\\n"
        f"        \\textbf{{Steps / Evals (Solved)}} & {row_steps()} \\\\\n"
        f"        \\textbf{{Mean Time / Instance}} & {row_time()} \\\\\n"
        f"        \\textbf{{Area Error [\\%]}} & {row_val('mean_area_error')} \\\\\n"
        f"        \\textbf{{Width Feasibility [\\%]}} & {row_val('width_feasibility')} \\\\\n"
        f"        \\textbf{{Orthogonality Dev. [$^\\circ$]}} & {row_val('mean_ortho_dev', is_pct=False, precision=2)} \\\\\n"
        f"        \\textbf{{Violation Rate [\\%]}} & {row_val('violation_rate')} \\\\\n"
        "        \\hline\n"
        "    \\end{tabular}\n"
        "\\end{table*}"
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Comprehensive Parallelized Benchmark Evaluation")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument(
        "--methods",
        type=str,
        default="cl_ppo,hybrid_ppo,ppo_no_cl,cma_es,slsqp,cobyla,de,mc_local,mc_global",
        help="Comma-separated list of methods to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/comparative_evaluation_v2_results.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--load-cached-heavy",
        action="store_true",
        help="Load precomputed CMA-ES, COBYLA, Diff-Evolution, MC-Global from existing v2 results if available",
    )

    args = parser.parse_args()
    n_episodes = args.n_episodes
    selected_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    num_workers = args.workers if args.workers else min(multiprocessing.cpu_count(), 8)

    method_key_map = {
        "cl_ppo": "CL-PPO",
        "hybrid_ppo": "CL-PPO + Polish",
        "ppo_no_cl": "PPO (No-CL)",
        "cma_es": "CMA-ES",
        "slsqp": "SLSQP",
        "cobyla": "COBYLA",
        "de": "Diff-Evolution",
        "mc_local": "MC-Local",
        "mc_global": "MC-Global",
    }

    cached_heavy_keys = {
        "cl_ppo": "CL-PPO",
        "ppo_no_cl": "PPO (No-CL)",
        "cma_es": "CMA-ES",
        "cobyla": "COBYLA",
        "de": "Diff-Evolution",
        "mc_local": "MC-Local",
        "mc_global": "MC-Global",
    }
    cached_results = {}

    out_path = PROJECT_ROOT / args.output
    if args.load_cached_heavy and out_path.exists():
        try:
            with open(out_path, "r") as f:
                prev_data = json.load(f).get("raw_results", {})
            for m_code, m_name in cached_heavy_keys.items():
                if m_code in selected_methods and m_name in prev_data and len(prev_data[m_name]) >= n_episodes:
                    cached_results[m_name] = prev_data[m_name][:n_episodes]
                    logger.info(f"Loaded {n_episodes} cached episodes for {m_name}.")
        except Exception as e:
            logger.warning(f"Could not load cached heavy results: {e}")

    methods_to_run = [m for m in selected_methods if method_key_map.get(m) not in cached_results]
    logger.info(
        f"Evaluating {n_episodes} episodes: running {methods_to_run} on {num_workers} workers "
        f"(cached: {list(cached_results.keys())})..."
    )

    results = {method_key_map[m]: [] for m in selected_methods if m in method_key_map}
    for m_name, m_list in cached_results.items():
        results[m_name] = m_list

    start_eval_time = time.perf_counter()

    if methods_to_run:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
            futures = [executor.submit(run_episode_task, seed, methods_to_run) for seed in range(n_episodes)]
            for i, fut in enumerate(futures):
                res_dict = fut.result()
                for k, val in res_dict.items():
                    if k != "seed" and k in results:
                        # Remove non-serializable or large assessment dicts before saving
                        val_clean = {
                            kk: vv
                            for kk, vv in val.items()
                            if kk not in ("assessment_checklist", "final_coords")
                        }
                        results[k].append(val_clean)
                if (i + 1) % max(1, n_episodes // 10) == 0 or (i + 1) == n_episodes:
                    logger.info(f"Evaluated {i+1}/{n_episodes} parcel instances...")

    total_eval_time = time.perf_counter() - start_eval_time
    logger.info(f"Completed evaluation in {total_eval_time:.2f}s total wall-clock time.")

    # Determine solvable mask (instances solved by at least one method)
    solvable_mask = [
        any(results[m][idx]["solved"] for m in results if len(results[m]) > idx)
        for idx in range(n_episodes)
    ]
    n_solvable = sum(solvable_mask)
    logger.info(f"Geometrically solvable instances across {n_episodes} seeds: {n_solvable}/{n_episodes} ({n_solvable/n_episodes*100:.1f}%)")

    # Aggregation
    aggregated = {}
    for method, res_list in results.items():
        agg = aggregate_results(res_list, solvable_mask=solvable_mask)
        aggregated[method] = agg

        logger.info(f"\n=================== {method} ===================")
        logger.info(f"Feasibility Ratio (All N={n_episodes}): {agg.get('success_rate', 0.0):.2f}%")
        logger.info(f"Feasibility Ratio (Solvable N={n_solvable}): {agg.get('solvable_success_rate', 0.0):.2f}%")
        logger.info(f"Mean Steps to Solution: {agg.get('mean_steps_solved', 0.0):.2f} +/- {agg.get('std_steps_solved', 0.0):.2f}")
        logger.info(f"Mean Time / Instance: {agg.get('mean_time_seconds', 0.0):.4f}s")
        logger.info(f"Mean Area Error: {agg.get('mean_area_error', 0.0):.2f}%")
        logger.info(f"Width Feasibility Rate: {agg.get('width_feasibility', 0.0):.2f}%")
        logger.info(f"Mean Ortho Deviation: {agg.get('mean_ortho_dev', 0.0):.2f}°")

    # Generate LaTeX table
    latex_table = format_latex_table_v2(aggregated, n_episodes, n_solvable)
    print("\n" + latex_table + "\n")

    latex_out = PROJECT_ROOT / "logs" / "comparative_evaluation_v2_table.tex"
    with open(latex_out, "w") as f:
        f.write(latex_table)
    logger.info(f"LaTeX table saved to: {latex_out}")

    serializable = {
        "metadata": {
            "n_episodes": n_episodes,
            "n_solvable": n_solvable,
            "methods": selected_methods,
            "total_wall_clock_seconds": total_eval_time,
        },
        "aggregated": aggregated,
        "raw_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=4)
    logger.info(f"Detailed JSON results saved to: {out_path}")


if __name__ == "__main__":
    main()
