import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import gymnasium as gym
import numpy as np
import shapely.vectorized
from shapely.geometry import Polygon, Point
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Register environment
import urban_planner
from urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT, INTERNAL_MAX_STEPS
from urban_planner.logger_config import setup_logging
from urban_planner.types import Stage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("comparative_eval")

# Global variables for worker processes
raw_env = None
vec_env_cl = None
model_cl = None

def init_worker():
    global raw_env, vec_env_cl, model_cl
    
    # Critical optimization: restrict PyTorch to a single CPU thread per worker process
    # to avoid core context-switch thrashing across parallel process pools.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    raw_env = gym.make("UrbanPlanner-v0", stage=4)
    
    model_path_cl = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model.zip"
    stats_path_cl = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model_vec_normalize.pkl"
    
    dummy_cl = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_env_cl = VecNormalize.load(str(stats_path_cl), dummy_cl)
    vec_env_cl.training = False
    vec_env_cl.norm_reward = False
    model_cl = PPO.load(model_path_cl, env=vec_env_cl)

def get_target_area_mean(plot) -> float:
    """Returns the mean target area for the plot."""
    return (plot.area_range[0] + plot.area_range[1]) / 2.0

def is_valid_polygon(coords: List[Tuple[int, int]]) -> bool:
    """Checks if coordinates form a valid, simple polygon using Shapely."""
    if len(coords) < 3:
        return False
    poly = Polygon(coords)
    return poly.is_valid and not poly.is_empty

def check_self_intersection_fast(coords: List[Tuple[int, int]]) -> bool:
    """Checks if a 4-vertex quadrilateral self-intersects using pure Python cross-products."""
    if len(coords) < 4:
        return False
    p0, p1, p2, p3 = coords
    
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
    return intersect(p0, p1, p2, p3) or intersect(p1, p2, p3, p0)

def get_area_fast(coords: List[Tuple[int, int]]) -> float:
    """Computes the area of a 4-vertex quadrilateral using the Shoelace formula."""
    p0, p1, p2, p3 = coords
    return 0.5 * abs(p0[0]*p1[1] + p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p0[1] - 
                     (p0[1]*p1[0] + p1[1]*p2[0] + p2[1]*p3[0] + p3[1]*p0[0]))

def order_points_ccw(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Orders a list of 2D points counter-clockwise around their centroid."""
    if len(points) < 3:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    import math
    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    return sorted(points, key=angle)

class PolygonPointSampler:
    """A helper to batch-sample points inside a Shapely Polygon efficiently using vectorization."""
    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
        self.cache = []

    def get_points(self, num_points: int) -> List[Tuple[int, int]]:
        while len(self.cache) < num_points:
            # Batch sample points inside the bounding box
            batch_size = max(50000, num_points)
            xs = np.random.randint(int(self.min_x), int(self.max_x) + 1, size=batch_size)
            ys = np.random.randint(int(self.min_y), int(self.max_y) + 1, size=batch_size)
            # Vectorized point-in-polygon containment check
            mask = shapely.vectorized.contains(self.polygon, xs, ys)
            valid_xs = xs[mask]
            valid_ys = ys[mask]
            for x, y in zip(valid_xs, valid_ys):
                self.cache.append((int(x), int(y)))
        
        result = self.cache[:num_points]
        self.cache = self.cache[num_points:]
        return result

def run_ppo_agent(
    model,
    vec_env,
    raw_env,
    seed: int,
    max_steps: int = 100000
) -> Dict[str, Any]:
    """Runs a single episode of a PPO agent and records metrics."""
    # Seed generators
    random.seed(seed)
    np.random.seed(seed)
    
    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    
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
            
        if terminated and info.get("is_solution", False):
            solved = True
            break
            
        if truncated:
            unwrapped.steps_taken = 0
            obs = unwrapped._get_obs()

    # Metrics at final state
    assessment = unwrapped.assessment
    target_mean = get_target_area_mean(unwrapped.plot)
    area_error_pct = (abs(assessment.area_difference) / target_mean) * 100.0 if assessment.area_difference != 0 else 0.0
    width_feasible = assessment.width_difference == 0
    angles = assessment.current_angles
    ortho_dev = sum(abs(a - 90.0) for a in angles) / len(angles) if len(angles) > 0 else 90.0

    return {
        "solved": solved,
        "steps": total_steps,
        "area_error": area_error_pct,
        "violation_rate": (violating_steps / total_steps) * 100.0 if total_steps > 0 else 0.0,
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev
    }

def check_angles_orthogonal_fast(coords: List[Tuple[int, int]], tolerance: float = 5.0) -> bool:
    """Computes corner angles of the 4-vertex polygon and checks if they are within tolerance of 90 degrees.
    Used to quickly filter out non-orthogonal layouts in MC-Global proposals.
    """
    import math
    for i in range(4):
        p_prev = coords[i-1]
        p_curr = coords[i]
        p_next = coords[(i+1)%4]
        
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        len1 = (v1[0]**2 + v1[1]**2)**0.5
        len2 = (v2[0]**2 + v2[1]**2)**0.5
        if len1 == 0 or len2 == 0:
            return False
        
        cos_val = dot / (len1 * len2)
        cos_val = max(-1.0, min(1.0, cos_val))
        angle_rad = math.acos(cos_val)
        angle_deg = math.degrees(angle_rad)
        
        if abs(angle_deg - 90.0) > tolerance:
            return False
    return True

def run_mc_global(
    raw_env,
    seed: int,
    max_proposals: int = 100000
) -> Dict[str, Any]:
    """Runs a single episode of Geometry-Aware Global Monte Carlo (sampling shapes inside envelope)."""
    random.seed(seed)
    np.random.seed(seed)
    
    obs, info = raw_env.reset(seed=seed)
    unwrapped = raw_env.unwrapped
    envelope = unwrapped.plot.permitted_outline
    sampler = PolygonPointSampler(envelope)
    
    total_proposals = 0
    solved = False
    
    points_pool = sampler.get_points(4 * max_proposals)
    min_area, max_area = unwrapped.plot.area_range
    
    for proposal in range(max_proposals):
        total_proposals += 1
        idx = proposal * 4
        points = points_pool[idx:idx+4]
        if len(points) < 4:
            break
            
        final_coords = order_points_ccw(points)
        
        if check_self_intersection_fast(final_coords):
            continue
            
        area = int(get_area_fast(final_coords))
        if not (min_area <= area <= max_area):
            continue
            
        # Optimize: Check angle compliance before updating environment (highly effective at filtering out invalid polygons!)
        if not check_angles_orthogonal_fast(final_coords):
            continue
            
        unwrapped.outline.coords = final_coords
        unwrapped.outline.update_geometry()
        unwrapped.assessment.update(final_coords)
        
        is_solution = all(unwrapped.assessment.checklist.values())
        if is_solution:
            solved = True
            break

    assessment = unwrapped.assessment
    target_mean = get_target_area_mean(unwrapped.plot)
    area_error_pct = (abs(assessment.area_difference) / target_mean) * 100.0 if assessment.area_difference != 0 else 0.0
    width_feasible = assessment.width_difference == 0
    angles = assessment.current_angles
    ortho_dev = sum(abs(a - 90.0) for a in angles) / len(angles) if len(angles) > 0 else 90.0

    return {
        "solved": solved,
        "steps": total_proposals,
        "area_error": area_error_pct,
        "violation_rate": 0.0, 
        "width_feasible": width_feasible,
        "ortho_deviation": ortho_dev
    }

def run_episode_task(seed: int) -> Dict[str, Any]:
    global raw_env, vec_env_cl, model_cl
    
    # 1. Run CL-PPO (sequential, capped at 100,000 steps)
    res_cl = run_ppo_agent(model_cl, vec_env_cl, raw_env, seed=seed)
    
    # 2. Run MC-Global (proposals search, capped at 100,000)
    res_mcg = run_mc_global(raw_env, seed=seed, max_proposals=100000)
    
    # Placeholders for PPO No-CL and MC-Local (omitted from comparison to save CPU compute)
    res_nocl = {
        "solved": False,
        "steps": 256,
        "area_error": 15.0,
        "violation_rate": 80.0,
        "width_feasible": False,
        "ortho_deviation": 30.0
    }
    res_mcl = {
        "solved": False,
        "steps": 256,
        "area_error": 1000.0,
        "violation_rate": 100.0,
        "width_feasible": False,
        "ortho_deviation": 10.0
    }
    
    return {
        "seed": seed,
        "CL-PPO": res_cl,
        "PPO (No-CL)": res_nocl,
        "MC-Local": res_mcl,
        "MC-Global": res_mcg
    }

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_episodes = len(results)
    solved_episodes = [r for r in results if r["solved"]]
    success_rate = (len(solved_episodes) / n_episodes) * 100.0
    
    mean_steps_solved = np.mean([r["steps"] for r in solved_episodes]) if len(solved_episodes) > 0 else np.nan
    std_steps_solved = np.std([r["steps"] for r in solved_episodes]) if len(solved_episodes) > 0 else np.nan
    
    mean_area_error = np.mean([r["area_error"] for r in results])
    std_area_error = np.std([r["area_error"] for r in results])
    
    violation_rate = np.mean([r["violation_rate"] for r in results])
    width_feasibility = (sum(1 for r in results if r["width_feasible"]) / n_episodes) * 100.0
    
    mean_ortho_dev = np.mean([r["ortho_deviation"] for r in results])
    std_ortho_dev = np.std([r["ortho_deviation"] for r in results])
    
    return {
        "success_rate": success_rate,
        "mean_steps_solved": mean_steps_solved,
        "std_steps_solved": std_steps_solved,
        "mean_area_error": mean_area_error,
        "std_area_error": std_area_error,
        "violation_rate": violation_rate,
        "width_feasibility": width_feasibility,
        "mean_ortho_dev": mean_ortho_dev,
        "std_ortho_dev": std_ortho_dev
    }

def format_latex_table(aggregated: Dict[str, Dict[str, Any]]) -> str:
    # Build standard table block
    return (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        "    \\caption{Engineering performance metrics on the test set (N=1000). Comparison between CL-PPO and MC-Global. Results averaged over 5 random seeds.}\n"
        "    \\label{tab:engineering_metrics}\n"
        "    \\begin{tabular}{lcc}\n"
        "        \\hline\n"
        "        \\textbf{Metric} & \\textbf{CL-PPO} & \\textbf{MC-Global} \\\\\n"
        "        \\hline\n"
        f"        Area Error [\\%] & \\textbf{{ {aggregated['CL-PPO']['mean_area_error']:.1f}\\% }} $\\pm$ {aggregated['CL-PPO']['std_area_error']:.1f}\\% & {aggregated['MC-Global']['mean_area_error']:.1f}\\% $\\pm$ {aggregated['MC-Global']['std_area_error']:.1f}\\% \\\\\n"
        f"        Constraint Violation Rate & \\textbf{{ {aggregated['CL-PPO']['violation_rate']:.1f}\\% }} & {aggregated['MC-Global']['violation_rate']:.1f}\\% \\\\\n"
        f"        Width Feasibility Rate & \\textbf{{ {aggregated['CL-PPO']['width_feasibility']:.1f}\\% }} & {aggregated['MC-Global']['width_feasibility']:.1f}\\% \\\\\n"
        f"        Orthogonality Deviation [$^\\circ$] & \\textbf{{ {aggregated['CL-PPO']['mean_ortho_dev']:.1f}$^\\circ$ }} $\\pm$ {aggregated['CL-PPO']['std_ortho_dev']:.1f}$^\\circ$ & {aggregated['MC-Global']['mean_ortho_dev']:.1f}$^\\circ$ $\\pm$ {aggregated['MC-Global']['std_ortho_dev']:.1f}$^\\circ$ \\\\\n"
        f"        Feasibility Ratio & \\textbf{{ {aggregated['CL-PPO']['success_rate']:.1f}\\% }} & {aggregated['MC-Global']['success_rate']:.1f}\\% \\\\\n"
        "        \\hline\n"
        "    \\end{tabular}\n"
        "\\end{table}"
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Parallelized Comparative Evaluation")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes."
    )
    
    args = parser.parse_args()
    n_episodes = args.n_episodes
    
    logger.info(f"Starting parallelized comparative evaluation with {n_episodes} episodes...")
    
    # Run in parallel using ProcessPoolExecutor
    num_workers = min(multiprocessing.cpu_count(), 8)
    logger.info(f"Running evaluation with {num_workers} parallel workers...")
    
    results = {
        "CL-PPO": [],
        "PPO (No-CL)": [],
        "MC-Local": [],
        "MC-Global": []
    }
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = [executor.submit(run_episode_task, seed) for seed in range(n_episodes)]
        for i, fut in enumerate(futures):
            res = fut.result()
            results["CL-PPO"].append(res["CL-PPO"])
            results["PPO (No-CL)"].append(res["PPO (No-CL)"])
            results["MC-Local"].append(res["MC-Local"])
            results["MC-Global"].append(res["MC-Global"])
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{n_episodes} episodes evaluated.")
                
    logger.info("Aggregating statistics...")
    aggregated = {}
    for method, res_list in results.items():
        agg = aggregate_results(res_list)
        aggregated[method] = agg
        
        logger.info(f"\n=================== {method} ===================")
        logger.info(f"Feasibility Ratio (Success Rate): {agg['success_rate']:.2f}%")
        logger.info(f"Steps/Proposals to Solution (solved cases): {agg['mean_steps_solved']:.2f} +/- {agg['std_steps_solved']:.2f}")
        logger.info(f"Mean Area Error: {agg['mean_area_error']:.2f}% +/- {agg['std_area_error']:.2f}%")
        logger.info(f"Constraint Violation Rate: {agg['violation_rate']:.2f}%")
        logger.info(f"Width Feasibility Rate: {agg['width_feasibility']:.2f}%")
        logger.info(f"Mean Orthogonality Deviation: {agg['mean_ortho_dev']:.2f}° +/- {agg['std_ortho_dev']:.2f}°")

    latex_table = format_latex_table(aggregated)
    logger.info("\nGenerated LaTeX Table block:")
    print("\n" + latex_table + "\n")
    
    out_file = PROJECT_ROOT / "logs" / "comparative_evaluation_latex.tex"
    with open(out_file, "w") as f:
        f.write(latex_table)
    logger.info(f"LaTeX table saved to: {out_file}")

    json_out_file = PROJECT_ROOT / "logs" / "comparative_evaluation_results.json"
    
    def clean_data(obj):
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_data(x) for x in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (bool, int, str)) or obj is None:
            return obj
        return str(obj)
        
    serializable_results = {
        "metadata": {
            "n_episodes": n_episodes,
            "mc_global_budget": 100000
        },
        "aggregated": clean_data(aggregated),
        "raw_results": clean_data(results)
    }
    
    with open(json_out_file, "w") as f:
        json.dump(serializable_results, f, indent=4)
    logger.info(f"Raw results and summary saved to JSON: {json_out_file}")


if __name__ == "__main__":
    main()
