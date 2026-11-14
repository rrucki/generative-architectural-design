"""
evaluate_diversity.py
---------------------
Stochastic Generative Diversity Evaluation.

Demonstrates that unlike deterministic single-point numerical optimizers (e.g., SLSQP),
the learned CL-PPO policy defines a stochastic generative distribution pi_theta(a|s)
capable of sampling diverse, geometrically distinct, regulation-compliant building
footprints for the exact same parcel instance.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from shapely.geometry import Polygon
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner
from urban_planner.config import INTERNAL_MAX_STEPS
from scripts.baselines.objective import _segments_intersect, _fast_rectangular_width


def principal_orientation_deg(coords: List[Tuple[int, int]]) -> float:
    """Computes the orientation angle in [0, 90) degrees of the longest polygon edge."""
    best_len = -1.0
    best_ang = 0.0
    for i in range(4):
        p1 = coords[i]
        p2 = coords[(i + 1) % 4]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            ang = math.degrees(math.atan2(dy, dx)) % 90.0
            best_ang = ang
    return best_ang


def evaluate_stochastic_diversity(
    n_parcels: int = 50,
    k_rollouts: int = 20,
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    raw_env = gym.make("UrbanPlanner-v0", stage=4)
    dummy = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_env = VecNormalize.load(
        str(PROJECT_ROOT / "models/stage_4_tunning_1757784606/best_model_vec_normalize.pkl"),
        dummy,
    )
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(
        PROJECT_ROOT / "models/stage_4_tunning_1757784606/best_model.zip",
        env=vec_env,
        device="cpu",
    )

    # Select first n_parcels feasible seeds where deterministic CL-PPO solves the parcel
    with open(PROJECT_ROOT / "logs/comparative_evaluation_v2_results.json") as f:
        v2 = json.load(f)["raw_results"]["CL-PPO"]
    feasible_seeds = [idx for idx, r in enumerate(v2) if r["solved"]][:n_parcels]

    unique_counts = []
    solved_rates_per_parcel = []
    pairwise_centroid_shifts = []
    pairwise_iou_dissimilarity = []
    aspect_ratio_ranges = []
    area_ranges_pct = []

    for p_seed in feasible_seeds:
        valid_coords_list = []
        valid_polys = []

        for k in range(k_rollouts):
            # Reset parcel with fixed p_seed so the parcel geometry is identical across all K rollouts
            obs, _ = raw_env.reset(seed=p_seed)
            unwrapped = raw_env.unwrapped

            # Seed numpy/torch differently per rollout k so stochastic policy samples diverse trajectories
            rollout_rng_seed = p_seed * 1000 + k + 1
            np.random.seed(rollout_rng_seed)
            torch.manual_seed(rollout_rng_seed)

            for step in range(INTERNAL_MAX_STEPS):
                batched = {key: np.expand_dims(val, axis=0) for key, val in obs.items()}
                norm_obs = vec_env.normalize_obs(batched)
                action, _ = model.predict(norm_obs, deterministic=False)
                obs, reward, terminated, truncated, info = raw_env.step(action[0])

                if terminated or truncated:
                    coords = list(unwrapped.assessment.current_outline_coords)
                    p0, p1, p2, p3 = coords
                    invalid = _segments_intersect(p0, p1, p2, p3) or _segments_intersect(p1, p2, p3, p0)
                    if info.get("is_solution", False) and not invalid:
                        valid_coords_list.append(tuple(coords))
                        valid_polys.append(Polygon(coords))
                    break

        solved_rates_per_parcel.append((len(valid_coords_list) / k_rollouts) * 100.0)
        unique_set = list(set(valid_coords_list))
        unique_counts.append(len(unique_set))

        if len(unique_set) >= 2:
            polys = [Polygon(c) for c in unique_set]
            centroids = [(np.mean([pt[0] for pt in c]), np.mean([pt[1] for pt in c])) for c in unique_set]
            areas = [p.area for p in polys]
            aspects = []
            for c, p in zip(unique_set, polys):
                w = max(1.0, _fast_rectangular_width(list(c)))
                l = max(w, p.area / w)
                aspects.append(l / w)

            c_dists = []
            iou_diffs = []
            for i in range(len(unique_set)):
                for j in range(i + 1, len(unique_set)):
                    c_dists.append(
                        math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                    )
                    inter = polys[i].intersection(polys[j]).area
                    union = polys[i].union(polys[j]).area
                    iou = inter / union if union > 0 else 1.0
                    iou_diffs.append(1.0 - iou)

            pairwise_centroid_shifts.append(float(np.mean(c_dists)))
            pairwise_iou_dissimilarity.append(float(np.mean(iou_diffs)))
            aspect_ratio_ranges.append(float(max(aspects) - min(aspects)))
            area_ranges_pct.append(float((max(areas) - min(areas)) / np.mean(areas) * 100.0))

    summary = {
        "n_parcels": n_parcels,
        "k_rollouts_per_parcel": k_rollouts,
        "mean_stochastic_feasibility_pct": float(np.mean(solved_rates_per_parcel)),
        "mean_unique_valid_layouts_per_parcel": float(np.mean(unique_counts)),
        "std_unique_valid_layouts_per_parcel": float(np.std(unique_counts)),
        "multi_solution_parcel_rate_pct": float(sum(1 for u in unique_counts if u >= 2) / n_parcels * 100.0),
        "mean_pairwise_centroid_shift_px": float(np.mean(pairwise_centroid_shifts)),
        "mean_pairwise_spatial_dissimilarity_1_minus_iou": float(np.mean(pairwise_iou_dissimilarity)),
        "mean_aspect_ratio_spread": float(np.mean(aspect_ratio_ranges)),
        "mean_area_variation_pct": float(np.mean(area_ranges_pct)),
    }

    out_path = PROJECT_ROOT / "logs" / "diversity_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== Stochastic Generative Diversity Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Saved diversity evaluation results to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-parcels", type=int, default=50)
    parser.add_argument("--k-rollouts", type=int, default=20)
    args = parser.parse_args()
    evaluate_stochastic_diversity(n_parcels=args.n_parcels, k_rollouts=args.k_rollouts)

