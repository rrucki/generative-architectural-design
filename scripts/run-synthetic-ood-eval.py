"""
run-synthetic-ood-eval.py
-------------------------
Out-of-Distribution (OOD) Synthetic Parcel Generalization Benchmark.

Evaluates the frozen CL-PPO policy and CL-PPO + Local Polish against CMA-ES and SLSQP
under three zero-shot distribution shifts in procedural parcel generation parameters:
1. Regime OOD-1 (Tight Setback Envelopes): Setback offset increased from [40, 80] px
   to [85, 115] px (with covered_area_ratio in [0.10, 0.18]), producing constricted
   interior buildable zones strictly outside the training setback support.
2. Regime OOD-2 (Highly Non-Convex Irregular Boundaries): Increased radial edge
   perturbation (EDGE_MOVE_FACTOR_MIN=3.2, EDGE_MOVE_FACTOR_MAX=5.5,
   offset in [45, 80] px).
3. Regime OOD-3 (High-Density Target Coverage): Increased covered_area_ratio from
   [0.10, 0.20] to [0.18, 0.26] with tighter area tolerance [0.80 * max_area, max_area]
   (offset in [40, 75] px).
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
from urban_planner.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    MIN_TARGET_WIDTH,
    TARGET_WIDTH_MIN_FACTOR,
    TARGET_WIDTH_MAX_FACTOR,
    GENERATION_PARAMS,
)
from urban_planner.domain.geometries import Plot, Outline
from urban_planner.logic.assessment import OutlineAssessment
from urban_planner.types import Stage
from urban_planner.utils import generate_random_outline

from scripts.baselines.run_hybrid_ppo import run_hybrid_ppo
from scripts.baselines.run_cma_es import run_cma_es
from scripts.baselines.run_constrained_opt import run_slsqp
from scripts.baselines.objective import _segments_intersect


def generate_ood_plot(regime: str) -> Plot:
    """Builds an OOD Plot according to the specified synthetic distribution-shift regime."""
    if regime == "ood_tight_setbacks":
        offset = random.randint(85, 115)
        cov_min, cov_max, low_r = 0.10, 0.18, 0.75
        coords, permitted_outline = generate_random_outline(offset)
    elif regime == "ood_irregular_nonconvex":
        orig_p = dict(GENERATION_PARAMS)
        GENERATION_PARAMS["EDGE_MOVE_FACTOR_MIN"] = 3.2
        GENERATION_PARAMS["EDGE_MOVE_FACTOR_MAX"] = 5.5
        try:
            offset = random.randint(45, 80)
            coords, permitted_outline = generate_random_outline(offset)
        finally:
            GENERATION_PARAMS.update(orig_p)
        cov_min, cov_max, low_r = 0.10, 0.20, 0.75
    elif regime == "ood_high_coverage":
        offset = random.randint(40, 75)
        cov_min, cov_max, low_r = 0.18, 0.26, 0.80
        coords, permitted_outline = generate_random_outline(offset)
    else:
        raise ValueError(f"Unknown OOD regime: {regime}")

    polygon = Polygon(coords)
    max_area = polygon.area * random.uniform(cov_min, cov_max)
    area_range = (int(max_area * low_r), int(max_area))
    min_x, min_y, max_x, max_y = permitted_outline.bounds
    max_w = min(max_x - min_x, max_y - min_y)
    lw = max(MIN_TARGET_WIDTH, int(max_w * TARGET_WIDTH_MIN_FACTOR))
    uw = int(max_w * TARGET_WIDTH_MAX_FACTOR)
    if lw >= uw:
        width_range = (lw, lw + 50)
    else:
        v1, v2 = random.randint(lw, uw), random.randint(lw, uw)
        width_range = (min(v1, v2), max(v1, v2))
    return Plot(
        coords=coords,
        permitted_outline=permitted_outline,
        area_range=area_range,
        width_range=width_range,
    )


def apply_ood_plot_override(raw_env: Any, regime: str, seed: int) -> None:
    """Generates an OOD Plot according to the specified distribution shift regime."""
    random.seed(seed)
    np.random.seed(seed)

    orig_params = dict(GENERATION_PARAMS)
    try:
        if regime == "ood_tight_setbacks":
            offset = random.randint(85, 115)
            cov_min, cov_max = 0.10, 0.18
            area_low_ratio = 0.75
        elif regime == "ood_irregular_nonconvex":
            GENERATION_PARAMS["EDGE_MOVE_FACTOR_MIN"] = 3.2
            GENERATION_PARAMS["EDGE_MOVE_FACTOR_MAX"] = 5.5
            offset = random.randint(45, 80)
            cov_min, cov_max = 0.10, 0.20
            area_low_ratio = 0.75
        elif regime == "ood_high_coverage":
            offset = random.randint(40, 75)
            cov_min, cov_max = 0.18, 0.26
            area_low_ratio = 0.80
        else:
            raise ValueError(f"Unknown OOD regime: {regime}")

        coords, permitted_outline = generate_random_outline(offset)
        polygon = Polygon(coords)
        covered_area_ratio = random.uniform(cov_min, cov_max)
        max_area = polygon.area * covered_area_ratio
        area_range = (int(max_area * area_low_ratio), int(max_area))

        min_x, min_y, max_x, max_y = permitted_outline.bounds
        plot_width = max_x - min_x
        plot_height = max_y - min_y
        max_achievable_width = min(plot_width, plot_height)

        lower_w = max(MIN_TARGET_WIDTH, int(max_achievable_width * TARGET_WIDTH_MIN_FACTOR))
        upper_w = int(max_achievable_width * TARGET_WIDTH_MAX_FACTOR)
        if lower_w >= upper_w:
            width_range = (lower_w, lower_w + 50)
        else:
            v1 = random.randint(lower_w, upper_w)
            v2 = random.randint(lower_w, upper_w)
            width_range = (min(v1, v2), max(v1, v2))

        unwrapped = raw_env.unwrapped
        unwrapped.plot = Plot(
            coords=coords,
            permitted_outline=permitted_outline,
            area_range=area_range,
            width_range=width_range,
        )
        unwrapped.outline = Outline(
            coords=[
                (0, 0),
                (WINDOW_WIDTH, 0),
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                (0, WINDOW_HEIGHT),
            ]
        )
        unwrapped.assessment = OutlineAssessment(
            unwrapped.plot.permitted_outline,
            unwrapped.outline.coords,
            unwrapped.plot.area_range,
            unwrapped.plot.width_range,
        )
        unwrapped.assessment.update(unwrapped.outline.coords)
        unwrapped.steps_taken = 0
        unwrapped.placement_solved_in_episode = False
        unwrapped.area_solved_in_episode = False
        unwrapped.width_solved_in_episode = False
        unwrapped.reward_calculator.reset()
        unwrapped.reward_calculator.initial_differences[Stage.PLACEMENT] = sum(
            unwrapped.assessment.outside_shape_distances
        )
        unwrapped.reward_calculator.initial_differences[Stage.AREA] = abs(
            unwrapped.assessment.area_difference
        )
        unwrapped.reward_calculator.initial_differences[Stage.WIDTH] = (
            unwrapped.assessment.width_difference
        )
        unwrapped.reward_calculator.initial_differences[Stage.ANGLE] = sum(
            unwrapped.assessment.angle_differences
        )
    finally:
        GENERATION_PARAMS.update(orig_params)


def main():
    parser = argparse.ArgumentParser(description="Synthetic OOD Parcel Evaluation")
    parser.add_argument("--n-per-regime", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=2000)
    args = parser.parse_args()

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

    regimes = [
        ("ood_tight_setbacks", "OOD-1: Tight Setback Envelopes (85-115 px)"),
        ("ood_irregular_nonconvex", "OOD-2: High Non-Convexity (Edge Factor up to 5.5)"),
        ("ood_high_coverage", "OOD-3: High Coverage Ratio (18-26%)"),
    ]

    summary = {}
    orig_from_random = Plot.from_random_generation

    for reg_key, reg_label in regimes:
        print(f"\n=== Evaluating {reg_label} ({args.n_per_regime} parcels) ===")
        # Monkeypatch Plot.from_random_generation so all baseline calls inside reset(seed=s) use the OOD generator
        Plot.from_random_generation = classmethod(lambda cls, rk=reg_key: generate_ood_plot(rk))

        reg_res = {"CL-PPO": [], "CL-PPO + Polish": [], "SLSQP": [], "CMA-ES": []}
        solvable_count = 0

        for i in range(args.n_per_regime):
            s = args.seed_offset + i
            # 1. CL-PPO + Polish (also records whether CL-PPO alone solved it via polish_evals == 0)
            r_hyb = run_hybrid_ppo(model, vec_env, raw_env, seed=s)
            r_ppo_solved = r_hyb["solved"] and r_hyb["polish_evals"] == 0
            reg_res["CL-PPO"].append({"solved": r_ppo_solved, "steps": r_hyb["ppo_steps"]})
            reg_res["CL-PPO + Polish"].append(r_hyb)

            # 2. SLSQP
            r_slsqp = run_slsqp(raw_env, seed=s, num_starts=5)
            reg_res["SLSQP"].append(r_slsqp)

            # 3. CMA-ES (fast budget 15,000 evals)
            r_cma = run_cma_es(raw_env, seed=s, max_evals=15000)
            reg_res["CMA-ES"].append(r_cma)

            if r_hyb["solved"] or r_slsqp["solved"] or r_cma["solved"]:
                solvable_count += 1

        Plot.from_random_generation = orig_from_random

        reg_agg = {"n_parcels": args.n_per_regime, "n_solvable": solvable_count, "methods": {}}
        for m_name, m_list in reg_res.items():
            solved_n = sum(1 for r in m_list if r["solved"])
            raw_pct = (solved_n / args.n_per_regime) * 100.0
            solv_pct = (solved_n / max(1, solvable_count)) * 100.0
            reg_agg["methods"][m_name] = {
                "raw_feasibility_pct": raw_pct,
                "solvable_feasibility_pct": min(100.0, solv_pct),
            }
            print(f"  {m_name:16s}: Raw = {raw_pct:5.1f}% | Solvable-Normalized = {min(100.0, solv_pct):5.1f}%")

        summary[reg_key] = reg_agg

    out_path = PROJECT_ROOT / "logs" / "synthetic_ood_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved OOD evaluation results to {out_path}")


if __name__ == "__main__":
    main()

