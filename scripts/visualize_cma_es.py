"""
visualize_cma_es.py
-------------------
Real-time Pygame visualization of CMA-ES optimizing building footprints
on irregular plots.

Usage:
    ./.venv/bin/python scripts/visualize_cma_es.py --episodes 5 --delay 0.03
"""

import argparse
import random
import sys
import time
from pathlib import Path
import cma
import gymnasium as gym
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner
from urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT
from urban_planner.types import Stage
from scripts.baselines.objective import (
    calculate_footprint_cost,
    evaluate_final_solution,
    order_points_ccw,
)


def visualize_cma_es(episodes: int = 5, delay: float = 0.02, start_seed: int = 42):
    env = gym.make("UrbanPlanner-v0", render_mode="human", stage=4)
    unwrapped = env.unwrapped

    print(f"\n=======================================================")
    print(f" Starting Real-time CMA-ES Visualizer ({episodes} episodes)")
    print(f" Press 'Q' or close Pygame window to exit early.")
    print(f"=======================================================\n")

    for ep in range(episodes):
        seed = start_seed + ep
        random.seed(seed)
        np.random.seed(seed)

        obs, info = env.reset(seed=seed)
        env.render()
        permitted_poly = unwrapped.plot.permitted_outline
        minx, miny, maxx, maxy = permitted_poly.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0

        w = max(50.0, min(150.0, (maxx - minx) * 0.35))
        h = max(50.0, min(150.0, (maxy - miny) * 0.35))
        x0 = np.array(
            [
                cx - w / 2.0, cy - h / 2.0,
                cx + w / 2.0, cy - h / 2.0,
                cx + w / 2.0, cy + h / 2.0,
                cx - w / 2.0, cy + h / 2.0,
            ],
            dtype=float,
        )

        sigma0 = max(15.0, 0.25 * min(maxx - minx, maxy - miny))
        lower_bounds = [0.0] * 8
        upper_bounds = [float(WINDOW_WIDTH), float(WINDOW_HEIGHT)] * 4

        opts = {
            "bounds": [lower_bounds, upper_bounds],
            "seed": seed,
            "maxfevals": 10000,
            "tolfun": 1e-4,
            "verb_disp": 0,
            "verbose": -9,
            "popsize": 12,
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        start_time = time.perf_counter()
        generation = 0
        fevals = 0
        best_x = x0
        best_cost = float("inf")
        solved = False

        while not es.stop() and not solved:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    print("Exiting visualizer.")
                    env.close()
                    return

            solutions = es.ask()
            fitnesses = []
            generation += 1

            for sol in solutions:
                cost = calculate_footprint_cost(sol, unwrapped)
                fitnesses.append(cost)
                fevals += 1

                if cost < best_cost:
                    best_cost = cost
                    best_x = np.copy(sol)

                    # Update environment coordinates to display current best candidate
                    x_clipped = np.clip(sol[0::2], 0.0, float(WINDOW_WIDTH))
                    y_clipped = np.clip(sol[1::2], 0.0, float(WINDOW_HEIGHT))
                    ordered_coords = order_points_ccw(list(zip(x_clipped, y_clipped)))
                    int_coords = [(int(round(c[0])), int(round(c[1]))) for c in ordered_coords]

                    unwrapped.outline.coords = int_coords
                    unwrapped.outline.update_geometry()
                    unwrapped.assessment.update(int_coords)

                    if all(unwrapped.assessment.checklist.values()):
                        solved = True

            es.tell(solutions, fitnesses)

            # Render current generation to screen
            env.render()

            if delay > 0:
                time.sleep(delay)

        elapsed = time.perf_counter() - start_time
        res_str = "SUCCESS (Solved)" if solved else "FINISHED (Budget Exceeded)"
        print(f"Episode {ep+1} [{res_str}]: Evals={fevals}, Time={elapsed:.2f}s, Best Cost={best_cost:.2f}")

        # Pause briefly on final layout so the user can inspect it
        pause_start = time.time()
        while time.time() - pause_start < 1.5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    env.close()
                    return
            env.render()
            time.sleep(0.05)

    print("\nAll visualization episodes finished.")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time CMA-ES visualizer")
    parser.add_argument("--episodes", type=int, default=5, help="Number of parcel episodes")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay per generation in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    args = parser.parse_args()

    visualize_cma_es(episodes=args.episodes, delay=args.delay, start_seed=args.seed)
