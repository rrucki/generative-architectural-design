"""
Script for testing and visualizing the UrbanPlanner-v0 environment.

This script can run the environment in two modes:
1.  'random': Takes random actions, useful for observing general dynamics.
2.  'manual': Prompts the user to enter a relative action (vertex, dx, dy) at each step.

It follows the standard script structure with command-line arguments for configuration.

Examples:
    * Run in random mode for 3 episodes with a 50ms delay
    python scripts/run-environment.py --episodes 3 --delay 0.05

    * Run in manual mode for 1 episode
    python scripts/run-environment.py --mode manual --episodes 1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# This import is crucial for registering the custom environment with Gymnasium.
# noinspection PyUnresolvedReferences
import urban_planner
from urban_planner.config import MAX_MOVE_DISTANCE
from urban_planner.envs.urban_planner import UrbanPlannerEnv
from urban_planner.logger_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def get_manual_action() -> np.ndarray | None:
    """
    Prompts the user for a relative action (vertex index, dx, dy) and validates the input.

    The user-provided dx, dy values are mapped to the environment's discrete action space.

    Returns:
        A numpy array representing the action if input is valid, otherwise None.
    """
    while True:
        try:
            prompt = (f"Enter action: vertex_idx(0-3) dx(-{MAX_MOVE_DISTANCE}..{MAX_MOVE_DISTANCE}) "
                      f"dy(-{MAX_MOVE_DISTANCE}..{MAX_MOVE_DISTANCE}) or 'q' to quit: ")
            user_input = input(prompt)

            if user_input.lower() in ['q', 'quit', 'exit']:
                logger.info("Manual exit requested by user.")
                return None

            parts = [int(p) for p in user_input.split()]
            if len(parts) != 3:
                logger.error(f"Invalid input: Expected 3 numbers (vertex_idx dx dy), but got {len(parts)}. Please try again.")
                continue

            vertex_index, dx, dy = parts

            if not (0 <= vertex_index <= 3):
                logger.error(f"Invalid vertex index: {vertex_index}. Must be between 0 and 3.")
                continue

            if not (-MAX_MOVE_DISTANCE <= dx <= MAX_MOVE_DISTANCE):
                logger.error(f"Invalid dx: {dx}. Must be between -{MAX_MOVE_DISTANCE} and {MAX_MOVE_DISTANCE}.")
                continue

            if not (-MAX_MOVE_DISTANCE <= dy <= MAX_MOVE_DISTANCE):
                logger.error(f"Invalid dy: {dy}. Must be between -{MAX_MOVE_DISTANCE} and {MAX_MOVE_DISTANCE}.")
                continue

            dx_action = dx + MAX_MOVE_DISTANCE
            dy_action = dy + MAX_MOVE_DISTANCE

            action = np.array([vertex_index, dx_action, dy_action], dtype=np.int64)
            return action

        except ValueError:
            logger.error("Invalid input: Please enter only integer numbers separated by spaces.")


def run_test(episodes: int, delay: float, mode: str, stage: int) -> None:
    """
    Runs the UrbanPlanner-v0 environment for visualization.

    Args:
        episodes: The number of episodes to run.
        delay: The time in seconds to wait between steps.
        mode: The execution mode ('random' or 'manual').
        stage: The training stage (1, 2, 3 or 4).
    """
    env = None
    try:
        logger.info(
            f"Creating environment: UrbanPlanner-v0 (mode: {mode}, stage: {stage})")
        env = gym.make("UrbanPlanner-v0", render_mode="human", stage=stage)

        for episode in range(episodes):
            logger.info(f"--- Starting Episode {episode + 1}/{episodes} ---")
            obs, info = env.reset()

            if mode == 'manual':
                unwrapped_env = env.unwrapped
                if isinstance(unwrapped_env, UrbanPlannerEnv) and unwrapped_env.plot:
                    permitted_coords = unwrapped_env.plot.permitted_outline.exterior.coords
                    coords_str = " ".join([f"{int(c[0])} {int(c[1])}" for c in permitted_coords[:-1]])
                    logger.info(f"Permitted Outline Coords (for reference): {coords_str}")
                else:
                    logger.warning("Could not access 'plot' attribute on the unwrapped environment.")

            terminated = False
            truncated = False
            total_reward = 0.0
            step_count = 0

            while not (terminated or truncated):
                env.render()

                if mode == 'manual':
                    logger.info("--- Agent Observation ---")
                    for key, value in obs.items():
                        logger.info(f"  {key}: {np.round(value, 2).tolist()}")
                    logger.info("-------------------------")
                    action = get_manual_action()
                    if action is None:
                        return
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                if delay > 0:
                    time.sleep(delay)

            env.render()
            logger.info(f"Episode finished after {step_count} steps.")
            logger.info(f"Total reward: {total_reward:.2f}")
            logger.info(f"Final info: {info}")

            if terminated:
                logger.info("Termination condition met (solution likely found). Pausing...")
                time.sleep(3)
            else:
                logger.info("Truncation condition met (max steps reached).")
                time.sleep(1)

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        raise
    finally:
        if env:
            logger.info("Closing environment.")
            env.close()


def main():
    """
    Parses command-line arguments and runs the environment test.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run and visualize the UrbanPlanner-v0 environment."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['random', 'manual'],
        default='random',
        help="Execution mode: 'random' for random actions, 'manual' for user input."
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Set the environment stage (1: placement, 2: placement + area)."
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.01,
        help="Delay in seconds between each step to make rendering watchable. Default: 0.01s.",
    )

    args = parser.parse_args()

    try:
        run_test(episodes=args.episodes, delay=args.delay, mode=args.mode,
                 stage=args.stage)
        logger.info("Script finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()