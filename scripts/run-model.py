"""
Script to run a pre-trained PPO agent in the UrbanPlanner-v0 environment.

This script loads a saved Stable-Baselines3 model and runs it for a specified
number of episodes, rendering the environment's state in real-time.

It provides command-line arguments to customize the model path, environment ID,
and other execution parameters.

Example:
    python scripts/run-model.py --model-path models/ppo_final.zip --episodes 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# This import is crucial for registering the custom environment with Gymnasium.
# noinspection PyUnresolvedReferences
import urban_planner
from urban_planner.logger_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def run_agent(
    model_path: Path,
    env_id: str,
    episodes: int,
    render_mode: str,
    max_episode_steps: Optional[int],
    stage: int,
) -> None:
    """
    Loads a PPO model and runs it in the specified Gymnasium environment.

    Args:
        model_path: Path to the saved model .zip file.
        env_id: The ID of the Gymnasium environment to create.
        episodes: The number of episodes to run.
        render_mode: The rendering mode for the environment (e.g., 'human').
        max_episode_steps: The maximum number of steps per episode. If None, no limit.
        stage: The environment stage (1 or 2) to run the model in.
    """
    if not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    expected_stats_filename = model_path.stem + "_vec_normalize.pkl"
    possible_stats_paths = [
        model_path.parent / expected_stats_filename,
        model_path.parent / "best_model_vec_normalize.pkl",
        model_path.parent / "final_model_vec_normalize.pkl",
        model_path.parent / "vec_normalize.pkl"
    ]

    stats_path = None
    for path in possible_stats_paths:
        if path.exists():
            stats_path = path
            logger.info(f"Używam pliku statystyk: {stats_path}")
            break

    if not stats_path:
        logger.error(f"Nie znaleziono pliku statystyk VecNormalize dla modelu: {model_path}")
        logger.error(f"Sprawdzono następujące nazwy plików w katalogu modelu: {', '.join([p.name for p in possible_stats_paths])}")
        logger.error("Ten model prawdopodobnie był trenowany z VecNormalize i nie może być uruchomiony bez tego pliku.")
        raise FileNotFoundError(f"Nie znaleziono pliku statystyk dla {model_path}")
    env = None
    try:
        def make_env():
            local_env = gym.make(env_id, render_mode=render_mode, stage=stage)
            if max_episode_steps:
                logger.info(f"Nakładanie wrappera TimeLimit z {max_episode_steps} krokami.")
                local_env = gym.wrappers.TimeLimit(local_env, max_episode_steps=max_episode_steps)
            return local_env

        logger.info("Tworzenie wektoryzowanego środowiska i wczytywanie statystyk normalizacji...")
        env = DummyVecEnv([make_env])
        env = VecNormalize.load(str(stats_path), env)

        env.training = False
        env.norm_reward = False

        logger.info(f"Wczytywanie modelu z: {model_path}")
        model = PPO.load(model_path, env=env)
        logger.info("Model loaded successfully.")

        logger.info(f"Starting evaluation for {episodes} episodes...")
        for episode in range(episodes):
            obs = env.reset()
            done = False
            num_steps = 0
            total_reward = 0.0

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                print(infos)
                reward = rewards[0]
                done = dones[0]
                info = infos[0]

                env.render()
                total_reward += reward
                num_steps += 1

            logger.info(
                f"Episode {episode + 1} finished. "
                f"Total Reward: {total_reward:.2f}, Steps: {num_steps}"
            )
            if info.get("is_solution", False):
                logger.info("Solution found! Pausing for 2 seconds...")
                time.sleep(2)

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        raise
    finally:
        if env:
            logger.info("Closing environment.")
            env.close()


def main():
    """
    Parses command-line arguments and runs the agent simulation.
    Handles application-level setup and teardown, including exit codes.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run a pre-trained PPO agent for the Urban Planner environment."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "ppo.zip",
        help="Path to the trained model .zip file (relative to project root). "
             "Default: models/ppo.zip",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="UrbanPlanner-v0",
        help="Gymnasium environment ID to run.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Set the environment stage to run the model in. Defaults to 2 for evaluating advanced models.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        help="Environment render mode (e.g., 'human', 'rgb_array').",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode. Set to 0 for no limit.",
    )

    args = parser.parse_args()
    max_steps = args.max_steps if args.max_steps > 0 else None

    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    try:
        run_agent(
            model_path=model_path,
            env_id=args.env_id,
            episodes=args.episodes,
            render_mode=args.render_mode,
            max_episode_steps=max_steps,
            stage=args.stage
        )
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()