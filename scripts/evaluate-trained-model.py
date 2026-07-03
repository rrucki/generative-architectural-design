"""
Script to evaluate a pre-trained PPO agent on the UrbanPlanner-v0 environment.

This script loads a specified model, its corresponding VecNormalize statistics,
and runs a series of evaluation episodes, printing the mean reward.

It is designed to be robust, automatically finding necessary files and
reading run configurations to set up the environment correctly.

Example:
    * Evaluate the 'best_model.zip' from a specific run and render the process
    python scripts/evaluate-trained-model.py \
        --model-path "stage_1_1754723612/best_model.zip" \
        --render

    * Evaluate a final model for 100 episodes without rendering
    python scripts/evaluate-trained-model.py \
        --model-path "my_ppo_run_12345/final_model.zip" \
        --n-eval-episodes 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# This import is crucial for registering the custom environment with Gymnasium.
# noinspection PyUnresolvedReferences
import urban_planner
from urban_planner.config import INTERNAL_MAX_STEPS
from urban_planner.logger_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


class SuccessCounterCallback(BaseCallback):
    """
    A callback to count successful episodes during evaluation.
    An episode is considered a success if it terminates before reaching
    the maximum number of steps, which implies a solution was found.
    This callback is compatible with both modern `model.learn()` and older
    `evaluate_policy` callback signatures by being a callable object.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_successes = 0
        self.n_episodes = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.n_episodes += 1
                    if 'steps' in info and info['steps'] < INTERNAL_MAX_STEPS:
                        self.n_successes += 1
        return True

    def __call__(self, locals_dict, _globals_dict) -> None:
        """
        This method is called by `evaluate_policy` in some versions of SB3.
        We update self.locals and call `_on_step` to stay compatible.
        The `on_step` method of the callback returns a boolean
        (True to continue, False to stop), but the old-style
        callback does not return anything, so we ignore the return value.
        """
        self.locals = locals_dict
        self._on_step()


def find_vec_normalize_stats(model_path: Path) -> Path:
    """
    Finds the VecNormalize statistics file associated with a model.
    It searches for files like 'best_model_vec_normalize.pkl' or a file
    matching the model's stem.
    """
    model_dir = model_path.parent
    expected_stats_name = model_path.stem + "_vec_normalize.pkl"

    possible_paths = [
        model_dir / expected_stats_name,
        model_dir / "best_model_vec_normalize.pkl",
        model_dir / "final_model_vec_normalize.pkl"
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found VecNormalize stats at: {path}")
            return path

    raise FileNotFoundError(
        f"Could not find a VecNormalize stats file for model {model_path}. "
        f"Searched in {model_dir} for names like '{expected_stats_name}', "
        "'best_model_vec_normalize.pkl', or 'final_model_vec_normalize.pkl'."
    )


def evaluate_agent(args: argparse.Namespace):
    """Loads a trained agent and evaluates its performance."""
    model_path = PROJECT_ROOT / "models" / args.model_path
    if not model_path.exists():
        logger.critical(f"Model file not found at: {model_path}")
        sys.exit(1)

    model_dir = model_path.parent
    config_path = model_dir / "run_config.json"

    stage = args.stage
    if config_path.exists():
        with open(config_path, 'r') as f:
            run_config = json.load(f)
        if stage is None:
            stage = run_config.get('stage', 1)
            logger.info(f"Using stage {stage} from run configuration file.")
        else:
            logger.info(
                f"Using stage {stage} from command-line argument (overrides config).")
    elif stage is None:
        logger.critical(
            "Stage not specified and run_config.json not found. Please provide --stage.")
        sys.exit(1)

    try:
        stats_path = find_vec_normalize_stats(model_path)
    except FileNotFoundError as e:
        logger.critical(e)
        sys.exit(1)

    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Using VecNormalize stats: {stats_path}")
    logger.info(f"Environment Stage: {stage}")
    logger.info(f"Number of evaluation episodes: {args.n_eval_episodes}")
    if args.render:
        logger.info("Rendering is ENABLED.")

    render_mode = "human" if args.render else None
    eval_env_kwargs = {'stage': stage, 'render_mode': render_mode}

    base_eval_env = make_vec_env(args.env_id, n_envs=1,
                                 vec_env_cls=DummyVecEnv,
                                 env_kwargs=eval_env_kwargs)

    eval_env = VecNormalize.load(str(stats_path), base_eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    success_callback = SuccessCounterCallback()

    mean_reward, std_reward = evaluate_policy(model, eval_env,
                                              n_eval_episodes=args.n_eval_episodes,
                                              render=args.render,
                                              deterministic=True,
                                              callback=success_callback)

    logger.info("--- Evaluation Finished ---")
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    success_rate = (success_callback.n_successes / success_callback.n_episodes) * 100 if success_callback.n_episodes > 0 else 0
    logger.info(
        f"Solutions found: {success_callback.n_successes}/{success_callback.n_episodes} ({success_rate:.1f}%)"
    )
    logger.info("---------------------------")
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent for the Urban Planner environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model .zip file to evaluate, relative to the 'models/' directory (e.g., 'my_run_123/best_model.zip').")
    parser.add_argument("--n-eval-episodes", type=int, default=50,
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--render", action="store_true",
                        help="If set, render the environment during evaluation.")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Set the training stage for the environment. If not set, it will be read from the run's config file.")
    parser.add_argument("--env-id", type=str, default="UrbanPlanner-v0",
                        help="The Gymnasium environment ID.")

    args = parser.parse_args()
    setup_logging()

    try:
        evaluate_agent(args)
        logger.info("Evaluation script finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during evaluation: {e}",
                        exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
