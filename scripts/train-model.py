"""
Script to train a PPO agent on the UrbanPlanner-v0 environment.

This script handles the entire training pipeline:
1.  Sets up a dedicated log directory for each run.
2.  Configures the PPO model with specified hyperparameters.
3.  Sets up callbacks for periodic evaluation and saving the best model.
4.  Launches the training process with TensorBoard logging.
5.  Saves final and best models with unique names directly in the 'models/' directory.
6.  Performs a final evaluation on the best saved model.

All critical parameters can be configured via command-line arguments.

Example:
    # Start a training run with a specific name
    python scripts/train-model.py --run-name "PPO_run_1"

    # Start a run with an auto-generated timestamp name
    python scripts/train-model.py
"""

import argparse
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# This import is crucial for registering the custom environment with Gymnasium.
# noinspection PyUnresolvedReferences
import urban_planner
from urban_planner.envs.generation_manager import EnvironmentGenerator
from urban_planner.logger_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def train_agent(args: argparse.Namespace) -> None:
    """
    Sets up and runs the entire training and evaluation pipeline.
    """
    run_timestamp = int(time.time())
    base_name = f"{args.run_name}_{run_timestamp}" if args.run_name else str(run_timestamp)

    log_dir = PROJECT_ROOT / "logs" / base_name
    model_dir = PROJECT_ROOT / "models" / base_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    logger.info(f"Starting new run: {base_name}")
    logger.info(f"Logs will be saved to: {log_dir}")
    logger.info(f"Models will be saved to: {model_dir}")

    with multiprocessing.Manager() as manager:
        logger.info("Starting environment generation manager...")
        generation_queue = manager.Queue(maxsize=args.n_envs * 2)
        generator = EnvironmentGenerator(queue=generation_queue)
        generator.start()

        try:
            n_envs = args.n_envs
            logger.info(f"Creating {n_envs} parallel environments...")
            train_env = make_vec_env(
                args.env_id,
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=str(log_dir),
                env_kwargs={'generation_queue': generation_queue}
            )

            eval_render_mode = "human" if args.render_eval else None
            eval_env = make_vec_env(
                args.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                env_kwargs={'render_mode': eval_render_mode}
            )

            eval_callback = EvalCallback(
                eval_env,
                eval_freq=max(args.eval_freq // n_envs, 1),
                best_model_save_path=str(model_dir),
                log_path=str(log_dir),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=args.render_eval,
                verbose=1,
            )

            def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
                def func(progress_remaining: float) -> float:
                    return end_value + (initial_value - end_value) * progress_remaining
                return func

            lr_schedule = linear_schedule(1e-5, 5e-7)
            policy_kwargs = dict(net_arch=args.net_arch)
            clip_schedule = linear_schedule(initial_value=0.1, end_value=0.02)
            model = PPO(
                "MultiInputPolicy",
                train_env,
                learning_rate=lr_schedule,
                n_steps=args.n_steps // n_envs,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                clip_range=clip_schedule,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(PROJECT_ROOT / "logs"),
            )

            logger.info(f"Starting training for {args.total_timesteps} timesteps...")
            logger.info("Launch TensorBoard to monitor progress: python scripts/view_logs.py")

            model.learn(
                total_timesteps=args.total_timesteps,
                callback=CallbackList([eval_callback]),
                tb_log_name=base_name,
            )

            final_model_path = model_dir / f"{base_name}_final.zip"
            model.save(final_model_path)
            logger.info(f"Training complete. Final model saved to: {final_model_path}")

            best_model_path = model_dir / f"{base_name}_best.zip"

            if best_model_path.exists():
                logger.info(f"Loading best model from {best_model_path} for final evaluation...")
                best_model = PPO.load(best_model_path, env=eval_env)
                mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=50, deterministic=True)
                logger.info(f"Final evaluation of BEST model -> Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            else:
                logger.warning("No best model was saved during training. Skipping final evaluation.")

        finally:
            logger.info("Stopping environment generator...")
            generator.stop()
            generator.join(timeout=5)
            if generator.is_alive():
                logger.warning("Generator process did not terminate gracefully, forcing termination.")
                generator.terminate()
            logger.info("Generator stopped.")


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for the Urban Planner environment.")
    parser.add_argument("--run-name", type=str, default=None, help="An optional name for this training run. If not provided, a timestamp will be used.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments to use.")
    parser.add_argument("--env-id", type=str, default="UrbanPlanner-v0", help="Gymnasium environment ID.")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000, help="Total number of training steps.")
    parser.add_argument("--n-steps", type=int, default=8192, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch-size", type=int, default=512, help="The minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=4, help="Number of epochs to run per PPO update (default: 4).")
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor.")
    parser.add_argument("--ent-coef", type=float, default=0.002, help="Entropy coefficient.")
    parser.add_argument("--clip-range", type=float, default=0.1, help="Clipping parameter for PPO.")
    parser.add_argument("--net-arch", type=int, nargs='+', default=[256, 256, 128, 64], help="Network architecture.")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluate the agent every n steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes to use for evaluation.")
    parser.add_argument("--render-eval", action="store_true", help="If set, render the environment during evaluation phases.")

    args = parser.parse_args()
    setup_logging()

    try:
        train_agent(args)
        logger.info("Script finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()