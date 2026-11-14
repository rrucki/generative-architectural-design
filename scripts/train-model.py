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
    * Start a training run with a specific name
    python scripts/train-model.py --run-name "PPO_run_1"

    * Start a run with an auto-generated timestamp name
    python scripts/train-model.py
    python scripts/train-model.py --run-name "stage1_linear_clip" --stage 1 \
        --lr-schedule-type linear \
        --lr-start 0.001 \
        --lr-end 0.0001 \
        --clip-schedule-type linear \
        --clip-start 0.2 \
        --clip-end 0.05 \
        --ent-coef 0.025

    python scripts/train-model.py \
    --run-name "stage_2_warmup" \
    --stage 1 \
    --resume-from-model "stage_1_1754140487/best_model.zip" \
    --total-timesteps 8000000 \
    --lr-schedule-type constant \
    --lr-start 0.0001 \
    --clip-schedule-type constant \
    --clip-start 0.1
"""

import argparse
import logging
import multiprocessing as mp
import json
import sys
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# This import is crucial for registering the custom environment with Gymnasium.
# noinspection PyUnresolvedReferences
import urban_planner
from urban_planner.envs.generation_manager import EnvironmentGenerator
from urban_planner.logger_config import setup_logging
from urban_planner.utils.callbacks import (
    MultiObjectiveEvalCallback,
    SyncVecNormalizeCallback,
    GradientNormCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def stop_generator(generator: EnvironmentGenerator):
    """Safely stops the environment generator process."""
    if generator and generator.is_alive():
        logger.info("Stopping environment generator...")
        generator.stop()
        generator.join(timeout=5)
        if generator.is_alive():
            logger.warning(
                "Generator process did not terminate gracefully, forcing termination.")
            generator.terminate()
        logger.info("Generator stopped.")


def train_agent(args: argparse.Namespace) -> None:
    """
    Sets up and runs the entire training and evaluation pipeline.
    """
    # --- 1. Determine run name, directories, and files to load ---
    model_path_to_resume = None
    stats_path_to_load = None

    if args.resume_run_name:
        base_name = args.resume_run_name
        log_dir = PROJECT_ROOT / "logs" / base_name
        model_dir = PROJECT_ROOT / "models" / base_name

        if not log_dir.exists() or not model_dir.exists():
            raise FileNotFoundError(f"Could not find existing log/model directories for run: {base_name}")

        model_path_to_resume = model_dir / "final_model.zip"
        stats_path_to_load = model_dir / "final_model_vec_normalize.pkl"

        if not model_path_to_resume.exists() or not stats_path_to_load.exists():
            raise FileNotFoundError(f"Could not find final_model.zip and/or final_model_vec_normalize.pkl in {model_dir}")

        logger.info(f"Resuming run '{base_name}' from interrupted state.")
        logger.info(f"Logs will continue to be saved to: {log_dir}")

    else:  # New run or resume-from-model into a new run
        run_timestamp = int(time.time())
        base_name = f"{args.run_name}_{run_timestamp}" if args.run_name else str(run_timestamp)
        log_dir = PROJECT_ROOT / "logs" / base_name
        model_dir = PROJECT_ROOT / "models" / base_name
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        config_path = log_dir / "run_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Run configuration saved to: {config_path}")
        logger.info(f"Starting new run: {base_name}")

        if args.resume_from_model:
            model_path_to_resume = PROJECT_ROOT / "models" / args.resume_from_model
            if not model_path_to_resume.exists():
                raise FileNotFoundError(f"Model to resume not found at: {model_path_to_resume}")

            model_dir_to_resume = model_path_to_resume.parent
            expected_stats_name = model_path_to_resume.stem + "_vec_normalize.pkl"
            possible_paths = [
                model_dir_to_resume / expected_stats_name,
                model_dir_to_resume / "best_model_vec_normalize.pkl",
                model_dir_to_resume / "final_model_vec_normalize.pkl"
            ]
            for path in possible_paths:
                if path.exists():
                    stats_path_to_load = path
                    break
            if not stats_path_to_load:
                logger.warning(f"Nie znaleziono pliku statystyk dla {model_path_to_resume}. Kontynuowanie z nowymi statystykami.")

    with mp.Manager() as manager:
        logger.info("Starting environment generation manager...")
        generation_queue = manager.Queue(maxsize=args.n_envs * 4)
        generator = EnvironmentGenerator(queue=generation_queue)
        generator.start()
        model = None

        try:
            n_envs = args.n_envs
            logger.info(f"Creating {n_envs} parallel base environments...")
            base_train_env = make_vec_env(
                args.env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                monitor_dir=str(log_dir),
                env_kwargs={'generation_queue': generation_queue, 'stage': args.stage}
            )
            eval_render_mode = "human" if args.render_eval else None
            base_eval_env = make_vec_env(
                args.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                env_kwargs={'render_mode': eval_render_mode, 'stage': args.stage}
            )

            if stats_path_to_load:
                logger.info(f"Wczytywanie statystyk VecNormalize z: {stats_path_to_load}")
                train_env = VecNormalize.load(str(stats_path_to_load), base_train_env)
                eval_env = VecNormalize.load(str(stats_path_to_load), base_eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                logger.info("Tworzenie nowych statystyk VecNormalize.")
                train_env = VecNormalize(base_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
                eval_env = VecNormalize(base_eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

            eval_callback = MultiObjectiveEvalCallback(
                eval_env,
                eval_freq=max(args.eval_freq // n_envs, 1),
                best_model_save_path=str(model_dir),
                log_path=str(log_dir),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=args.render_eval,
                verbose=1,
            )

            sync_callback = SyncVecNormalizeCallback(eval_callback)
            callback_list = CallbackList([sync_callback, eval_callback])

            def linear_schedule(initial_value: float, end_value: float) -> \
            Callable[[float], float]:
                def func(progress_remaining: float) -> float:
                    return end_value + (
                                initial_value - end_value) * progress_remaining

                return func

            logger.info("Configuring schedules based on command-line arguments...")

            def create_schedule(schedule_type: str, start_val: float, end_val: float):
                if schedule_type == 'linear':
                    logger.info(f"Using LINEAR schedule: {start_val} -> {end_val}")
                    return linear_schedule(start_val, end_val)
                elif schedule_type == 'constant':
                    logger.info(f"Using CONSTANT value: {start_val}")
                    def const_func(_: float) -> float:
                        return start_val
                    return const_func
                else:
                    raise ValueError(f"Invalid schedule type: {schedule_type}")

            lr_schedule = create_schedule(args.lr_schedule_type, args.lr_start, args.lr_end)
            clip_schedule = create_schedule(args.clip_schedule_type, args.clip_start, args.clip_end)

            policy_kwargs = dict(
                net_arch=args.net_arch
            )

            if model_path_to_resume:
                logger.info(
                    f"Resuming training from model: {model_path_to_resume}")
                eval_callback.eval_env = eval_env
                model = PPO.load(
                    model_path_to_resume,
                    env=train_env,
                    learning_rate=lr_schedule,
                    n_steps=args.n_steps // n_envs,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    ent_coef=args.ent_coef,
                    clip_range=clip_schedule,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=str(PROJECT_ROOT / "logs"),
                )

            else:
                logger.info(
                    "Creating a new PPO model with a custom policy...")
                model = PPO(
                    "MultiInputPolicy",
                    train_env,
                    learning_rate=lr_schedule,
                    n_steps=args.n_steps // n_envs,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    ent_coef=args.ent_coef,
                    clip_range=clip_schedule,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=str(PROJECT_ROOT / "logs"),
                )

            logger.info(
                f"Starting a single training run for {args.total_timesteps} timesteps...")
            logger.info(
                "Launch TensorBoard to monitor progress: python scripts/view_logs.py")

            grad_callback = GradientNormCallback(log_freq=1000)

            # Resume logging in the same file if resuming a run, otherwise start fresh.
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=CallbackList([sync_callback, eval_callback, grad_callback]),
                tb_log_name=base_name,
                reset_num_timesteps=not args.resume_run_name
            )


        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                "Training interrupted by user. The final model will be saved in the 'finally' block.")
        finally:
            if model:
                final_model_path = model_dir / "final_model.zip"
                model.save(final_model_path)
                stats_path = model_dir / "final_model_vec_normalize.pkl"
                train_env.save(str(stats_path))
                logger.info(
                    f"Training process finished. Final model saved to: {final_model_path} and stats to: {stats_path}"
                )

                best_model_path = model_dir / "best_model.zip"
                if best_model_path.exists():
                    logger.info(
                        f"Loading best model from {best_model_path} for final evaluation...")

                    best_stats_path = best_model_path.parent / "best_model_vec_normalize.pkl"
                    if not best_stats_path.exists():
                         raise FileNotFoundError(f"Could not find stats for best model at {best_stats_path}")

                    eval_env_for_final_eval = VecNormalize.load(str(best_stats_path), eval_env)
                    eval_env_for_final_eval.training = False

                    best_model = PPO.load(best_model_path, env=eval_env_for_final_eval)

                    mean_reward, std_reward = evaluate_policy(best_model,
                                                              eval_env_for_final_eval,
                                                              n_eval_episodes=50,
                                                              deterministic=True)
                    logger.info(
                        f"Final evaluation of BEST model -> Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                else:
                    logger.warning(
                        "No best model was saved during training. Skipping final evaluation.")

            stop_generator(generator)


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for the Urban Planner environment.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="An optional name for this training run. If not provided, a timestamp will be used.")
    parser.add_argument("--resume-run-name", type=str, default=None,
                        help="Name of a previous run to resume training from its final state, continuing in the same log directory.")
    parser.add_argument("--resume-from-model", type=str, default=None,
                        help="Path to a .zip model file to resume training from, relative to the 'models' directory.")
    parser.add_argument("--stage", type=int, default=3,
                        choices=[1, 2, 3, 4],
                        help="Set the training stage.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments to use.")
    parser.add_argument("--env-id", type=str, default="UrbanPlanner-v0",
                        help="Gymnasium environment ID.")
    parser.add_argument("--total-timesteps", type=int, default=1_024_000_000,
                        help="Total number of training steps for the entire run.")
    parser.add_argument("--n-steps", type=int, default=8192,
                        help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="The minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=2,
                        help="Number of epochs to run per PPO update (default: 4).")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="The discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Factor for trade-off of bias vs variance for GAE.")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient for the loss calculation.")
    parser.add_argument('--lr-schedule-type', type=str, default='linear', choices=['linear', 'constant'],
                        help='Schedule type for learning rate.')
    parser.add_argument('--lr-start', type=float, default=2e-3, help='Initial learning rate.')
    parser.add_argument('--lr-end', type=float, default=7e-4, help='Final learning rate (for linear schedule).')

    parser.add_argument('--clip-schedule-type', type=str, default='linear', choices=['linear', 'constant'],
                        help='Schedule type for clip range.')
    parser.add_argument('--clip-start', type=float, default=0.2, help='Initial clip range.')
    parser.add_argument('--clip-end', type=float, default=0.11, help='Final clip range (for linear schedule).')

    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (constant value).")
    parser.add_argument("--net-arch", type=int, nargs='+', default=[256, 256],
                        help="Network architecture.")
    parser.add_argument("--eval-freq", type=int, default=100_000,
                        help="Evaluate the agent every n steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=24,
                        help="Number of episodes to use for evaluation.")
    parser.add_argument("--render-eval", action="store_true",
                        help="If set, render the environment during evaluation phases.")

    args = parser.parse_args()
    setup_logging()

    if args.resume_run_name and (args.run_name or args.resume_from_model):
        parser.error("--resume-run-name cannot be used with --run-name or --resume-from-model.")

    try:
        train_agent(args)
        logger.info("Script finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}",
                        exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass
    main()