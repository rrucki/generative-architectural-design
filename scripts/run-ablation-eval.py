import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Register environment
import urban_planner
from urban_planner.config import INTERNAL_MAX_STEPS
from urban_planner.logger_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("ablation_eval")


def run_ppo_agent(
    model,
    vec_env,
    raw_env,
    seed: int,
    max_steps: int = INTERNAL_MAX_STEPS
) -> bool:
    """Runs a single episode of a PPO agent and returns if it successfully solved it."""
    # Seed generators to ensure identical procedural generation
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    obs, info = raw_env.reset()
    
    solved = False
    
    for step in range(max_steps):
        # Normalize observation manually for VecNormalize model
        batched_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        norm_obs = vec_env.normalize_obs(batched_obs)
        
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action[0])
        
        if terminated and info.get("is_solution", False):
            solved = True
            break
            
        if truncated:
            break

    return solved


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Run ablation study evaluation (Stage 2 and Stage 3 models under Stage 4 constraints)."
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes."
    )
    
    args = parser.parse_args()
    n_episodes = args.n_episodes
    
    logger.info(f"Starting ablation evaluation with {n_episodes} episodes...")
    
    # 1. Paths to ablation models
    # Stage 2 represents "w/o Width Reward" (only trained containment and area)
    model_path_stage2 = PROJECT_ROOT / "models" / "stage_2_1754945765" / "final_model.zip"
    stats_path_stage2 = PROJECT_ROOT / "models" / "stage_2_1754945765" / "final_model_vec_normalize.pkl"
    
    # Stage 3 represents "w/o Geometry (Angle) Reward" (trained containment, area, and width)
    model_path_stage3 = PROJECT_ROOT / "models" / "stage_3_1756213572" / "final_model.zip"
    stats_path_stage3 = PROJECT_ROOT / "models" / "stage_3_1756213572" / "final_model_vec_normalize.pkl"
    
    # Verify existences
    for path in [model_path_stage2, stats_path_stage2, model_path_stage3, stats_path_stage3]:
        if not path.exists():
            logger.critical(f"Required path does not exist: {path}")
            sys.exit(1)
            
    # Setup environments (always evaluate under Stage 4: full constraints)
    raw_env = gym.make("UrbanPlanner-v0", stage=4)
    
    # Load Stage 2 model
    dummy_s2 = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_env_s2 = VecNormalize.load(str(stats_path_stage2), dummy_s2)
    vec_env_s2.training = False
    vec_env_s2.norm_reward = False
    model_s2 = PPO.load(model_path_stage2, env=vec_env_s2)
    
    # Load Stage 3 model
    dummy_s3 = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_env_s3 = VecNormalize.load(str(stats_path_stage3), dummy_s3)
    vec_env_s3.training = False
    vec_env_s3.norm_reward = False
    model_s3 = PPO.load(model_path_stage3, env=vec_env_s3)

    results = {
        "stage_2": [],
        "stage_3": []
    }
    
    # Run loop
    for i in range(n_episodes):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Evaluating episode {i + 1}/{n_episodes}...")
            
        # Run Stage 2 model
        solved_s2 = run_ppo_agent(model_s2, vec_env_s2, raw_env, seed=i)
        results["stage_2"].append(solved_s2)
        
        # Run Stage 3 model
        solved_s3 = run_ppo_agent(model_s3, vec_env_s3, raw_env, seed=i)
        results["stage_3"].append(solved_s3)

    # Compute success rates
    success_rate_s2 = (sum(results["stage_2"]) / n_episodes) * 100.0
    success_rate_s3 = (sum(results["stage_3"]) / n_episodes) * 100.0
    
    logger.info("\n=================== Ablation Study Results ===================")
    logger.info(f"w/o Width Reward (Stage 2 Model) Feasibility Ratio: {success_rate_s2:.2f}%")
    logger.info(f"w/o Geometry Reward (Stage 3 Model) Feasibility Ratio: {success_rate_s3:.2f}%")
    
    # Save to JSON
    out_file = PROJECT_ROOT / "logs" / "ablation_evaluation_results.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "stage_2_success_rate": success_rate_s2,
            "stage_3_success_rate": success_rate_s3
        }, f, indent=4)
    logger.info(f"Ablation results saved to: {out_file}")


if __name__ == "__main__":
    main()
