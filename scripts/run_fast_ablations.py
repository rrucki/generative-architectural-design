"""
run_fast_ablations.py
---------------------
Controlled RL ablation study comparing curriculum granularities and policy optimization algorithms:
1. 4-Stage CL-PPO (Stage 1 -> Stage 2 -> Stage 3 -> Stage 4)
2. 2-Stage CL-PPO (Stage 2 [Placement+Area] -> Stage 4 [All 4 Constraints])
3. 1-Stage PPO No-CL (Stage 4 from scratch)
4. 4-Stage CL-A2C (Advantage Actor-Critic with identical [256, 256] architecture)

Also logs policy/value L2 gradient norms via GradientNormCallback to quantify
gradient variance and destructive interference across curriculum configurations.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner
from urban_planner.utils.callbacks import GradientNormCallback
from scripts.baselines.objective import _segments_intersect


def evaluate_policy_on_stage(
    model: Any,
    vec_env: VecNormalize,
    stage: int = 4,
    n_episodes: int = 100,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    raw_env = gym.make("UrbanPlanner-v0", stage=stage)
    unwrapped = raw_env.unwrapped
    solved = 0
    rewards = []
    placement_ok = 0
    area_ok = 0
    width_ok = 0
    angle_ok = 0

    from urban_planner.types import Stage as St

    for i in range(n_episodes):
        s = seed_offset + i
        random.seed(s)
        np.random.seed(s)
        obs, _ = raw_env.reset(seed=s)
        ep_rew = 0.0
        ep_solved = False
        for _ in range(256):
            batched = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            norm_obs = vec_env.normalize_obs(batched)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = raw_env.step(action[0])
            ep_rew += float(r)
            if term or trunc:
                coords = unwrapped.assessment.current_outline_coords
                p0, p1, p2, p3 = coords
                invalid = _segments_intersect(p0, p1, p2, p3) or _segments_intersect(p1, p2, p3, p0)
                if info.get("is_solution", False) and not invalid:
                    ep_solved = True
                break
        rewards.append(ep_rew)
        if ep_solved:
            solved += 1
        ck = unwrapped.assessment.checklist
        if ck[St.PLACEMENT]:
            placement_ok += 1
        if ck[St.AREA]:
            area_ok += 1
        if ck[St.WIDTH]:
            width_ok += 1
        if ck[St.ANGLE]:
            angle_ok += 1

    return {
        "feasibility_pct": (solved / n_episodes) * 100.0,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "placement_rate_pct": (placement_ok / n_episodes) * 100.0,
        "area_rate_pct": (area_ok / n_episodes) * 100.0,
        "width_rate_pct": (width_ok / n_episodes) * 100.0,
        "angle_rate_pct": (angle_ok / n_episodes) * 100.0,
    }


def make_vec_env(stage: int, n_envs: int = 8, prev_vec_env: VecNormalize = None) -> VecNormalize:
    venv = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=stage) for _ in range(n_envs)])
    norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if prev_vec_env is not None:
        norm.obs_rms = prev_vec_env.obs_rms
    return norm


def run_ablation_suite(total_steps: int = 128_000, n_eval: int = 100) -> Dict[str, Any]:
    torch.set_num_threads(2)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    results = {}

    from urban_planner.domain.geometries import Plot
    orig_from_random = Plot.from_random_generation
    print("Pre-generating pool of 64 training parcels to accelerate environment resets...")
    random.seed(12345)
    np.random.seed(12345)
    plot_pool = [orig_from_random() for _ in range(64)]
    pool_idx = 0

    def fast_pool_plot(cls):
        nonlocal pool_idx
        p = plot_pool[pool_idx % len(plot_pool)]
        pool_idx += 1
        return p

    def eval_with_real_gen(model, vec_env, stage, n_episodes):
        Plot.from_random_generation = orig_from_random
        try:
            return evaluate_policy_on_stage(model, vec_env, stage=stage, n_episodes=n_episodes)
        finally:
            Plot.from_random_generation = classmethod(fast_pool_plot)

    Plot.from_random_generation = classmethod(fast_pool_plot)

    # 1. 1-Stage PPO (No-CL, Stage 4 directly)
    print(f"\n[1/4] Running 1-Stage PPO (No-CL) for {total_steps} steps...")
    t0 = time.perf_counter()
    env_nocl = make_vec_env(stage=4, n_envs=8)
    cb_nocl = GradientNormCallback()
    model_nocl = PPO(
        "MultiInputPolicy",
        env_nocl,
        n_steps=1024,
        batch_size=512,
        n_epochs=2,
        learning_rate=1e-3,
        ent_coef=0.015,
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=42,
        verbose=0,
    )
    model_nocl.learn(total_timesteps=total_steps, callback=cb_nocl)
    env_nocl.training = False
    env_nocl.norm_reward = False
    eval_nocl = eval_with_real_gen(model_nocl, env_nocl, stage=4, n_episodes=n_eval)
    eval_nocl["wall_clock_s"] = time.perf_counter() - t0
    eval_nocl["mean_grad_norm"] = float(np.mean(cb_nocl.batch_grad_norms)) if cb_nocl.batch_grad_norms else 0.0
    results["1-Stage PPO (No-CL)"] = eval_nocl
    print("  1-Stage PPO (No-CL):", eval_nocl)

    # 2. 2-Stage CL-PPO (Stage 2 [Placement+Area] -> Stage 4 [All 4 Constraints])
    print(f"\n[2/4] Running 2-Stage CL-PPO (Stage 2 -> Stage 4) for {total_steps} steps...")
    t0 = time.perf_counter()
    half_steps = total_steps // 2
    env_2s_1 = make_vec_env(stage=2, n_envs=8)
    cb_2s = GradientNormCallback()
    model_2s = PPO(
        "MultiInputPolicy",
        env_2s_1,
        n_steps=1024,
        batch_size=512,
        n_epochs=2,
        learning_rate=1e-3,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=42,
        verbose=0,
    )
    model_2s.learn(total_timesteps=half_steps, callback=cb_2s)
    s2_mid_eval = eval_with_real_gen(model_2s, env_2s_1, stage=2, n_episodes=n_eval)

    env_2s_2 = make_vec_env(stage=4, n_envs=8, prev_vec_env=env_2s_1)
    model_2s.set_env(env_2s_2)
    model_2s.learn(total_timesteps=half_steps, callback=cb_2s)
    env_2s_2.training = False
    env_2s_2.norm_reward = False
    eval_2s = eval_with_real_gen(model_2s, env_2s_2, stage=4, n_episodes=n_eval)
    eval_2s["stage2_intermediate"] = s2_mid_eval
    eval_2s["wall_clock_s"] = time.perf_counter() - t0
    eval_2s["mean_grad_norm"] = float(np.mean(cb_2s.batch_grad_norms)) if cb_2s.batch_grad_norms else 0.0
    results["2-Stage CL-PPO (S2->S4)"] = eval_2s
    print("  2-Stage CL-PPO:", eval_2s)

    # 3. 4-Stage CL-PPO (Stage 1 -> Stage 2 -> Stage 3 -> Stage 4, matched short budget)
    print(f"\n[3/4] Running 4-Stage CL-PPO (S1->S2->S3->S4) for {total_steps} steps...")
    t0 = time.perf_counter()
    q_steps = total_steps // 4
    prev_env = None
    model_4s = None
    cb_4s = GradientNormCallback()
    stage_metrics = {}
    for st in [1, 2, 3, 4]:
        cur_env = make_vec_env(stage=st, n_envs=8, prev_vec_env=prev_env)
        if model_4s is None:
            model_4s = PPO(
                "MultiInputPolicy",
                cur_env,
                n_steps=1024,
                batch_size=512,
                n_epochs=2,
                learning_rate=1e-3,
                ent_coef=0.02,
                policy_kwargs=policy_kwargs,
                device="cpu",
                seed=42,
                verbose=0,
            )
        else:
            model_4s.set_env(cur_env)
        model_4s.learn(total_timesteps=q_steps, callback=cb_4s)
        stage_metrics[f"stage_{st}"] = eval_with_real_gen(model_4s, cur_env, stage=st, n_episodes=50)
        prev_env = cur_env

    prev_env.training = False
    prev_env.norm_reward = False
    eval_4s = eval_with_real_gen(model_4s, prev_env, stage=4, n_episodes=n_eval)
    eval_4s["per_stage_progression"] = stage_metrics
    eval_4s["wall_clock_s"] = time.perf_counter() - t0
    eval_4s["mean_grad_norm"] = float(np.mean(cb_4s.batch_grad_norms)) if cb_4s.batch_grad_norms else 0.0
    results["4-Stage CL-PPO (Matched Short Budget)"] = eval_4s
    print("  4-Stage CL-PPO (Short):", eval_4s)

    # 4. 4-Stage CL-A2C (Advantage Actor-Critic with identical [256, 256] architecture)
    print(f"\n[4/4] Running 4-Stage CL-A2C (S1->S2->S3->S4) for {total_steps} steps...")
    t0 = time.perf_counter()
    prev_env_a2c = None
    model_a2c = None
    cb_a2c = GradientNormCallback()
    for st in [1, 2, 3, 4]:
        cur_env = make_vec_env(stage=st, n_envs=8, prev_vec_env=prev_env_a2c)
        if model_a2c is None:
            model_a2c = A2C(
                "MultiInputPolicy",
                cur_env,
                n_steps=64,
                learning_rate=7e-4,
                ent_coef=0.015,
                policy_kwargs=policy_kwargs,
                device="cpu",
                seed=42,
                verbose=0,
            )
        else:
            model_a2c.set_env(cur_env)
        model_a2c.learn(total_timesteps=q_steps, callback=cb_a2c)
        prev_env_a2c = cur_env

    prev_env_a2c.training = False
    prev_env_a2c.norm_reward = False
    eval_a2c = eval_with_real_gen(model_a2c, prev_env_a2c, stage=4, n_episodes=n_eval)
    eval_a2c["wall_clock_s"] = time.perf_counter() - t0
    eval_a2c["mean_grad_norm"] = float(np.mean(cb_a2c.batch_grad_norms)) if cb_a2c.batch_grad_norms else 0.0
    results["4-Stage CL-A2C (Matched Short Budget)"] = eval_a2c
    print("  4-Stage CL-A2C (Short):", eval_a2c)

    Plot.from_random_generation = orig_from_random

    # Also include full converged 4-Stage CL-PPO and 64M No-CL from existing checkpoints
    dummy_full = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    vec_full = VecNormalize.load(
        str(PROJECT_ROOT / "models/stage_4_tunning_1757784606/best_model_vec_normalize.pkl"),
        dummy_full,
    )
    vec_full.training = False
    vec_full.norm_reward = False
    model_full = PPO.load(
        PROJECT_ROOT / "models/stage_4_tunning_1757784606/best_model.zip",
        env=vec_full,
        device="cpu",
    )
    results["4-Stage CL-PPO (Full Converged)"] = evaluate_policy_on_stage(
        model_full, vec_full, stage=4, n_episodes=n_eval
    )

    out_file = PROJECT_ROOT / "logs" / "ablation_2stage_and_a2c.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "metadata": {"short_budget_steps": total_steps, "n_eval_episodes": n_eval},
                "results": results,
            },
            f,
            indent=4,
        )
    print(f"\nSaved ablation results to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=320_000)
    parser.add_argument("--n-eval", type=int, default=100)
    args = parser.parse_args()
    run_ablation_suite(total_steps=args.steps, n_eval=args.n_eval)
