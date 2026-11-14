import logging
import os

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

logger = logging.getLogger(__name__)


class SyncVecNormalizeCallback(BaseCallback):
    """
    A callback to synchronize the statistics of the evaluation environment
    with the training environment before each evaluation. This is crucial
    when using VecNormalize to ensure that the model is evaluated on
    the same data distribution it is trained on.

    :param eval_callback: The evaluation callback instance that will be triggered.
    """

    def __init__(self, eval_callback: EvalCallback):
        super().__init__(verbose=0)
        self.eval_callback = eval_callback

    def _on_step(self) -> bool:
        if self.eval_callback.eval_freq > 0 and self.n_calls % self.eval_callback.eval_freq == 0:
            if isinstance(self.training_env, VecNormalize) and isinstance(
                    self.eval_callback.eval_env, VecNormalize):
                self.eval_callback.eval_env.obs_rms = self.training_env.obs_rms
                self.eval_callback.eval_env.ret_rms = self.training_env.ret_rms
                logger.debug("Synced VecNormalize statistics from training to evaluation environment.")
        return True


class MultiObjectiveEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent on two objectives: mean reward and mean episode length.

    Saves the model as "best" if:
    1. The mean reward is higher than the previous best.
    2. The mean reward is the same, but the mean episode length is lower.

    This callback overrides the default _on_step to avoid redundant evaluations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_ep_length = np.inf

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self.evaluations_results) > 0:
                    np.savez(
                        self.log_path,
                        timesteps=self.evaluations_timesteps,
                        results=self.evaluations_results,
                        ep_lengths=self.evaluations_length,
                        **kwargs,
                    )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose > 0:
                logger.info(f"Eval num_timesteps={self.num_timesteps}, "
                            f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.model.logger.record("eval/mean_reward", self.last_mean_reward)
            self.model.logger.record("eval/mean_ep_length", mean_ep_length)

            is_new_best = False
            if mean_reward > self.best_mean_reward:
                is_new_best = True
                logger.info(f"New best mean reward: {mean_reward:.2f} > {self.best_mean_reward:.2f}")
            elif mean_reward == self.best_mean_reward:
                if mean_ep_length < self.best_mean_ep_length:
                    is_new_best = True
                    logger.info(
                        f"Same mean reward ({mean_reward:.2f}), but new best episode length: "
                        f"{mean_ep_length:.2f} < {self.best_mean_ep_length:.2f}"
                    )

            if is_new_best:
                self.best_mean_reward = mean_reward
                self.best_mean_ep_length = mean_ep_length
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model"))
                    if isinstance(self.training_env, VecNormalize):
                        self.training_env.save(
                            os.path.join(self.best_model_save_path,
                                         "best_model_vec_normalize.pkl"))
                    if self.verbose > 0:
                        logger.info(
                            f"Saving new best model to {self.best_model_save_path}")

        return True


class GradientNormCallback(BaseCallback):
    """
    Logs the L2 norm of the policy gradients to TensorBoard and CSV.
    Uses an optimizer pre-step hook so gradient norms are captured during actual
    PPO/A2C minibatch updates rather than during no_grad rollout collection.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._batch_grad_norms = []
        self.batch_grad_norms = []
        self._hook_handle = None

    def _on_training_start(self) -> None:
        def _pre_step_hook(optimizer, args, kwargs):
            total_norm_sq = 0.0
            has_grad = False
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    has_grad = True
                    param_norm = p.grad.detach().data.norm(2).item()
                    total_norm_sq += param_norm * param_norm
            if has_grad:
                self._batch_grad_norms.append(total_norm_sq ** 0.5)

        if hasattr(self.model.policy, "optimizer") and hasattr(
            self.model.policy.optimizer, "register_step_pre_hook"
        ):
            self._hook_handle = self.model.policy.optimizer.register_step_pre_hook(
                _pre_step_hook
            )

    def _on_rollout_start(self) -> None:
        if self._batch_grad_norms:
            mean_norm = float(np.mean(self._batch_grad_norms))
            self.batch_grad_norms.append(mean_norm)
            self.logger.record("train/policy_grad_norm", mean_norm)
            self._batch_grad_norms.clear()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None