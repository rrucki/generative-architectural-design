import gymnasium as gym
import numpy as np


class ClipReward(gym.RewardWrapper):
    """
    A reward wrapper that clips the reward to a specified range.

    This is useful for preventing excessively large or small reward values,
    which can sometimes destabilize the learning process.
    """
    def __init__(self, env: gym.Env, min_reward: float, max_reward: float):
        """
        Initializes the ClipReward wrapper.

        Args:
            env: The environment to wrap.
            min_reward: The minimum value to clip the reward to.
            max_reward: The maximum value to clip the reward to.
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: float) -> float:
        """Clips the reward to the defined min/max range."""
        return np.clip(reward, self.min_reward, self.max_reward)