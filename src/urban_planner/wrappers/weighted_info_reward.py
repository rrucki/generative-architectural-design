from typing import Dict, Any

import gymnasium as gym


class WeightedInfoReward(gym.Wrapper):
    """
    A generic wrapper that recalculates the reward based on a weighted sum
    of values found in the `info` dictionary.

    This allows for flexible reward shaping without modifying the core
    environment's reward logic. The environment must be configured to
    return the necessary values in its `info` dict.
    """

    def __init__(self, env: gym.Env, reward_weights: Dict[str, float]):
        """
        Initializes the WeightedInfoReward wrapper.

        Args:
            env: The environment to wrap.
            reward_weights: A dictionary mapping keys in the `info` dict
                            to their corresponding weight in the reward sum.
                            Example: {"distance_traveled": 0.8, "energy_used": -0.2}
        """
        super().__init__(env)
        self.reward_weights = reward_weights

    def step(self, action: Any):
        """
        Executes a step and computes the new weighted reward.
        """
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = sum(
            weight * info.get(key, 0) for key, weight in
            self.reward_weights.items()
        )

        return obs, reward, terminated, truncated, info