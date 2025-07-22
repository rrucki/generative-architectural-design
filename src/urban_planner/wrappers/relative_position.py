import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from typing import cast

class RelativePosition(gym.ObservationWrapper):
    """
    An observation wrapper that computes the relative position between
    an "agent" and a "target" observation.

    It assumes the input observation is a dictionary containing 'agent' and
    'target' keys and returns their element-wise difference.
    """
    def __init__(self, env: gym.Env):
        """
        Initializes the RelativePosition wrapper.

        It automatically infers the shape of the observation space from the
        underlying environment's observation space and performs safety checks.
        """
        super().__init__(env)

        if not isinstance(self.observation_space, Dict):
            raise TypeError(
                f"RelativePosition wrapper requires a Dict observation space, "
                f"but got {type(self.observation_space).__name__}."
            )

        obs_space = cast(Dict, self.observation_space)

        required_keys = {"agent", "target"}
        if not required_keys.issubset(obs_space.spaces):
            raise ValueError(
                f"RelativePosition wrapper requires 'agent' and 'target' keys "
                f"in the observation space. Found: {list(obs_space.spaces.keys())}"
            )

        agent_space = obs_space.spaces["agent"]
        target_space = obs_space.spaces["target"]

        if agent_space.shape != target_space.shape:
            raise ValueError(
                f"The 'agent' and 'target' observation spaces must have the same shape. "
                f"Got agent: {agent_space.shape}, target: {target_space.shape}"
            )

        self.observation_space = Box(
            shape=agent_space.shape,
            low=-np.inf,
            high=np.inf,
            dtype=agent_space.dtype
        )

    def observation(self, obs: dict) -> np.ndarray:
        """Returns the difference between the target and agent observations."""
        return obs["target"] - obs["agent"]