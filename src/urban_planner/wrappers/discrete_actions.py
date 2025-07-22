from typing import List, Any

import gymnasium as gym
from gymnasium.spaces import Discrete


class DiscreteActions(gym.ActionWrapper):
    """
    An action wrapper that converts discrete actions into continuous or complex actions.

    This is useful when you want to use an algorithm that requires a Discrete
    action space, but your environment's natural action space is continuous
    or structured.
    """
    def __init__(self, env: gym.Env, disc_to_cont: List[Any]):
        """
        Initializes the DiscreteActions wrapper.

        Args:
            env: The environment to wrap.
            disc_to_cont: A list where the index represents the discrete action
                          and the value is the corresponding action to be passed
                          to the underlying environment.
        """
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act: int) -> Any:
        """Maps the discrete action `act` to its corresponding complex action."""
        return self.disc_to_cont[act]