import gymnasium as gym
import numpy
from gymnasium import spaces
import numpy as np


class AnimalEnv(gym.Env):
    def __init__(self, animal=None):
        self.animal = animal

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Discrete(1)

        # We have 4 actions, corresponding to "stay", "right", "forward", "left"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0.0, 0.0], dtype=numpy.double),
            1: np.array([0.1, 1.0], dtype=numpy.double),
            2: np.array([0.1,-1.0], dtype=numpy.double),
            3: np.array([1.0, 0.0], dtype=numpy.double),
        }
        self._action_to_penalty = {
            0: 0.0,
            1: 0.001,
            2: 0.001,
            3: 0.015
        }

    def _get_obs(self):
        return self.animal.view().flatten()

    def _get_info(self):
        return {
            "kills": 0,
            "lifetime": 0
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # We will sample the target's location randomly until it does not coincide with the agent's location

        observation = self.animal.view().flatten()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        movement = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.animal.walk(movement, action)
        reward = -self._action_to_penalty[action]
        if self.animal.type == 1:
            reward += self.animal.hunt()
        elif self.animal.type == 2:
            reward += 0.002
        # An episode is done iff the agent has reached the target
        terminated = not self.animal.alive
        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
