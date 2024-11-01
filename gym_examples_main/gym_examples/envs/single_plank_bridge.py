import gymnasium as gym
import math
from gymnasium import spaces
import numpy as np
from shapely.geometry import Point, Polygon

def bezier_curve(points, num_points=100):
    n = points.shape[0] - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += binomial_coeff(n, j) * (t[i]**j) * ((1 - t[i])**(n - j)) * points[j]
    return curve

def binomial_coeff(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def dist_to_edge(source, polygon):
    distance = polygon.exterior.distance(source)

def point_in_shape(source, polygon):
    return polygon.contains_point(source)

class BridgeEnv(gym.Env):

    def __init__(self, bridge_length=100, distortion=0.9, agent=None):
        super().__init__()
        self.agent = agent
        self.bridge_length = bridge_length
        primal_segment = 20
        central = np.arange(0, bridge_length, bridge_length/primal_segment, dtype=float)[:, np.newaxis]
        # print(central)
        primal_shift = np.random.uniform(-1, 1, size=(primal_segment, 1)) * bridge_length * distortion / 2
        primal_shift[0] = primal_shift[-1] = 0.0
        central = np.concatenate((central, primal_shift), axis=1)
        print(central)
        curve   = bezier_curve(central)
        # print(curve)
        v_left  = curve - np.random.uniform(3, 3 + distortion * 2, size=(bridge_length, 2)) * np.array([0, 1])
        v_right = np.flip(curve + np.random.uniform(3, 3 + distortion * 2, size=(bridge_length, 2)) * np.array([0, 1]), axis=0)
        verts   = np.concatenate((v_left, v_right), axis=0)
        # print(verts)
        self.surface = Polygon(verts)
        self.start = verts[0, :]

        self._action_to_direction = {
            0: np.array([0.0, 0.0], dtype=np.double),
            1: np.array([0.1, 1.0], dtype=np.double),
            2: np.array([0.1, -1.0], dtype=np.double),
            3: np.array([1.0, 0.0], dtype=np.double),
        }
        self._action_to_penalty = {
            0: 0.0,
            1: 0.001,
            2: 0.001,
            3: 0.015
        }

    def _get_obs(self):
        return self.agent.view().flatten()

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.agent.tp(self.start)

        observation = self.agent.view().flatten()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        movement = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.agent.walk(action)
        reward = -self._action_to_penalty[action]
        # An episode is done iff the agent has reached the target
        terminated = not point_in_shape(Point(self.agent.location), self.surface)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

if __name__ == "__main__":
    BE = BridgeEnv()
