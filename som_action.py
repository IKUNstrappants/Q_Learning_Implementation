import numpy as np
import random
import torch
from utilities import *
from math import exp

'''device = torch.device(
    #"cuda" if torch.cuda.is_available() else
    #"mps" if torch.backends.mps.is_available() else
    "cpu"
)'''

class SOM():
    def __init__(self,  weight_dim=2, n_actions=25, learning_rate=0.003, lamda=0.5, epsilon=1, decay_factor=0.99, margin=0.5):
        self.weight_dim = weight_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.lamda = lamda
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.margin = margin

        grid_x = torch.tensor([-5, 0, 5, 10, 15], dtype=torch.float32).repeat(5)
        grid_y = torch.tensor([[-15, -7.5, 0, 7.5, 15]], dtype=torch.float32).repeat(5, 1).T.flatten()
        # print(grid_y, grid_x)
        self.grid = torch.cat((grid_x[:, None], grid_y[:, None]), dim=1).to(device=device())
        # print(self.grid)

    def perturbed_action(self,unit):
        proposed_action = self.grid[unit,:]
        if self.lr <= 0:
            return proposed_action
        noise = (torch.rand(self.weight_dim, device=device()) - 0.5) * 2 * self.epsilon
        return proposed_action + noise
        
    def update_weights(self,action,iteration):
        if self.lr <= 0:
            return
        learning_rate = self.lr * (self.decay_factor ** iteration)
        lamda = self.lamda * (self.decay_factor ** iteration)
        for i in range(self.grid.shape[0]):
            distance_to_bmu = vectorDistance(self.grid[i], action).item() / 50
            if distance_to_bmu <= self.margin:
                continue
            neighborhood_func = exp(- distance_to_bmu**2 / (2 * lamda **2))
            delta = learning_rate * neighborhood_func * (action - self.grid[i])
            self.grid[i] = self.grid[i] + delta

class CAM():
    def __init__(self,  weight_dim=2, learning_rate=0.003):
        self.weight_dim = weight_dim
        self.lr = learning_rate

        grid_x = torch.tensor([-20, 30], dtype=torch.float32).repeat(2)
        grid_y = torch.tensor([[-30., 30.]], dtype=torch.float32).T.repeat(2, 1).T.flatten()
        self.grid = torch.cat((grid_x[:, None], grid_y[:, None]), dim=1).to(device=device())

    def propose_action(self, weight):
        return torch.sum(self.grid * weight.T, dim=0)

    def random_action(self):
        random_weights = torch.rand(self.grid.shape[0])
        weights = random_weights / random_weights.sum()
        return torch.sum(self.grid * weights[:, None], dim=0), weights.reshape(1, -1)

if __name__ == "__main__":
    som = SOM()
    som.update_weights(torch.tensor([0, 0], dtype=torch.float32), 1)