import numpy as np
import random
class SOM():
    def __init__(self,  weight_dim=2, n_actions=25, learning_rate=0.003, lamda=0.5, epsilon=1, decay_factor=0.99):
        self.weight_dim = weight_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.lamda = lamda
        self.epsilon = epsilon
        self.decay_factor = decay_factor

        grid_x = np.array([-3, 0, 3, 6, 9], dtype=float).repeat(5)
        grid_y = np.array([[-10, -5, 0, 5, 10]], dtype=float).T.repeat(5, axis=1).T.flatten()
        self.grid = np.concatenate((grid_x[:, np.newaxis], grid_y[:, np.newaxis]), axis=1)

    def perturbed_action(self,unit):
        proposed_action = self.grid[unit,:]
        noise = np.random.uniform(-1, 1, self.weight_dim) * self.epsilon

        return proposed_action + noise
        
    def update_weights(self,action,iteration):
    
        learning_rate = self.lr * (self.decay_factor ** iteration)
        lamda = self.lamda * (self.decay_factor ** iteration)
        
        for i in range(self.grid.shape[0]):
            distance_to_bmu = np.linalg.norm(self.grid[i]-action)
            neighborhood_func = np.exp(- distance_to_bmu**2 / (2 * lamda **2))
            delta = learning_rate * neighborhood_func * (action - self.grid[i])
            self.grid[i] += delta

if __name__ == "__main__":
    som = SOM()