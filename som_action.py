import numpy as np
import random
class SOM():
    def __init__(self,  weight_dim, width, height,learning_rate,lamda,epsilon,decay_factor=0.99):
        self.weight_dim = weight_dim
        self.width = width
        #self.grid = np.random.rand(height, width, weight_dim)
        self.lr = learning_rate
        self.lamda = lamda
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        
        direction = [-15,-5,0,5,15]
        self.grid = np.array([[[random.uniform(0,5),direction[i]] for i in range(width)] for _ in range(height)])
        #print("shape of grid is ",self.grid.shape)
        
    def choose_proposed_action(self,unit):
        col = unit % self.width 
        row = unit // self.width if col != 0 else unit // self.width - 1
        
        return self.grid[row,col-1,:]
        
    def perturbed_action(self,unit):
        proposed_action = self.choose_proposed_action(unit)
        noise = np.random.uniform(-1, 1, self.weight_dim) * self.epsilon
          
        return proposed_action + noise
        
    def update_weights(self,action,iteration):
    
        learning_rate = self.lr * (self.decay_factor ** iteration)
        lamda = self.lamda * (self.decay_factor ** iteration)
        
        for i in range (self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                
                distance_to_bmu = np.linalg.norm(self.grid[i,j]-action)
                neighborhood_func = np.exp(- distance_to_bmu**2 / (2 * lamda **2))

                delta = learning_rate * neighborhood_func * (action - self.grid[i, j])
                self.grid[i, j] += delta
                
        
        
    