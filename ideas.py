import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Brain:
    def __init__(self, AI):
        self.AI=AI(4)

    def select_action(self, viewCache):
        eval = self.AI(torch.flatten(viewCache))
        command = torch.argmax(eval).item()
        if   command == 0: return 0., 0.
        elif command == 1: return 1., 0.
        elif command == 2: return 0., 1.
        elif command == 3: return 0.,-1.

class visionNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcl1 = nn.Linear(3, 5)
        self.act = nn.LeakyReLU()
        # self.fcl2 = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fcl1(x)
        x = self.act(x)
        return x

class IdleAI:
    def __init__(self, *args):
        pass
    def forward(self):
        return torch.tensor([1, 0, 0, 0])

class PredatorAI(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.fcl1 = nn.Linear(14, 128)
        self.fcl2 = nn.Linear(128, 64)
        self.fcl3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fcl1(x))
        x = F.leaky_relu(self.fcl2(x))
        x = F.leaky_relu(self.fcl3(x))
        return x


class PreyAI(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.fcl1 = nn.Linear(21, 128)
        self.fcl2 = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fcl1(x))
        x = F.leaky_relu(self.fcl2(x))
        return x