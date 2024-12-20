import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import *

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.LayerNorm(hidden1)
        self.bn2 = nn.LayerNorm(hidden2)
        self.init_weights(init_w)
        # self.scale = nn.Parameter(torch.tensor([ 8., 10.], device=device()))
        # self.bias  = nn.Parameter(torch.tensor([3.], device=device()))
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        # out = out * self.scale # + self.bias
        # out[0] = out[0] + self.bias
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.LayerNorm(hidden1)
        self.bn2 = nn.LayerNorm(hidden2)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x) # 先对状态进行特征提取
        out = self.bn1(out)
        out = self.relu(out)
        #print("out size is\n", out.size())
        #print(a.size())
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out