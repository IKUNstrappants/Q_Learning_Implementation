import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from DDPG.DDPG_model import (Actor, Critic)
from collections import namedtuple, deque
import random
from .random_noise import OrnsteinUhlenbeckProcess
from .memory import SequentialMemory

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    #"mps" if torch.backends.mps.is_available() else
    "cpu"
)
USE_CUDA = torch.cuda.is_available()

# from ipdb import set_trace as debug
def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray):
    return torch.from_numpy(ndarray).to(device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))  
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, hidden1, hidden2, init_w, learning_rate, noise_theta, noise_mu, noise_sigma, batch_size, tau, discount, epsilon):
        
        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.nb_actions, hidden1, hidden2, init_w)
        self.actor_target = Actor(self.nb_states, self.nb_actions, hidden1, hidden2, init_w)
        self.actor_optim  = Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(self.nb_states, self.nb_actions, hidden1, hidden2, init_w)
        self.critic_target = Critic(self.nb_states, self.nb_actions, hidden1, hidden2, init_w)
        self.critic_optim  = Adam(self.critic.parameters(), lr=learning_rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory1 = ReplayMemory(10000)
        self.memory2 = SequentialMemory(limit=6000000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=noise_theta, mu=noise_mu, sigma=noise_sigma)

        # Hyper-parameters
        self.batch_size = batch_size
        self.tau = tau
        self.discount = discount
        self.depsilon = 1.0 / epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory1.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_state_batch = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state).to(dtype=torch.float32)
        #print(state_batch[0].dtype)
        # print("action is \n",batch.action)
        action_batch = torch.cat((batch.action)).to(dtype=torch.float32)
        # print(action_batch.size())
        reward_batch = torch.cat(batch.reward).to(dtype=torch.float32)
        
        next_q_values = torch.zeros(self.batch_size, device=device)
        

        # Prepare for the target q batch
        next_q_values[non_final_mask] = self.critic_target([
            non_final_next_state_batch,
            self.actor_target(non_final_next_state_batch),
        ]).squeeze(-1)


        target_q_batch = reward_batch + self.discount  * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([state_batch, action_batch])
        #print(f"\n\nQ value is {q_batch}\n\n")
        
        value_loss = criterion(q_batch, target_q_batch)
        print(value_loss)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            state_batch,
            self.actor(state_batch) # a = \miu(s)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1):
        if self.is_training:
            self.memory1.push(self.s_t, self.a_t, s_t1, r_t)
            self.s_t = s_t1
            
    def observe2(self, r_t, s_t1, done):
        if self.is_training:
            self.memory2.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1
            
    def random_action(self):
        action = np.random.uniform(-1.,1.,(1, self.nb_actions)) * np.array([3, 1])
        self.a_t = to_tensor(action).to(device=device, dtype=torch.float32)
        #print(f"random choose {self.a_t.size()}\n")
        return self.a_t

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(to_numpy(s_t)))
        )
        action += self.is_training * max(self.epsilon, 0)* self.random_process.sample()
        # action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = to_tensor(action).to(device=device, dtype=torch.float32) * torch.tensor([3,1], device=device,dtype=torch.float32)#.reshape(-1)
        # print(f"actor choose {self.a_t.size()}\n")
        return self.a_t

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()
        
    def update_policy2(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory2.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch),
            self.actor_target(to_tensor(next_state_batch)),
        ])
        

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float32))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        q_batch = q_batch.float() if isinstance(q_batch, torch.Tensor) else torch.tensor(q_batch, dtype=torch.float32)
        target_q_batch = target_q_batch.float() if isinstance(target_q_batch, torch.Tensor) else torch.tensor(target_q_batch, dtype=torch.float32)
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss = value_loss.float()
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch)) # a = \miu(s)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
