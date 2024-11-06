from ideas import PredatorAI
from animal_scene import grassland
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import namedtuple, deque
from itertools import count
from som_action import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame

environment = grassland(num_hunter=1, num_prey=100, num_OmegaPredator=15, size=100)
# set up matplotlib
plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    #"mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

class example_DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

hunter = environment.hunters[0]
# Get number of actions from gym action space
n_actions = hunter.perception.action_space.n
# Get the number of state observations
state, info = hunter.perception.reset()
n_observations = len(state)

policy_net = PredatorAI(n_actions).to(device)
target_net = PredatorAI(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[hunter.perception.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
score_cache = []

def plot_durations(show_result=False):
    fig = plt.figure(1, figsize=(12, 5))
    fig.clf()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.2, 1], height_ratios=[1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 2])
    score = torch.tensor(score_cache, dtype=torch.float)
    if show_result:
        ax1.set_title('Result')
    else:
        ax1.set_title('Training...')
    ax1.plot(score.numpy())
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('reward')
    # Take 50 episode averages and plot them too
    if len(score) >= 50:
        means = score.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax1.plot(means.numpy())

    scatter = som.grid
    ax2.set_title('Self Organizing Map')
    ax2.set_xlabel('forward')
    ax2.set_ylabel('rotation')
    ax2.scatter(scatter[:, 0], scatter[:, 1], c=np.arange(scatter.shape[0]), cmap='viridis')

    plt.pause(1)  # pause a bit so that plots are updated
    '''
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    '''


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


max_iter = 600

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 400
else:
    num_episodes = 50

pygame.init()


flag = 1 # implement som or not
som = None
if flag:
    som = SOM(weight_dim=2,learning_rate=0.2,lamda=0.5,epsilon=1,decay_factor=0.99)

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    environment.reset()
    state, info = hunter.perception.reset()
    # print(state, type(state))
    state = torch.tensor(state.clone(), dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # if i_episode % 10 == 0:
        environment._update_possessed_entities()
        environment._render_frame()
        done = False
        if t >= max_iter: done=True
        action = select_action(state)
        # print("action is \n",action)
        if flag:
            continuous_action = som.perturbed_action(action.item()) # step 3 and 4 in the paper
            # print("action is ",continuous_action)
            observation, reward, terminated, truncated, _ = hunter.perception.continuous_step(continuous_action) # use continuous action to obtain the reward
        else:
            observation, reward, terminated, truncated, _ = hunter.perception.step(action.item())
        
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated or done

        if flag:
            current_state_action_value = policy_net(state).gather(1,action)
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            if flag:
                next_state_value = target_net(next_state).max(1).values
                # Compute the expected Q values
                expected_state_action_value = (next_state_value * GAMMA) + reward
                
                if expected_state_action_value > current_state_action_value:
                    som.update_weights(continuous_action,t)
                
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            score_cache.append(hunter.score)
            plot_durations()
            # if i_episode % 10 == 0:
            environment.close()
            print(f"{i_episode}th episode: {t} iterations, end up with {hunter.score} reward")
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

pygame.quit()