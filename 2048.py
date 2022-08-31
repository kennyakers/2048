'''
This algorithm is based on DeepMind's "Human-level control through deep reinforcement learning" paper,
specifically Algorithm 1: Deep Q-learning with experience replay: https://rdcu.be/cOL7K

The idea here is to use two separate DQNs: one "online" network Q whose q-values are continuously updated,
and one "target" network Q^ whose q-values are updated every target_param_update_freq steps to generate the target
Q-values y_j used in the update rule for the online network. More precisely from the paper: "every C updates we
clone the network Q to obtain a target network Q^ and use Q^ for generating the
Q-learning targets yj forthe followingC updatesto Q. This modification makesthe
algorithm more stable compared to standard online Q-learning, where an update
thatincreasesQ(st,at) often also increasesQ(st 1 1,a)for all a and hence also increases
the target yj, possibly leading to oscillations or divergence of the policy. Generating
the targets using an olderset of parameters adds a delay between the time an update
to Q is made and the time the update affects the targets yj, making divergence or
oscillations much more unlikely."
'''

from collections import deque
import itertools
import logging
import sys
import torch
from tqdm import trange
from dqn import DQN
from game import Game, Action
from random import sample
from torch import nn
import numpy as np

# Hyperparamters
gamma = 0.99
batch_size = 1
learning_rate = 0.0005
transition_buffer_max_size = 50000
min_replay_size = 1000 # Min number of transitions to have stored before computing gradients
target_param_update_freq = 1000 # Number of steps where we set the target parameters to the online parameters
epsilon_start = 1.0 # We use annealing epsilon-greedy to transition from exploring to exploiting as the agent trains
epsilon_end = 0.02
epsilon_decay_num_steps = 10000 # Over how many steps to decay from start to end

# Initialize both networks & optimizer
online_network = DQN(batch_size)
target_network = DQN(batch_size)
target_network.load_state_dict(online_network.state_dict()) # Set parameters to the same
optimizer = torch.optim.Adam(online_network.parameters(), lr=learning_rate)

# Initialize memory buffers
episode_reward = 0.0
reward_memory = deque([episode_reward], maxlen=100)
replay_memory = deque(maxlen=transition_buffer_max_size) # Called "D" in the paper

game = Game()
loss_function = nn.SmoothL1Loss() # Huber Loss
logging.getLogger().setLevel(logging.INFO)
logging.info("Initializing experience replay")

def batch_to_tensor(given_batch, action_batch=False):
    dtype = torch.long if action_batch else torch.float32
    batch = list(map(lambda x: torch.tensor(x, dtype=dtype).unsqueeze(0).unsqueeze(0), given_batch))
    return torch.cat(batch, 0)

for _ in trange(min_replay_size):
    action = sample(list(Action.MOVES.values()), 1)[0]

    # Reward is the sum of the numbers on all merged tiles. Source: https://arxiv.org/pdf/2110.10374.pdf
    old_board = game.board
    old_sum = np.sum(old_board)
    game.move(action)
    new_sum = np.sum(game.board)
    reward = new_sum - old_sum
    done = game.is_done()

    transition = (old_board, action, reward, done, game.board)
    replay_memory.append(transition) # Save this transition

    if done:
        game.reset()

# Training
game.reset()
logging.info("Beginning training")
for step in itertools.count():
    epsilon = np.interp(step, [0, epsilon_decay_num_steps], [epsilon_start, epsilon_end]) # Epsilon annealing

    if np.random.uniform(0, 1) < epsilon:
        action = sample(list(Action.MOVES.values()), 1)[0]
    else:
        action = online_network.act(batch_to_tensor(game.board))

    # Reward is the sum of the numbers on all merged tiles. Source: https://arxiv.org/pdf/2110.10374.pdf
    old_board = game.board
    old_sum = np.sum(old_board)
    game.move(action)
    new_sum = np.sum(game.board)
    reward = new_sum - old_sum
    done = game.is_done()

    # Experience replay: agent’s experiences at each time-step, e_t = (s_t, a_t, r_t, s_t+1)
    transition = (old_board, action, reward, done, game.board)
    replay_memory.append(transition) # Save this transition

    episode_reward += reward

    if done:
        game.reset() # Reset
        reward_memory.append(episode_reward)
        episode_reward = 0.0

    if step >= 10000:
        logging.info("Time to play")
        game.reset()
        game.show()
        while True:
            action = online_network.act(batch_to_tensor(game.board))
            game.move(action)
            game.show()
            if game.is_done():
                sys.exit(1)
            


    # Perform gradient descent step
    transitions = sample(replay_memory, batch_size) # Sample random minibatch of transitions

    old_boards = batch_to_tensor([t[0] for t in transitions])
    actions = batch_to_tensor([t[1] for t in transitions], True)
    rewards = batch_to_tensor([t[2] for t in transitions])
    dones = batch_to_tensor([t[3] for t in transitions])
    new_boards = batch_to_tensor([t[4] for t in transitions])

    target_vals = target_network(new_boards) # Compute targets
    max_target_vals = target_vals.max(dim=1, keepdim=True)[0]
    # print("target vals shape", target_vals.shape) # 4x4x4
    # print("max target shape", max_target_vals.shape)
    # print("rewards shape", rewards.shape)
    # print("dones shape", dones.shape)
    targets = rewards + gamma * (1 - dones) * max_target_vals # Update formula from paper. (1 - dones) will be 0 if done is true (all 1s)

    online_vals = online_network(old_boards) # Compute Huber loss
    # print("online shape", online_vals.shape) # 1x32
    # print("actions shape", actions.shape) # 1x1
    action_q_values = online_vals.gather(dim=1, index=actions) # state_action_values 1x1
    # print("action_q_values", action_q_values.shape)
    # print("target_vals", target_vals)
    loss = loss_function(action_q_values, target_vals)

    # And finally, gradient descent
    optimizer.zero_grad() # Reset gradients
    loss.backward() # Compute gradients
    optimizer.step() # Apply gradients

    # Update target network
    if step % target_param_update_freq == 0:
        target_network.load_state_dict(online_network.state_dict())
        logging.info(f"Step: {step} | Reward: {np.mean(reward_memory)}")
        
