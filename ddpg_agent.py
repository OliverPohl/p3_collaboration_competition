import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from prioritized_memory import ReplayBuffer



import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

BUFFER_SIZE = int(1e6)  # replay buffer size
#BATCH_SIZE = 128        # minibatch size
GAMMA = 0.1#99            # discount factor
TAU = 1e-3          # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed,Batch_size=None, Learning_Rate=None, N_Bootstrap=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.Batch_size = Batch_size

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.Batch_size, random_seed)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.Batch_size, GAMMA, random_seed, N_Bootstrap)
        self.t_step = 0
        self.Learning_Rate = Learning_Rate
        self.N_Bootstrap=N_Bootstrap



    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        self.t_step = (self.t_step+1) % self.Learning_Rate
        next_state = torch.FloatTensor([next_state])# torch.from_numpy(np.asarray(next_state)).float().to(device)
        state = torch.FloatTensor([state])#   torch.from_numpy(np.asarray(state)).float().to(device)
        action = torch.FloatTensor([action])   #torch.from_numpy(np.asarray(action)).float().to(device)
        action_next = self.actor_target(next_state)
        Q_target_next = self.critic_target(next_state, action_next)
        Q_target = reward + (GAMMA * Q_target_next.cpu().data.numpy()* (1 - done) )
        Q_expected = self.critic_local(state, action).cpu().data.numpy()
        self.memory.add(state, action, reward, next_state, done, np.abs(Q_target - Q_expected)[0][0], BUFFER_SIZE)
        #if (reward > 0):
        #    print (self.t_step)
        #    print (reward)
        #    print (np.abs(Q_target - Q_expected)[0][0])
        #    print (self.memory.prios[-1])
        #    print ("next")
        if self.t_step == 0:
            if len(self.memory) > self.Batch_size:
                if self.N_Bootstrap < 2:
                    experiences, indices, is_weights = self.memory.sample()
                else:
                    experiences, indices, is_weights = self.memory.sample_bootstrap(self.N_Bootstrap)
                self.learn(experiences, indices, is_weights, GAMMA)

    def act(self, state, reduction=1., add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample(reduction)
            action += noise
        return action# np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, indices, is_weights, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)

        Q_targets = rewards + (gamma * Q_targets_next *(1 - dones))#
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # update prios


        new_deltas = np.abs(Q_targets.cpu().data.numpy() - Q_expected.cpu().data.numpy())
        #if(len(rewards.cpu().data.numpy()[rewards.cpu().data.numpy()>0])):
        #    index = np.where(rewards.cpu().data.numpy()>0)[0][0]
        #    print ("Hohohohohhohhoh")
        #    print(Q_targets.cpu().data.numpy()[index])
        #    print(Q_expected.cpu().data.numpy()[index])
        #    print (new_deltas[index, 0])
        self.memory.update_prios(indices, new_deltas[:, 0])
        #if(len(rewards.cpu().data.numpy()[rewards.cpu().data.numpy()>0])):
        #    print (self.memory.prios[indices[index]])
        #    print ("next")

        critic_loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_targets)).mean()
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.,theta=0.15, sigma=2.3):# theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, reduction):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + reduction*dx
        return dx #self.state

