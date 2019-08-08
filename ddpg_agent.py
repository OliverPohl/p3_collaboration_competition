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

BUFFER_SIZE = int(1e5)  # replay buffer size
TAU = 1e-2   #-3        # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, Batch_size=None, Learning_Rate=None, N_Bootstrap=1, LR_actor=1e-4, LR_critic=1e-3, gamma=0.99, theta=.2, sigma=1., prio_exponent = 0, prio_beta = 0, prio_epsilon = .001 ):
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
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_critic, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, theta, sigma)
        self.Batch_size = Batch_size

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.Batch_size, gamma, random_seed, N_Bootstrap, prio_exponent , prio_beta, prio_epsilon)
        self.t_step = 0
        self.Learning_Rate = Learning_Rate
        self.N_Bootstrap = N_Bootstrap

        self.gamma = gamma
        self.N_train = 6



    def step(self, state, action, reward, next_state, done):
        """Save experience, priorizatoin in replay memory, and use random sample from buffer to learn."""
        self.t_step = (self.t_step+1) % self.Learning_Rate
        next_state = torch.FloatTensor([next_state])
        state = torch.FloatTensor([state])
        action = torch.FloatTensor([action])
        action_next = self.actor_target(next_state)
        # calculate priorization
        Q_target_next = self.critic_target(next_state, action_next)
        Q_target = reward + (self.gamma * Q_target_next.cpu().data.numpy()*(1 - done))
        Q_expected = self.critic_local(state, action).cpu().data.numpy()
        self.memory.add(state, action, reward, next_state, done, np.abs(Q_target - Q_expected)[0][0], BUFFER_SIZE)
        if (self.t_step == 0) and (len(self.memory) > self.Batch_size):
            for _ in range(self.N_train):
                if self.N_Bootstrap < 2:
                    experiences, indices, is_weights = self.memory.sample()
                else:
                    experiences, indices, is_weights = self.memory.sample_bootstrap()
                self.learn(experiences, indices, is_weights)


    def act(self, state, reduction=1., add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise*reduction
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, indices, is_weights):
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
        Q_targets = rewards + (self.gamma * Q_targets_next *(1 - dones))#
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # update prios in prio_memory
        new_deltas = np.abs(Q_targets.cpu().data.numpy() - Q_expected.cpu().data.numpy())
        self.memory.update_prios(indices, new_deltas[:, 0])
        critic_loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_targets)).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
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
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

