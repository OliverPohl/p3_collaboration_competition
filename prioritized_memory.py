import random
import numpy as np
from collections import namedtuple
import torch
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Store experience tuples and priorization and giving samples using bootstrap+priorization"""

    def __init__(self, action_size, buffer_size, batch_size, gamma, seed, N_Bootstrap, prio_exponent = 0, prio_beta = 0, prio_epsilon = .001):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            prio_epsilon: constant added to priorization factors
            prio_beta, prio_exponent :  weight exponent for priorization
            N_Bootstrap: bootstrap depth
        """
        self.prio_epsilon =  prio_epsilon
        self.prio_exponent =prio_exponent
        self.prio_beta =prio_beta
        self.action_size = action_size
        self.memory = []  
        self.prios = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.N_Bootstrap = N_Bootstrap
        self.gamma_pot = [gamma**i for i in range(N_Bootstrap)]

    def add(self, state, action, reward, next_state, done, delta, buffer_size):
        """Add a new experience and priorization to memory."""
        prio = delta + self.prio_epsilon
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.prios.append(prio)
        if len(self.memory) >= buffer_size:
            self.memory.pop(0)
            self.prios.pop(0)

    def update_prios(self, indices, new_deltas):
        new = np.array(self.prios)
        new[indices] = new_deltas + self.prio_epsilon
        self.prios = new.tolist()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def create_indices_and_weights_for_sample(self):
        """Create random indices for sampling, calculate weights out of prios"""
        probs = (np.asarray(self.prios)**self.prio_exponent)/sum(np.asarray(self.prios)**self.prio_exponent)  # TODO: Maybe do before outside?
        indices = [np.random.choice(np.arange(len(self.memory)), p=probs) for _ in range(self.batch_size)]
        less_probs = np.asarray(probs)[indices].tolist()
        is_weights = np.power(self.batch_size * less_probs, -self.prio_beta)
        is_weights /= is_weights.max()
        return(indices, is_weights)


    def sample(self):
        """Randomly sample a batch of experiences from memory using priorization."""
        indices, is_weights = self.create_indices_and_weights_for_sample()
        new_indices = [k for k in indices if (k+self.N_Bootstrap < len(self.memory))]
        experiences = [self.memory[k] for k in new_indices]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return [states, actions, rewards, next_states, dones], new_indices, is_weights



    def sample_bootstrap(self):
        """Randomly sample a batch of experiences from memory using priorization and bootstrapping."""
        indices, is_weights = self.create_indices_and_weights_for_sample()
        liste, new_indices = self.sample_sequences(indices)
        experiences = [self.adapt_sample_for_n_bootstrap(element) for element in liste]  #length_seq*0.5
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return [states, actions, rewards, next_states, dones], new_indices, is_weights



    def sample_sequences(self, indices):
        """create experience sample sequences of lengths N_Bootstrap"""
        length_mem = len(self.memory)
        new_indices = [k for k in indices if (k+self.N_Bootstrap < length_mem)]
        liste = [list(itertools.islice(self.memory, int(k), int(k+self.N_Bootstrap))) for k in new_indices ] # ), 2)) for
        return liste, new_indices


    def adapt_sample_for_n_bootstrap(self, sample, reward_scale=1.):
        """transforms sample experience sequences in sample experiences"""
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        state = sample[0].state
        action = sample[0].action
        done = sample[0].done
        next_state = sample[0].next_state #  sample[-1].next_state
        reward_summed = 0
        for j, exp in enumerate(sample):
            if exp.done:
                reward_summed += exp.reward * self.gamma_pot[j]
                break
            else:
                reward_summed += exp.reward * self.gamma_pot[j]

        reward_summed = reward_scale*reward_summed / (j+1.)   #[0.5**i for i in range(N_Bootstrap)]/sum([0.5**i for i in range(N_Bootstrap)])
        e = experience(state, action, reward_summed, next_state, done)
        return e



class StandardReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

