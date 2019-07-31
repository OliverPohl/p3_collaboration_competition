import random
import numpy as np
from collections import namedtuple
import torch
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, gamma, seed, N_Bootstrap):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.prio_epsilon =  .001
        self.prio_exponent = .6
        self.prio_beta = .6
        self.action_size = action_size
        self.memory = []  
        self.prios = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma_pot = [gamma**i for i in range(N_Bootstrap)]

    def add(self, state, action, reward, next_state, done, delta, buffer_size):
        """Add a new experience to memory."""
        prio = delta + self.prio_epsilon
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.prios.append(prio)
        if len(self.memory) >= buffer_size:
            self.memory.pop(0)
            self.prios.pop(0)
    def create_indices_and_weights_for_sample(self):
        probs = (np.asarray(self.prios)**self.prio_exponent)/sum(np.asarray(self.prios)**self.prio_exponent)  # TODO: Maybe do before outside?
        indices = [np.random.choice(np.arange(len(self.memory)), p=probs) for _ in range(self.batch_size)]
        less_probs = np.asarray(probs)[indices].tolist()
        is_weights = np.power(self.batch_size * less_probs, -self.prio_beta)
        is_weights /= is_weights.max()
        return(indices, is_weights)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices, is_weights = self.create_indices_and_weights_for_sample()
        experiences = [self.memory[idx] for idx in indices]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return [states, actions, rewards, next_states, dones], indices, is_weights

    def update_prios(self, indices, new_deltas):
        new = np.array(self.prios)
        new[indices] = new_deltas +self.prio_epsilon
        self.prios = new.tolist()
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


    ## bootstrap sampling, i.e. using more reward information along trajectory
    def sample_sequences(self, length_seq, indices):
        length_mem = len(self.memory)
        new_indices = [k for k in indices if (k+length_seq < length_mem)]
        liste = [list(itertools.islice(self.memory, int(k), int(k+length_seq), 2)) for k in new_indices ]
        return liste, new_indices


    def adapt_sample_for_n_bootstrap(self, sample):
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #if any([element.done for element in sample]):
        #    return None
        #else:
        state = sample[0].state
        action = sample[0].action
        done = sample[0].done
        next_state = sample[0].next_state #  sample[-1].next_state
        #reward_summed = sum([exp.reward*self.gamma_pot[j] for j,exp in enumerate(sample)])
        reward_summed = 0
        for j, exp in enumerate(sample):
            reward_summed += exp.reward * self.gamma_pot[j]
            if(exp.done):
                break
        reward_summed = reward_summed/(j+1.)
        e = experience(state, action, reward_summed, next_state, done)
        return e

    def sample_bootstrap(self, length_seq):
        indices, is_weights = self.create_indices_and_weights_for_sample()
        liste, new_indices = self.sample_sequences(length_seq, indices)
        experiences = [self.adapt_sample_for_n_bootstrap(element) for element in liste]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return [states, actions, rewards, next_states, dones], new_indices, is_weights


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
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "prio"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, prio):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, prio)
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
        prios = torch.from_numpy(np.vstack([e.prio for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones, prios)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

