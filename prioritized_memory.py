import random
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, gamma, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.prio_epsilon =  .1
        self.prio_exponent = .6
        self.prio_beta = .6
        self.action_size = action_size
        self.memory = []  
        self.prios = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma_pot = [gamma**i for i in range(N_Bootstrap)]
    
    def add(self, state, action, reward, next_state, done, delta):
        """Add a new experience to memory."""
        prio = delta + self.prio_epsilon
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.prios.append(prio)
        if len(self.memory)>=buffer_size:
            self.memory.pop(0)
            self.prios.pop(0)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = (np.asarray(self.prios)**prio_exponent)/sum(np.asarray(self.prios)**prio_exponent) 
        indices = [np.random.choice(np.arange(self.batch_size), p=probs) for _ in range(self.batch_size)]
        less_probs = np.asarray(probs)[indices].tolist()
        is_weights = np.power(self.batch_size * less_probs, -self.prio_beta)
        is_weights /= is_weights.max()
        experiences = [self.memory[idx] for idx in indices]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return [states, actions, rewards, next_states, dones], indices, is_weights

    def update_prios(indices, new_deltas):
        new = np.array(self.prios)+self.prio_epsilon
        new[indices] = new_prios
        self.prios = new.tolist()
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




    ## TODO!

    ## bootstrap extra stuff
    def sample_sequences(self, length_seq , k_seq, memory):
      
        length_mem = len(memory)
        liste = [ list(itertools.islice(memory, k, k+length_seq)) for k in random.sample(range(length_mem), k_seq) 
                                                                                        if (k+length_seq<length_mem)]
        return liste


    def adapt_sample_for_n_bootstrap(self, sample, gamma=0.99):
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if any([element.done for element in sample]):
            return None   # TODO: to be adapted
        else:
            state = sample[0].state
            action = sample[0].action
            done = sample[0].done
            next_state = sample[0].next_state #  sample[-1].next_state
            reward_summed = sum([exp.reward*self.gamma_pot[j] for j,exp in enumerate(sample)])
            e = experience(state, action,  reward_summed,next_state, done)
            return e

    def sample_bootstrap(self,length_seq , k_seq, memory, gamma=0.99):
        liste=self.sample_sequences(length_seq , k_seq, memory)
        experiences = [self.adapt_sample_for_n_bootstrap(element, gamma) for element in liste]
        #print (np.shape(new_samples))
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return [states, actions, rewards, next_states, dones], indices, probs







