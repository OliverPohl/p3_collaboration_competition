# Deep reinforcement learning: Collaboration and Competition

### Introduction
The repository contains the third and final project of the Udacity Deep Reinforcement Learning Nanodegree.

For this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is used.

It is solved by the DDPG algorithm enhanced by bootstrap and priorized replay methods. This enables solving the environment in less than 200 episodes. The algorithm shall be explained in the following.



![Trained Agent][image1]

### Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Methods

- Tennis.ipynb
def ddpg:
	Input:
		env - Tennis environment
		agents - 2 Agents which have to be defined before
		n_episodes - 500 - Number of episodes
		Batch_size - 128 - batch size for learning 
		N_Bootstrap - 4 - Depth of bootstrapping
		seed - 0 - seed for noise, weights initializing and sampling mini batches
	Output: 
		scores
		trained agents

- agent.py

	class agent
		Input:
			Learning_Rate - 4 - how many steps until learning
			LR_actor - 10^-3 - learning rate actor network
			LR_critic - 10^-3 - learning rate critic network 
			gamma - 0.99 - decay rate 
			theta, sigma - 0.15, 0.2 - for OU noise
			prio_exponent, prio_beta, prio_epsilon - 0.8, 0.8, 0.001 - for priorized replay
		def step:
			Takes new experiences, stores their priorization and intiates learning.
		def act:
			Uses actor network to extract action given a state. Uses noise.
		def learn:
				First the critic is updated using minibatches of experiences. As in DQN,  Q_target values are calculated using Bellman equation. The mean square losses is defined by comparing the latter with the local values of Q(state, action). Using an optimizer scheme (Adam), the information is backpropagated. 
	For the actor, the target function is just the negative value of the critic - Q(state, action). The mean of these values for all states and actions gets backpropagated into the neural net of the actor. The priorization of experiences, which were used in mini batches, are updated. 
		def soft_update:
			Synchronizes local and target networks.

	class OUNoise
		Simulation of an OU process (linear stochastic differential equation). In ddpg_agent it is added to the actions provided by the actor.

- model.py

	class Actor:
	Defines a small neuronal network in torch with 4 fully connected layers. 
	The input layer has the size of the state space, while the output layer has the size of the action space. Hence, for any given state, such a matrix 	returns an action. 
	After all the layers have dimensions: d_state_space- 256 -128 - d_action_space

	class Critic
	Defines a small neural network in “torch” with 4 fully connected layers. 
	The input layer has the size of the state space, while the output layer has the size 1. The second layer has 400 neurons and the action values. 
	Hence, for any given state, such a matrix returns an action value for a given action and state.
	After all the layers have dimensions: d_state_space - 256 +d_action_space - 128 - 1


