
# Report


## Learning Algorithm - Policy based methods  - MADDPG( Multi Agent RL )


The main point in MARL is that there is also a joint set of actions, from differnt agents. Agents can share their experiences and accelerate learning
Even here the state transitions are markovian just like MDP, next state depends only on the present state and the action taken in the present state. However it now depends on the joint action.


**Approaches :**

* Train each agent individually without considering existence of others. Consider other agents as part of the environment and learn its own policy. Since all are learning simultaneously, the environment as seen from the perspetive of a single agent changes dynamically. (**Non stationarity of the environemnt**) In most single RL algos, environment is assumed to be stationary which leads to certain convergence guarantees.
* **Meta agent** - takes into account the existence of multiple agents. A single policy is learned for all the agents. It takes as input, the present state of the environment and returns the action of each agent in the form of a single join action vector. Typically a single reward function given the environment state and action vector returns a global reward.

The joint action space would increase exponentially with n. If environment is partially observable locally, each agent will have a different observation. So this appraoch works well only when the agent knows everything about the environment.

I followed the second apprach to create a meta agent which has a colelction of DDPG agents each with their own set of local and target Actor-Critic networks and i used a Shared Replay buffer to store the collective experiences.




## Hyper parameters and Other Changes


I modified the architecture of the Actor and Critic Neural networks. I tried tuning several hyperparameters like the learning rates, batchsize and the weight decay


* Activation function  - Leaky Relu for all layers except the last one.  Using torch.tanh for the final one
* I had the **batch_size as 256**


Tuning other hyperparameters might help converge even faster


            * BUFFER_SIZE = int(1e6)  # replay buffer size
            * GAMMA = 0.99            # discount factor
            * TAU = 1e-3              # for soft update of target parameters
            * LR_ACTOR = 1e-4         # learning rate of the actor
            * LR_CRITIC = 1e-3        # learning rate of the critic
            * WEIGHT_DECAY = 0        # L2 weight decay
            * HIDDEN_LAYERS=(512,256) # Modified the architecture to include an additional hidden layers with 512 and 256 units
            * UPDATE_EVERY = 5       # 20 for 20 agents case and 4 for single
            * DROPOUT = 0.2
            * num_update = 1
            * NOISE_START = 1.0       # epsilon decay for the noise process added to the actions
            * NOISE_DECAY = 1e-6      # decay for for subrtaction of noise
            * NOISE_SIGMA = 0.2       # sigma for Ornstein-Uhlenbeck noise


## Plot of Rewards





![alt text](https://github.com/snknitin/continuous-control/blob/master/curve-single.PNG)


![alt text](https://github.com/snknitin/continuous-control/blob/master/curve-twenty.PNG)

## Ideas for Future Work

* Try optimizing further using batchnorm and dropout, and use those weights as the starting point for each of the 20 agents
* I would like to try out Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG) since literature suggests it should achieve better performance.
* Even Proximal Policy Optimization (PPO), as it is known to give good performance with continuous control tasks.

From the note mentioned in the project page, i came across Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm as another method for adapting DDPG for continuous control, which seems interesting.