
# Report


## Learning Algorithm - Policy based methods  - MADDPG( Multi Agent RL )


The main point in MARL is that there is also a joint set of actions, from differnt agents. Agents can share their experiences and accelerate learning
Even here the state transitions are markovian just like MDP, next state depends only on the present state and the action taken in the present state. However it now depends on the joint action.


**Approaches :**

* Train each agent individually without considering existence of others. Consider other agents as part of the environment and learn its own policy. Since all are learning simultaneously, the environment as seen from the perspetive of a single agent changes dynamically. (**Non stationarity of the environemnt**) In most single RL algos, environment is assumed to be stationary which leads to certain convergence guarantees.
* **Meta agent** - takes into account the existence of multiple agents. A single policy is learned for all the agents. It takes as input, the present state of the environment and returns the action of each agent in the form of a single join action vector. Typically a single reward function given the environment state and action vector returns a global reward.

The joint action space would increase exponentially with n. If environment is partially observable locally, each agent will have a different observation. So this appraoch works well only when the agent knows everything about the environment.

I followed the second approach to create a meta agent which has a colelction of DDPG agents each with their own set of local and target Actor-Critic networks and i used a Shared Replay buffer to store the collective experiences.

Explore and expolit trade off is handled using the OU noise

DDPG has been explained [here](https://github.com/snknitin/continuous-control/blob/master/report.md) in my previous project


## Algorithm

* Use local actor network to predict action vector for each agent and add OU noise to the actions of all agents, then clip between -1 and +1 
* Add the experience tuples from all agents to the shared replay buffer and sample a batch of tuples for learning
* Once buffer is sufficiently filled beyond the batchsize , update and learn few times every few updates
* Update the critic:
    * Get the next action by feeding next state into target actor 
    * Get the action-value of next state/next action pair by feeding these into target critic - call this Q_next
    * Compute the 'actual' Q value for the current state/action pair as Q_current = r + γ*Q_next
    * Get the predicted Q by feeding the current state and current action into the local critic - call this Q_pred
    * Compute the MSE loss between Q_current and Q_predicted, and update the weights of the local critic
* Update the actor:
    * Get the predicted actions for the current states based on the local actor
    * Get the Q-values (expected reward) of these by passing the current states and predicted actions through the local critic
    * Use the negative mean of the Q-values as a loss to update the weights of the local actor
* Soft-Update the actor and critic target network weights by moving these slightly toward the local weights 
 

## Hyper parameters and Other Changes


I modified the architecture of the Actor and Critic Neural networks several times depending on the strategy applied for state space observation of each DDPG agent along with different number and layer sizes including Batchnorm and Dropout layers. Dropout disn't seem to particuarly help the convergence. I tried tuning several hyperparameters like the learning rates, batchsize, Starting noise, epsilon decay, number of updates etc. I wanted to try prioritized replay but there were few errors in my implementation


* Weights initialized per **Xavier initialization strategy** which helped
* Activation function  -  **Relu** instead of Leaky_relu for all layers except the last one.  Using torch.tanh for the final one in the actor
* I tried **batch_size as 32,64,128,256,512,1024**
* I tried **hidden layers as (32,16)/(256,126)/(400,300) and settled on (128,128) as suggested in my project feedback**
* I tried multiple learning rates between 1e-3 to 5e-4
* Sampled random actions initially for few episodes to help debug


Tuning other hyperparameters might help converge even faster


            
           * BATCH_SIZE = 128        # minibatch size
           * GAMMA = 0.99            # discount factor
           * TAU = 1e-3              # for soft update of target parameters
           * LR_ACTOR = 1e-3         # learning rate of the actor
           * LR_CRITIC = 1e-3        # learning rate of the critic
           * WEIGHT_DECAY = 0        # L2 weight decay
           * UPDATE_EVERY = 1
           * DROPOUT =0.2
           * NUM_UPDATES = 1
           * NOISE_SIGMA = 0.2       # Ornstein-Uhlenbeck noise parameter, volatility
           * NOISE_THETA = 0.15      # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
           * NOISE_START = 3.0       # initial value for epsilon in noise decay process in Agent.act()
           * EPS_EP_END = 200        # episode to end the noise decay process
           * EPS_FINAL = 0           # final value for epsilon after decay
           * PER_ALPHA = 0.6         # importance sampling exponent
           * PER_BETA = 0.4          # prioritization exponent       


## Plot of Rewards

**SUCCESS**

![alt text](https://github.com/snknitin/collaboration-tennis/blob/master/static/Capture.PNG)


![alt text](https://github.com/snknitin/collaboration-tennis/blob/master/static/results.PNG)

**FAILED**


![alt text](https://github.com/snknitin/collaboration-tennis/blob/master/static/Capture_0.PNG)

![alt text](https://github.com/snknitin/collaboration-tennis/blob/master/static/Capture2.PNG)

![alt text](https://github.com/snknitin/collaboration-tennis/blob/master/static/Capture3.PNG)




## Ideas for Future Work

* Try optimizing the hyperparameters for better performance
* I tried the noise decay in multiple attempts and it led to initial bad policy and was stuck there for a really long time, so i used fixed noise for this trial
* Prioritized experience relay seems like a good alternative for the shared buffer but i had some issues with my implementation which required changing the architecture, so i avoided it for now
* I would like to try out A3C and even Proximal Policy Optimization (PPO), as it is known to give good performance with continuous control tasks.I came across Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm as another method for adapting DDPG for continuous control, which seems interesting.
