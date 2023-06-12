# Human-level control through deep reinforcement learning

## Table of contents
+ The Problem
+ The Solution
+ The Architecture
+ The Algorithm Pseudocode

## The Problem

Reinforcement learning is known to be unstable or even to diverge when a non-linear function 
approximator such as a neural network is used to represent the action-value function. we almost
always get chatters around near optimal value functions. Each time we improve the
policy, there's no guarantee once we use a function approximator that the improvement
step is really an improvement. We kind of make progress, and then take a step away from that progress and so forth.

This instability has several causes: the correlations present in the sequence of observations, 
the fact that small updates to Q may significantly change the policy and therefore change
the data distribution, i.e. an update that increases Q(*st*, *at*) often increases Q(*st+1*, *at*)
for all a and hence also increases the target, possibly leading to oscillations or
divergence of the policy. Another cause is the correlations between action-values and the target values.


## The Solution

The novel artificial agent, termed a deep Q-network can learn successful policies directly
from high-dimensional sensory inputs using end-to-end reinforcement learning. The agent
was tested on the challenging domain of Atari 2600 games. Receiving only the pixels and the game
score as inputs, the agent was able to surpass the performance of all previous algorithms
and achieve a level comparable to that of a professional human games tester across a set
of 49 games, using the same algorithm, network architecture and hyperparameters.

DQN is able to combine reinforcement learning with a class of artificial neural network known as deep neural networks. The goal of
a reinforcement learning agent to maximize expected sum of rewards in the future. More formally, a deep convolutional neural network
is used to approximate the optimal action-value function;

<p align="center">
<img src ="https://user-images.githubusercontent.com/19307995/44312836-c06a1b00-a3fe-11e8-8cb7-8f13a2e2bf68.png"/>
</p>

which is the maximum sum of rewards *r* discounted by 𝛾 at each time-step *t*, achievable
by a behaviour policy π = P(a|s), after making an observation (*s*) and taking action (*a*).


DQN addresses these instabilities with a novel variant of Q-learning, which uses two key
ideas. First, a biologically inspired mechanism termed experience replay that randomizes
over the data is used, thereby removing correlations in the observation sequence and smoothing over
changes in the data distribution.

Second, an iterative update that adjusts the action-values (Q) towards target values
that are only periodically updated is used, thereby reducing correlations with the target.
Generating the targets using an older set of parameters adds a delay between the time an
update to Q is made and the time the update affects the targets, making divergence or 
oscillations much more unlikely.

In a nutshell, This approach has several advantages over standard online Q-learning.
First, each step of experience is potentially used in many weight updates, which allows
for greater data efficiency. Second, learning directly from consecutive samples
is inefficient, owing to the strong correlations between the samples: randomizing
the samples breaks these correlations and therefore reduces the variance of the updates.
By using experience replay, the behaviour distribution is averaged over many of its previous
states, smoothing out learning and avoiding oscillations or divergence in the parameters.

This is a comparison of the DQN agent with the best reinforcement learning methods in the 
literature,

<p align="center">
  <img src = "https://user-images.githubusercontent.com/19307995/44564109-a2374e80-a761-11e8-9f17-a91a351b70f2.png"/>
</p>

The performance of the DQN is normalized with respect to a professional human games
tester (that is, 100% level) and random play (that is, 0% level). The normalized performance 
of DQN expressed as a percentage, is calculated as: 100 x (DQN score - random play score)
/(human score - random play score). It can be seen that DQN outperforms competing methods
in almost all the games, and performs at a level that is broadly comparable with or
superior to a professional human games tester (that is, operationalized as a level of
75% or above) in the majority of games. Audio output was disabled for both human players
and agents. The sensory input was equated between human and the agent. Error bars indicate s.d. across the 30 evaluation episodes, starting with
different initial conditions.


## The Architecture

The approximate value function Q(s,a;θ) is parameterized using the deep convolutional
neural network shown;

<p align="center">
<img src = "https://user-images.githubusercontent.com/19307995/44313225-9ddb0080-a404-11e8-895c-769309806581.png"/>
</p>

in which *θi* are the parameters of the Q-network at iteration *i*. To perform experience
replay, the agent's experience *et* = (*St*, *at*, *rt*, *St+1*) at each time-step *t* is stored in a data
set *Dt* = {*e1*,....,*et*}. During learning, Q-learning updates are applied on samples or
mini-batches of experience *(s,a,r,s′)* ~ *U(D)*, drawn uniformly at random from the pool
of stored samples. The Q-learning update at iteration *i* uses the following loss function;

<p align="center">
<img src = "https://user-images.githubusercontent.com/19307995/44313266-a849ca00-a405-11e8-85b7-a2b9ff7f6888.png"/>
</p>

in which 𝛾 is the discount factor determining the agent's horizon, *θi* are the parameters
of the Q-network at iteration *i* and minused *θi* are the network parameters used to compute the
target at iteration *i*. The target network parameters are only updated with the Q-network parameters
*θi* every C steps and are held fixed between individual updates.

## The Algorithm Pseudocode

The tasks considered are those in which an agent interacts with an environment, in this
case the Atari emulator, in a sequence of actions, observations and rewards. At each
time-step t the agent selects an action *at* from a set of legal game actions *A* = *{1,....,K}*.
The action is passed to the emulator and modifies its internal state and the game score.

The emulator's internal state is not observed by the agent; instead the agent observes
an image from the emualator, which is a vector of pixel values representing the current
screen. In addition it recieves a reward *rt* representing the change in game score.

There is a change to the reward structure of the games during training only. As the
scale of scores varies greatly form game to game, all positive rewards were clipped
at 1 and all negative rewards were clipped at -1, leaving 0 rewards unchanged.
Clipping the rewards in this manner limits the scale of the error derivative and makes
it easier to use the same learning rate across mulitple games. At the same time, it could affect the performance of the agent since it cannot differeniate between rewards of different magnitude.

The algorithm is shown here;

<p align="center">
<img src = "https://user-images.githubusercontent.com/19307995/44313359-b7317c00-a407-11e8-988f-d6324a74f726.png"/>
</p>

This algorithm is model-free: it solves the reinforcement learning task directly using
samples from the emulator, without explicitly estimating the reward and transition dynamics.
It is also off-policy: it learns about the greedy policy , while following a behaviour
distribution that ensures adequate exploration of the state space. In practice, the 
behaviour distribution is often selected by an ε-greedy policy that follows the
greedy policy with probability 1 - ε and selects a random action with probability ε.


## Installation and dependencies

The code is written in Python 3. It's recommended if you create a separate environment and install all next packages,
+ OpenAI gym
+ Tensorflow
+ numpy








