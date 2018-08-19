
## The Problem

Reinforcement learning is known to be unstable or even to diverge when a non-linear function 
approximator such as a neural network is used to represent the action-value function. we almost
always get chatters around near optimal value functions. Each time we improve the
policy, there's no guarantee once we use a function approximator that the improvement
step is really an improvement. We kind of make progress, and then take a step away from that progress and so forth.

This instability has several causes: the correlations present in the sequence of observations, 
the fact that small updates to Q may significantly change the policy and therefore change
the data distribution, and the correlations between action-values and the target values.

## The Solution

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

## The Algortihm Pseudocode

The tasks consider are those in which an agent interacts with an environment, in this
case the Atari emulator, in a sequence of actions, observations and rewards. At each
time-step t the agent selects an action *at* from a set of legal game actions *A* = *{1,....,K}*.
The action is passed to the emulator and modifies its internal state and the game score.

The emulator's internal state is not observed by the agent; instead the agent observes
an image from the emualator, which is a vector of pixel values representing the current
screen. In addition it recieves a reward *rt* representing the change in game score. The algorithm is shown here;

<p align="center">
<img src = "https://user-images.githubusercontent.com/19307995/44313359-b7317c00-a407-11e8-988f-d6324a74f726.png"/>
</p>

## Installation and dependencies

The code is written in Python 3. It's recommended if you create a separate environment and install all next packages,
+ OpenAI gym
+ Tensorflow
+ numpy








