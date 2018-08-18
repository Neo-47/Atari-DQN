import gym
from gym.wrappers import Monitor
import itertools
import numpy as np 
import os
import random
import sys
import tensorflow as tf 
import plotting
from collections import deque, namedtuple

if("../" not in sys.path):
	sys.path.append("../")

VALID_ACTIONS = [0, 1, 2, 3]

env = gym.envs.make("Breakout-v0")