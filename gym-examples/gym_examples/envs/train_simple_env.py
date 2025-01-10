'''
This is a recreation of CADRL env, in this recreation I will not be using any global CONFIG file,
rather will pass arguments to the env, and initialize the env, this will improve readability, 
and easy to make different kind of envs for training and testing(evaluating)

This env will have different kinds of non-learning agents each with their unique control policies(by unique I mean only two kind - for now).
Some properties of this environment, 

1) area of environment scales with number of agents - to maintain some sense of constant distance between agents at beginning 
2) will use sparse reward based on MIT-ACL 



Non_learning agent - 
1) 


Learning agent - 
1)




env desc - 
1) start-end location of agents 
2) non-learning agent control policy 
3) render 
4) GYM env super-class 
5) GYM API 

'''

# Required imports
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import matplotlib.pyplot as plt

# Register your custom environment
#from gymnasium.envs.registration import register #this line () is not required - not a local package 'remove line when confirmed'
from simple_env import SimpleEnv

import sys
import os
from datetime import datetime
import logging



env = SimpleEnv(max_uavs=12, max_vertiports=14, seed = 42)
env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    if done:
        break
