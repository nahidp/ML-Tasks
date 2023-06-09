#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2022
@author: nahid parvaresh
"""


import random
import numpy as np
from ctypes import *      #to use c++ structures
import QLNew
import csv
import random
import time
import pandas as pd
import _pickle as pkl
from datetime import datetime
from py_interface import *


#location structure containing the horizontal location of UAV-BS and UEs
class location(Structure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float)
    ]

#statistic structure containing all the stat parameters and reward that all are put into the shared memory by environment.cc and read by Agent-QL.py 
class statistics(Structure):
    _pack_ = 1
    _fields_ = [
        ("current_reward", c_float),
        ("current_throughput", c_float),
        ("current_delay", c_float),
        ("current_pl", c_float),
    ]    
    
# Env is a structure in shared memory in which cpp (environment) part put data to be fetched by python (Agent-QL.py) part
# Env contains state space and statistics
# state space contains the indexes of UAVs (array [2])
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ("uav_index", c_int),
        ("stat", statistics)
    ]


# Act is a structure in shared memory in which Agent-QL.py put data to be fetched by cpp (environment.cc) part
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action_code', c_int)
    ]


MAX_EPISODES = 601
seed_ql = 434
area = 1000
subArea = 50

action_count = 5     # determining the number of action to be considered as the number of columns in q-table
ue_num = 50
enb_num = 1
#state_count = (2*(ue_num)) + (2*(enb_num))      # x and y coordinate of UAV-BS
state_count = int((area/subArea)*(area/subArea));     # x and y coordinate of UAV-BS


lr = 0.01
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.9
epsilon_min = 0.1

#define an object of the Q-learning class to manage the Q-table and select the best actions
agent = QLNew.QLearningAgent (state_size=state_count, action_size=action_count, lr = lr, gamma=gamma, seed=seed_ql)

mempool_key = 1234                                              # memory pool key, arbitrary integer large than 1000
mem_size = 4096                                                 # memory pool size in bytes
memblock_key = 2333                                             # memory block key, need to keep the same in the ns-3 script

exp = Experiment(mempool_key, mem_size, 'DescreteQL', '../../')     # Set up the ns-3 environment

seed = 3
run = seed
ns3Settings = {'seed': seed, 'run': run}

print("Start Time =", datetime.now().strftime("%H:%M:%S"))

# train the DQL model
try:
    for episode_index in np.arange(MAX_EPISODES):
        
        episode_throughput = 0      # for plotting the sum of throughput at each episode
        episode_reward = 0
        episode_delay = 0
        episode_pl = 0
        exp.reset() 
        
        rl = Ns3AIRL(memblock_key, Env, Act)                    # Link the shared memory block with ns-3 script
        
        pro = exp.run(setting=ns3Settings, show_output=True)       # Set and run the ns-3 script (environment.cc)
        print("episode_index",episode_index)
 
        with rl as data:
            if data == None:
               break


            #read the first location of the UAV-BS from the shared memory 
            state = data.env.uav_index  
            
            #select the first action and out it into the shaared memory
            action = agent.get_action(state) 
            data.act.action_code = action   
            
        while not rl.isFinish():
            with rl as data:
                if data == None:
                   break
                
                #read the next state fro mthe shared memory
                next_state = data.env.uav_index 


                #read the assigned reward from the shared memory
                reward = data.env.stat.current_reward   #step 9 of my algorithm (reward is read from the shared memory)

                #read the throughput (received data), delay, packet loss from the shared memory and add them to the total stats of the episode 
                episode_throughput += data.env.stat.current_throughput
                episode_delay += data.env.stat.current_delay
                episode_pl += data.env.stat.current_pl
                episode_reward += reward

                
                #select the next best action and put it into the shared memory. to be fetched by environment.cc
                action = agent.get_action(state)          
                data.act.action_code = action 
                
                #update the Q-table after a new experience is gained
                agent.update_q_table(state, action, reward, next_state)

                state = next_state           


        #storing the cumulative throughput, delay, packet loss and reward into the saved files
        throughput_to_append = str(episode_index) + ',' + str(episode_throughput) + '\n'
        file_episode_throughput = open('episode_throughput.txt', 'a')
        file_episode_throughput.write(throughput_to_append)
        file_episode_throughput.close()

        delay_to_append = str(episode_index) + ',' + str(episode_delay/200) + '\n'
        file_episode_delay = open('episode_delay.txt', 'a')
        file_episode_delay.write(delay_to_append)
        file_episode_delay.close()

        pl_to_append = str(episode_index) + ',' + str(episode_pl/200) + '\n'
        file_episode_pl = open('episode_pl.txt', 'a')
        file_episode_pl.write(pl_to_append)
        file_episode_pl.close()        
        
        reward_to_append = str(episode_index) + ',' + str(episode_reward) + '\n'
        file_episode_reward = open('episode_reward.txt', 'a')
        file_episode_reward.write(reward_to_append)
        file_episode_reward.close()
        ######################################################################

        #update the epsilon for the next episode
        agent.update_epsilon()

    print("End Time =", datetime.now().strftime("%H:%M:%S"))
    
    #save the model for the future use
    agent.save_qtable()
    
except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp
