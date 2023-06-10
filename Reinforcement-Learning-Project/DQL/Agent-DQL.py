#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2022
@author: Nahid Parvaresh
"""

import random
import math
import numpy as np
from ctypes import *
import DQLNew
import csv
import random
import time
import pandas as pd
import _pickle as pkl
from datetime import datetime
from py_interface import *


# The environment is shared between ns-3 and python with the same shared memory using the ns3-ai model.


#location structure containing the horizontal location of UAV-BS and UEs
class location(Structure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
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
    

# Env is a structure in shared memory in which cpp (environment) part put data to be fetched by python (Agent-DQL.py) part
# Env contains state space and statistics
# state space contains the location of the UAV and UEs
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ("uav_location", location),
        ("ue_location", location * 50),
        ("stat", statistics)
    ]


#Act is a structure in shared memory in which Agent-DQL.py puts data to be fetched by cpp (environment.cc) part. Act contains the action_code to be performed by the UAV-BS
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action_code', c_int)
    ]


MAX_EPISODES = 601
seed_dql = 434 
area = 2000
subArea = 20
uav_height_min = 50
uav_height_max = 300

action_count = 5     # determining the number of actions to be considered as the number of outputs of the DQN
ue_num =50 
enb_num = 1
state_count = (2*(ue_num)) + (2*(enb_num))      #the state space contains the x and y coordinates of the UAV-BS and UEs


lr = 0.01                    #learning rate
gamma = 0.95                 #discount factor
epsilon_min = 0.1            #the minimum epsilon to be considered in testing a saved model

load_checkpoint = False      #if true the Agent will load the saved model 


#define an object of the deep Q-learning class (DQLNew.py)
agent = DQLNew.DQN (state_size=state_count, action_size=action_count, lr = lr, gamma=gamma, seed=seed_dql)

mempool_key = 1234                                                 # memory pool key, arbitrary integer large than 1000
mem_size = 4096                                                    # memory pool size in bytes
memblock_key = 2333                                                # memory block key, need to keep the same in the ns-3 script

exp = Experiment(mempool_key, mem_size, 'DQL', '../../')           # Set up the ns-3 environment path

if load_checkpoint:                         
    agent.load_model()
    agent.epsilon = agent.epsilon_min


ns3Settings = {'seed': 3, 'run': 3}

print("Start Time =", datetime.now().strftime("%H:%M:%S"))


loss_min = 10000
episode_max = 0


# train the DQL model
try:
    for episode_index in np.arange(MAX_EPISODES):
        ns3Settings['seed'] = random.randint(1,10)
        ns3Settings['run'] = random.randint(1,10)
        exp.reset() 
        rl = Ns3AIRL(memblock_key, Env, Act)                    # Link the shared memory block with ns-3 script (environment.cc)
        pro = exp.run(setting=ns3Settings, show_output=True)    # Set and run the ns-3 script (environment.cc)

        
        #for storing the sum of throughput/delay/pl/reward at each episode
        episode_throughput = 0      
        episode_delay = 0
        episode_pl = 0
        episode_reward = 0

        print("episode_index",episode_index)
 
        
        with rl as data:
            if data == None:
               break

            #####################initial state##################################
            #state space contains the horizontal location of UEs and the UAV-BS
            state_all = np.array([data.env.uav_location.x/1000,data.env.uav_location.y/1000], dtype=np.float32)  #step 2 of my algorithm 

            for i in range(ue_num):
                state_ue = np.array([data.env.ue_location[i].x/1000,data.env.ue_location[i].y/1000], dtype=np.float32)
                state_all = np.concatenate([state_all,state_ue])

            state = state_all
            state = np.reshape(state, [1,state_count]) 
            #####################initial state##################################


            #####################initial action##################################
            action = agent.get_action(state)    #action is selected by ACDQL algorithm and written into shared memory
            data.act.action_code = action      
            #####################initial action##################################

            
        while not rl.isFinish():
            with rl as data:
                if data == None:
                   break


                #####################reward##################################
                reward = data.env.stat.current_reward                    #reading the reward from the shared memory which has been written by environment.cc
                episode_throughput += data.env.stat.current_throughput   #reading the received data rate from the shared memory
                episode_delay += data.env.stat.current_delay.            #reading the delay from the shared memory
                episode_pl += data.env.stat.current_pl                   #reading the packet loss from the shared memory
                episode_reward += reward
                #####################reward##################################
                


                #####################new state##################################  
                next_state = np.array([data.env.uav_location.x/1000,data.env.uav_location.y/1000], dtype=np.float32)  #step 8 of my algorithm (new state is read from the shared memory)
                for i in range(ue_num):
                    state_ue_new = np.array([data.env.ue_location[i].x/1000,data.env.ue_location[i].y/1000], dtype=np.float32)
                    next_state = np.concatenate([next_state,state_ue_new])
                
                next_state = np.reshape(next_state, [1, state_count])
                #####################new state##################################


      
                #####################storing experience#########################
                if not load_checkpoint:
                    agent.add_to_replay_memory(state, action, reward, next_state) 
                #####################storing experience#########################



                #####################choosing the next action##################################
                action = agent.get_action(next_state)    #step 3 of my algorithm (action is selected by NN and written into shared memory)
                data.act.action_code = action       # step 3 done
                #print("action code: ",action)
                #####################choosing the next action##################################
             
    
                state = next_state           #step 11 of my algorithm

                
                if not load_checkpoint:
                    agent.update_counter()
                    agent.learn_and_update_weights_by_reply()


        if not load_checkpoint:   
            if agent.counter>1000 and episode_throughput>episode_max:
                episode_max = episode_throughput
                agent.save_model()

                
                
        #storing the cumulative throughput, delay, packet loss, reward and TD loss of each episode into the saved files
        loss = agent.get_loss()
        loss_to_append = str(episode_index) + ',' + str(loss) + '\n'
        file_episode_loss = open('episode_loss.txt', 'a')
        file_episode_loss.write(loss_to_append)
        file_episode_loss.close()


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

        #decreasing the epsilon in the epsilon-greedy strategy so that in next iteration, we exploit more and explore less
        agent.update_epsilon()

    print("End Time =", datetime.now().strftime("%H:%M:%S"))

except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp


