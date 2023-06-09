#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2022
@author: Nahid Parvaresh
"""


import random
import math
import numpy as np
from ctypes import *
from sac_tf2 import Agent
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
  
  
# Env is a structure in shared memory in which cpp (environment) part put data to be fetched by python (Agent-ACDQL.py) part
# Env contains state space and statistics
# state space contains the location of the UAV and UEs
class Env(Structure):
  _pack_ = 1
  _fields_ = [
      ("uav_location", location),
      ("ue_location", location * 50),
      ("stat", statistics)
  ]

  

# Act is a structure in shared memory in which Agent-ACDQL.py puts data to be fetched by cpp (environment.cc) part. Act contains the x and y movement to be performed by the UAV-BS
class Act(Structure):
  _pack_ = 1
  _fields_ = [
      ('action_x', c_float),
      ('action_y', c_float)
  ]


MAX_EPISODES = 601
seed_dql = 434 
area = 2000
subArea = 20
uav_height_max = 300

ue_num =50   
enb_num = 1
state_count = (2*(ue_num)) + (2*(enb_num))      #the state space contains the x and y coordinates of the UAV-BS and UEs

lr = 0.01               #learning rate
gamma = 0.95            #discount factor


load_checkpoint = False          #if true the Agent will load the saved model

#define an object of the actor-critic deep Q-leanring class (sac_tf2.py)
agent = Agent(input_dims=[state_count],
          n_actions=1)

mempool_key = 1234                                                 # memory pool key, arbitrary integer large than 1000
mem_size = 4096                                                    # memory pool size in bytes
memblock_key = 2333                                                # memory block key, need to keep the same in the ns-3 script

exp = Experiment(mempool_key, mem_size, 'ACDQL', '../../')         # Set up the ns-3 environment

if load_checkpoint:
  agent.load_model()

ns3Settings = {'seed': 3, 'run': 3}

print("Start Time =", datetime.now().strftime("%H:%M:%S"))

# train the DQL model
try:
  for episode_index in np.arange(MAX_EPISODES):
      agent.reset_loss()
      ns3Settings['seed'] = random.randint(1,5)
      ns3Settings['run'] = random.randint(1,5)
      exp.reset() 
      rl = Ns3AIRL(memblock_key, Env, Act)                       # Link the shared memory block with ns-3 script (environment.cc)
      pro = exp.run(setting=ns3Settings, show_output=True)       # Set and run the ns-3 script (environment.cc)

      
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
          state_all = np.array([data.env.uav_location.x/1000,data.env.uav_location.y/1000], dtype=np.float32)  

          for i in range(ue_num):
              state_ue = np.array([data.env.ue_location[i].x/1000,data.env.ue_location[i].y/1000], dtype=np.float32)
              state_all = np.concatenate([state_all,state_ue])

          state = state_all
          state = np.reshape(state, [1,state_count]) 
          #####################initial state##################################


          #####################initial action##################################
          action = agent.choose_action(state)    #action is selected by ACDQL algorithm and written into shared memory
          data.act.action_x = action      
          data.act.action_y = action      
          #####################initial action##################################


      while not rl.isFinish():
      #while step_count<=200:
          with rl as data:
              if data == None:
                 break


              #####################reward##################################
              reward = data.env.stat.current_reward                    #reading the reward from the shared memory which has been written by environment.cc
              episode_throughput += data.env.stat.current_throughput   #reading the received data rate from the shared memory
              episode_delay += data.env.stat.current_delay             #reading the delay from the shared memory
              episode_pl += data.env.stat.current_pl                   #reading the packet loss from the shared memory
              episode_reward += reward                                
              #####################reward##################################



              #####################new state##################################  
              next_state = np.array([data.env.uav_location.x/1000,data.env.uav_location.y/1000], dtype=np.float32)  
              for i in range(ue_num):
                  state_ue_new = np.array([data.env.ue_location[i].x/1000,data.env.ue_location[i].y/1000], dtype=np.float32)
                  next_state = np.concatenate([next_state,state_ue_new])

              next_state = np.reshape(next_state, [1, state_count])
              #####################new state##################################



              #####################storing experience#########################
              if not load_checkpoint:
                  agent.remember(state, action, reward, next_state) 
              #####################storing experience#########################



              #####################choosing the next action##################################
              action = agent.choose_action(next_state)    
              data.act.action_x = action       
              data.act.action_y = action       
              #print("action code: ",action)
              #####################choosing the next action##################################


              state = next_state          


              if not load_checkpoint:
                  agent.learn()

      #storing the cumulative throughput, delay, packet loss, reward and TD loss of each episode into the saved files
      actor_loss = agent.get_loss()
      actor_loss = actor_loss/200


      actor_loss_to_append = str(episode_index) + ',' + str(actor_loss) + '\n'
      file_episode_loss_actor = open('episode_actor_loss.txt', 'a')
      file_episode_loss_actor.write(actor_loss_to_append)
      file_episode_loss_actor.close()


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



  print("End Time =", datetime.now().strftime("%H:%M:%S"))

except Exception as e:
  print('Something wrong')
  print(e)
finally:
  del exp
