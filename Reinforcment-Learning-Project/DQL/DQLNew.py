import numpy as np
from collections import deque
import random
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_squared_error


# The DQL learning class to chose the best action at each time step and update the DQN
class DQN:    

    def __init__(self, state_size, action_size, lr, gamma, seed, chkpt_dir='models/'):

        self.counter = 0              #the counter for contoroling the rate of DQN update
        self.step_size = 5            #the step size for updating the DQN parameters
        self.loss = 0                 #TD loss
        self.lr = lr                  #learning rate
        self.gamma = gamma            #discount factor
   
        self.replay_memory_buffer = deque(maxlen=100000)         #the ER memory with the size of 100000 (this size is a hyper parameter)
        self.batch_size = 128                                    #the mini batch size for the training of DQN

        self.num_action_space = action_size                      #the size of the DQN's output layer
        self.num_observation_space = state_size                  #the size of the DQN's input layer
        self.epsilon = 1                                         #the epsilon for epsilon-greedy strategy
        self.epsilon_min = 0.01                                  #the minimum epsilon so that we always have a minimum level of exploration
        self.epsilon_decay = 0.99                                #the decaying parameter for decreasing the epsilon at each espisode
        self.seed = seed
        self.saved_file = os.path.join(chkpt_dir, '_DQL')
        

        # Add a few lines to caputre the seed for reproducibility.
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.model = self.initialize_model()
        
    
    def initialize_model(self):
      
        #create a linear stack of layers for the DQN
        model = Sequential()
        
        #add the input layer, hidden layers and output layers to the sequential DQN model
        model.add(Dense(50, input_dim=self.num_observation_space, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_action_space, activation='linear'))
        
        #define the optimizer for the DQN
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.lr))
        
        return model

    #choose the best action according to epsilon-greedy strategy.
    #Exploration with the probability of epsilon
    #Exploitation with the probability of 1-epsilon
    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_action_space)
            return action

        action = np.argmax(self.model.predict(obs)[0])
        return action
        
    #store a gained experience into the ER memory
    def add_to_replay_memory(self, state, action, reward, next_state):
        self.replay_memory_buffer.append((state, action, reward, next_state))

    #learn and update the DQN after each time step
    def learn_and_update_weights_by_reply(self):
      
        #training of the DQN should be done after experiencing 1000 movement. Also, trainign is done every step size and after ER reaches the size of the batch size
        if len(self.replay_memory_buffer) < self.batch_size or self.counter<1000 or self.counter % self.step_size != 0: return

        #choose a random batch from ER memory for the training od DQN
        random_sample = self.get_random_sample_from_replay_mem()
        
        #seperate states, actions, rewards and next_states from the randomly-selected batch
        states, actions, rewards, next_states = self.get_attribues_from_sample(random_sample)

        #calculate the targets (labels) that contains the actual rewards
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states)[0]))
        
        #calculate the output of the DQN to be considered as the outputted Q-values 
        target_vec = self.model.predict_on_batch(states)
        
        #use the targets (labels) to adjust the weights of the DQN 
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        history = self.model.fit(states, target_vec, epochs=1, verbose=0)
        
        #store the resulted loss 
        self.loss = history.history['loss'][0]

    
    #update the epsilon for the next round (this function is called at the end of each episode in the Agent-DQL.py)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #get the current loss of the DQN
    def get_loss (self):
        return self.loss

    #seperate the states, actions, rewards, next_states from a batch file
    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states

    #randomely sample batch size records from ER memory
    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    #increase the counter after storing each record of experience
    def update_counter(self):
        self.counter += 1

    #save the DQL model to be used later for test or other purposes
    def save_model(self):
        shutil.rmtree(self.saved_file)
        print("old model removed!")
        os.mkdir(self.saved_file)
        print("saved directory created again")
        self.model.save(self.saved_file)
        print('... saving models ...')


    #load a saved model
    def load_model(self):
        self.model = keras.models.load_model(self.saved_file)
        print('... loading models ...')

