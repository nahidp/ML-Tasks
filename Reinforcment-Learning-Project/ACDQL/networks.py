import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 


#the implementation of critic DNN
class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims          #the number of neurons on the first hidden layer
        self.fc2_dims = fc2_dims          #the number of neurons on the second hidden layer
        self.n_actions = n_actions        #the number of neurons on the output layers
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    #feed forward the input state in the critic network 
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))     #concatanating the state and action to be forwarded to the first hidden layer
        action_value = self.fc2(action_value)           #forwarding the output of the 1st layer to 2nd hidden layer
        q = self.q(action_value)      #forwarding the output of the 2nd layer to the output layer to calculate the Q-values
        return q

    
#the implementation of the value network to derive Q function
class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

#the implementation of the Actor DNN
class ActorNetwork(keras.Model):
    def __init__(self, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='relu')     #the 1st hidden layer of the actor network
        self.fc2 = Dense(self.fc2_dims, activation='relu')     #the 2nd hidden layer of the actor network
        self.mu = Dense(self.n_actions, activation=None)       #the 1st output of the actor network wich is the mean of the normal distribution
        self.sigma = Dense(self.n_actions, activation=None)    #the 2nd output of the actor network wich is the std of the normal distribution

    #feed forward to the actor network to ontain the mean and std of the normal distribution for an input
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    #choosing an action from the normal distribution with mean and std outputted from the actor network
    def sample_normal(self, state, reparameterize=True):
        
        #feed forward the state to the actor network and obtain mean and std
        mu, sigma = self.call(state)     
        
        #create a normal distribution with mean mu, and standard deviation sigma
        probabilities = tfp.distributions.Normal(mu, sigma)

        #choose a sample from normal distribution
        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        #convert the selected action to between -max_action and +max_action
        action = tf.math.tanh(actions)*self.max_action
        
        #calculate the log probability of the chosen action and add a noise to avoid log(0)
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        

        #compute the sum of elements across dimensions
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action,log_probs

