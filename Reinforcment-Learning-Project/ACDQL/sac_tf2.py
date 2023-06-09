import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, alpha=0.0001, beta=0.003, input_dims=[102], gamma=0.99, n_actions=1, max_size=1000000,
            layer1_size=50, layer2_size=50, batch_size=256):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.critic_loss = 0

        #define an object of the actor network class for deriving the policy function
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', 
                                    max_action=10)
        
        #define two objects of the critic network class for deriving the value function
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters()

    def choose_action(self, observation):
        
        state = tf.convert_to_tensor([observation])
        
        #choose an action between -max_action and +max_action according to the current state
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state):
        #store the gained experience in the ER memory
        self.memory.store_transition(state, action, reward, new_state)

    def update_network_parameters(self):
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(targets[i])

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        print("learning started...")
        state, action, reward, new_state = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        #using tf.GradientTape to record the operations onto a tape to be able to calculate the gradient
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            #use the actor network to predict the best actions and the corresponding log_probs
            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs = tf.squeeze(log_probs,1)
            
            #use the critic networks to evaluate the chosen action by the actor network
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            
            #consider the minimum q-value outputted from the two critic networks
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            #evaluate the effectiveness of the selected actions by the actor network (by comparing the outputted probs and their values)
            value_target = critic_value - log_probs
            
            #compute the mean squared error loss for the value network
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
            
        #calculate the gradient of the recorded critic losses 
        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        
        #apply the gradient to update the weights of the critic network
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            #calculate the minus (gradient ascent) actor loss according to the q-value of the selected action
            actor_loss = -tf.math.reduce_mean(q1_new_policy)
            self.actor_loss += -actor_loss
            
        #calculate the gradient of the recorded actor losses 
        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        
        #apply the gradient to update the weights of the actor network
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))

        

        with tf.GradientTape(persistent=True) as tape:
            #calculate the target Q-value according to the Bellman equation
            q_hat = rewards + self.gamma*value_
            
            #calculate the critic values according to the current critic networks
            q1_old_policy = tf.squeeze(self.critic_1(states, actions), 1)
            q2_old_policy = tf.squeeze(self.critic_2(states, actions), 1)
            
            #calculate the loss between the real Q-values and the target @-values
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
            
            if critic_1_loss.numpy() > critic_2_loss.numpy():
                self.critic_loss += critic_2_loss.numpy()
            else:
                self.critic_loss += critic_1_loss.numpy()
               
        #calculate the gradient of the recorded Q-values losses related to the critic networks
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        #apply gradients to update the weights of the critic networks
        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))
        
        #update the weights of the value network with the weights of the target value network
        self.update_network_parameters()

    def reset_loss(self):
        self.critic_loss = 0

    def get_loss (self):
        return self.critic_loss


    def reset_loss(self):
        self.actor_loss = 0

    def get_loss (self):
        return self.actor_loss
