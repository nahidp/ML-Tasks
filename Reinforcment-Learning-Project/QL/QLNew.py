import random
import numpy as np

# The Q learning class to chose the action at each time step and update the q table
class QLearningAgent:
    def __init__(self, state_size, action_size, lr, gamma, seed):

        self.counter = 0

        self.lr = lr
        self.gamma = gamma

        self.batch_size = 128
        self.num_action_space = action_size
        self.epsilon =  1  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.seed = seed

        self.state = np.zeros(state_size, dtype=int)
        self.action = np.zeros(action_size, dtype=int)

        # Add a few lines to caputre the seed for reproducibility.
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.memory = []
        # Create a clean Q-Table.
        self.q = np.zeros(shape=(state_size, action_size))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(self.epsilon)


    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_action_space)
        return np.argmax(self.q[obs])


    def update_q_table(self, state, action, reward, next_state):
        
        self.q[state, action] += self.lr * (reward + self.gamma * max(self.q[next_state, :]) - self.q[state, action])

    def save_qtable(self):
        np.save(f"qtables/qtable.npy", self.q)

    def load_qtable(self):
        self.q = np.load(f"qtables/qtable.npy")
