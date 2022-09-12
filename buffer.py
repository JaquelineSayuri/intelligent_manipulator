import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.episode_transitions = []

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
        
    def save_buffer(self, seed):
        with open(f"buffer_backup/state_buffer_backup_{seed}.txt", "wb") as file:
                pickle.dump(self.state_memory, file)
        with open(f"buffer_backup/new_state_buffer_backup_{seed}.txt", "wb") as file:
                pickle.dump(self.new_state_memory, file)
        with open(f"buffer_backup/action_buffer_backup_{seed}.txt", "wb") as file:
                pickle.dump(self.action_memory, file)
        with open(f"buffer_backup/reward_buffer_backup_{seed}.txt", "wb") as file:
                pickle.dump(self.reward_memory, file)
        with open(f"buffer_backup/terminal_buffer_backup_{seed}.txt", "wb") as file:
                pickle.dump(self.terminal_memory, file)

    def load_buffer(self, seed):
        with open(f"buffer_backup/state_buffer_backup_{seed}.txt", "r") as file:
                self.state_memory = pickle.load(file)
        with open(f"buffer_backup/new_state_buffer_backup_{seed}.txt", "r") as file:
                self.new_state_memory = pickle.load(file)
        with open(f"buffer_backup/action_buffer_backup_{seed}.txt", "r") as file:
                self.action_memory = pickle.load(file)
        with open(f"buffer_backup/reward_buffer_backup_{seed}.txt", "r") as file:
                self.reward_memory = pickle.load(file)
        with open(f"buffer_backup/terminal_buffer_backup_{seed}.txt", "r") as file:
                self.terminal_memory = pickle.load(file)

