import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 

class CriticNetwork(keras.Model):
    def __init__(self, name='critic', chkpt_dir='model'):
        super(CriticNetwork, self).__init__()
        self.fc_dims = [10, 256, 256, 1]
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc_dims[0], activation='relu')
        self.fc2 = Dense(self.fc_dims[1], activation='relu')
        self.fc3 = Dense(self.fc_dims[2], activation='relu')
        self.q = Dense(self.fc_dims[3], activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)
        q = self.q(action_value)

        return q

class ValueNetwork(keras.Model):
    def __init__(self, name='value', chkpt_dir='model'):
        super(ValueNetwork, self).__init__()
        self.fc_dims = [6, 256, 256, 1]
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc_dims[0], activation='relu')
        self.fc2 = Dense(self.fc_dims[1], activation='relu')
        self.fc3 = Dense(self.fc_dims[2], activation='relu')
        self.v = Dense(self.fc_dims[3], activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)
        state_value = self.fc3(state_value)
        
        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):
    def __init__(self, max_action, n_actions=2, name='actor', chkpt_dir='model'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc_dims = [6, 256, 256, self.n_actions]
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-2
        self.step = 0
        self.min_action_tensor = tf.convert_to_tensor(
            [[-self.max_action for i in range(self.n_actions)] for j in range(512)],
            dtype=float
        )
        self.max_action_tensor = tf.convert_to_tensor(
            [[self.max_action for i in range(self.n_actions)] for j in range(512)],
            dtype=float
        )

        self.fc1 = Dense(self.fc_dims[0], activation='relu')
        self.fc2 = Dense(self.fc_dims[1], activation='relu')
        self.fc3 = Dense(self.fc_dims[2], activation='relu')
        self.mu = Dense(self.fc_dims[3], activation=None)
        self.sigma = Dense(self.fc_dims[3], activation=None)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, start_steps):
        #'''
        if self.step < start_steps:
            if state.shape[0] == 1:
                probabilities = tfp.distributions.Uniform([self.min_action_tensor[0]], [self.max_action_tensor[0]])
            else:
                probabilities = tfp.distributions.Uniform(self.min_action_tensor, self.max_action_tensor)
            actions = probabilities.sample()
            action = actions
        else:
            mu, sigma = self.call(state)
            probabilities = tfp.distributions.Normal(mu, sigma)
            actions = probabilities.sample()
            action = tf.math.tanh(actions)*self.max_action
        self.step += 1
        '''
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()
        action = tf.math.tanh(actions)*self.max_action
        self.step += 1
        '''
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs