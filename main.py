# -*- coding: utf-8 -*-
import random
import numpy as np
from environment import SlopeGame
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
import time

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        NVIDIA model used
        Image normalization to avoid saturation and make gradients work better.
        Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Drop out (0.5)
        Fully connected: neurons: 100, activation: ELU
        Fully connected: neurons: 50, activation: ELU
        Fully connected: neurons: 10, activation: ELU
        Fully connected: neurons: 1 (output)
        # the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        dropout avoids overfitting
        ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation='elu', input_shape=(self.state_size[0], self.state_size[1], self.state_size[2])))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    time.sleep(1)
    env = SlopeGame()
    state_size = env.observation_space
    action_size = env.action_space[1]
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1,] + state_size)
        time = 0
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1,] + state_size)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            time += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")