#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 07:35:58 2021

"""
import os

import matplotlib.pyplot as plt
from snake_game import SnakeGame
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import datetime
import imageio
from tqdm import tqdm


def plot_board(name, board):
    plt.figure(figsize=(10, 10))
    plt.imshow(board)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def plot_rewards(path, name, training, evaluation):
    plt.figure(figsize=(10, 10))
    xticks = range(0, len(training), 1)

    plt.plot(xticks, training, linestyle='--', label='Training', color='cornflowerblue')
    plt.plot(xticks, evaluation, linestyle='-', label='Evaluation', color='orange')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Reward")

    plt.tight_layout()
    plt.savefig(path + "/" + name)


def CNNModel(inputs):
    layer = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inputs)
    layer = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    layer = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(layer)
    layer = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(layer)
    layer = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same')(layer)
    layer = Flatten(name='features')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    return layer


def snakeModel(state_shape, action_shape, optimizer, loss, cnn=False):
    init = HeUniform()
    inputs = Input(shape=state_shape, name='inputs')
    layer = inputs
    if cnn:
        layer = CNNModel(inputs)
    layer = Dense(64, input_shape=state_shape, activation='relu', kernel_initializer=init)(layer)
    layer = Dense(32, activation='relu', kernel_initializer=init)(layer)
    outputs = Dense(action_shape, activation='linear', kernel_initializer=init)(layer)
    model = Model(input=inputs, output=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def decide(agent, state, _epsilon):
    if np.random.rand() <= _epsilon:
        return np.random.sample(ACTIONS.keys(), 1)
    predicted = agent.predict(state.reshape([1, state.shape[0]])).flatten()
    return np.argmax(predicted)


def train(replay_memory, model, target_model):
    mini_batch = random.sample(replay_memory, BATCH_SIZE)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    x = []
    y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            # TODO: experiment with different heuristics
            max_future_q = reward + DISCOUNT * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q
        x.append(observation)
        y.append(current_qs)
    model.fit(np.array(x), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=True)


RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PATH = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

BORDER = 1
WIDTH = 30
LENGTH = 30
ACTIONS = {'LEFT': -1, 'AHEAD': 0, 'RIGHT': 1}

BATCH_SIZE = 64 * 2
MEMORY_LENGTH = 50000
EPOCHS = 300
TRAIN_STEPS = 4

MIN_EPOCHS = 1000
MAX_EPSILON = 1
MIN_EPSILON = 0.01

DECAY_RATE = 0.01
DISCOUNT = 0.618

LEARNING_RATE = 0.001


def main():
    game = SnakeGame(WIDTH, LENGTH, border=1)
    training_gif = imageio.get_writer('training.gif', mode='I')
    final_gif = imageio.get_writer('play.gif', mode='I')

    snake = snakeModel((WIDTH + BORDER, LENGTH + BORDER), len(ACTIONS), Adam(lr=LEARNING_RATE), Huber())
    snake_copy = snakeModel((WIDTH + BORDER, LENGTH + BORDER), len(ACTIONS), Adam(lr=LEARNING_RATE), Huber())
    snake_copy.set_weights(snake.get_weights())
    memory = deque(maxlen=MEMORY_LENGTH)

    # TODO: Experiment with different boards
    epsilon = MAX_EPSILON
    steps = 0
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        total_reward = 0
        board, reward, done, info = game.reset()
        done = False
        while not done:
            steps += 1
            if True:
                file_name = PATH + "/" + epoch + "_" + steps
                plot_board(file_name, board)
                training_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)

            action = decide(snake, board, epsilon)
            next_board, reward, done, info = game.step(action)

            memory.append([board, action, reward, next_board, done])
            if len(memory) >= MIN_EPOCHS and (steps % TRAIN_STEPS == 0 or done):
                train(memory, snake, snake_copy)

            board = next_board
            total_reward += reward
            if done:
                print('Reward: total = {}, final = {}'.format(total_reward, epoch, reward))
                total_reward += 1
            if steps >= 100:
                print('Copying snake network weights to snake copy.')
                snake_copy.set_weights(snake.get_weights())
                steps = 0
                break
            # TODO: Evaluate
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)

    # TODO: Generate rewards plot
    # TODO: Final Play
