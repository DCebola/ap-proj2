#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 07:35:58 2021

"""
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
import os


def plot_board(name, _board, text):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(_board)
    plt.gca().text(3, 3, text, fontsize=45, color='yellow')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)


def plot_rewards(path, name, training, evaluation):
    fig = plt.figure(figsize=(10, 10))
    xticks = range(0, len(training), 1)
    plt.plot(xticks, training, linestyle='--', label='Training', color='cornflowerblue')
    plt.plot(xticks, evaluation, linestyle='-', label='Evaluation', color='orange')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(path + "/" + name)
    plt.close(fig)


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
    layer = Flatten(name='features')(layer)
    layer = Dense(64, input_shape=state_shape, activation='relu', kernel_initializer=init)(layer)
    layer = Dense(32, activation='relu', kernel_initializer=init)(layer)
    outputs = Dense(action_shape, activation='softmax', kernel_initializer=init)(layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def decide(agent, state, _epsilon):
    if np.random.rand() <= _epsilon:
        return np.random.choice(POLICY, 1)
    predicted = agent.predict(state.reshape(BATCHED_SHAPE)).flatten()
    # print("PREDICTED:", np.argmax(predicted) - 1)
    return np.argmax(predicted) - 1


def train(replay_memory, model, target_model):
    mini_batch = random.sample(replay_memory, BATCH_SIZE)
    current_states = np.array([transition[0] for transition in mini_batch])
    qs = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs = target_model.predict(new_current_states)
    x = []
    y = []
    for index, (observation, _action, _reward, new_observation, _done) in enumerate(mini_batch):
        if not _done:
            max_future_q = _reward + DISCOUNT * np.max(future_qs[index])
        else:
            max_future_q = _reward
        current_qs = qs[index]
        current_qs[_action] = max_future_q
        x.append(observation)
        y.append(current_qs)
    model.fit(np.array(x), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=True)


def generateGame(scenario):
    return SnakeGame(WIDTH, HEIGHT, border=BORDER,
                     food_amount=scenario.get('FOOD_AMOUNT'),
                     max_grass=scenario.get('MAX_GRASS'),
                     grass_growth=scenario.get('GRASS_GROWTH'))


RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PATH = "logs/" + datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")

BORDER = 1
WIDTH = 30
HEIGHT = 30
BOARD_SHAPE = (WIDTH + 2 * BORDER, HEIGHT + 2 * BORDER, 3)
BATCHED_SHAPE = (1, WIDTH + 2 * BORDER, HEIGHT + 2 * BORDER, 3)
POLICY = [-1, 0, 1]

BATCH_SIZE = 64 * 2
MEMORY_LENGTH = 50000
EPOCHS = 300
MIN_EPOCHS = 1000
MAX_STEPS = 300
TRAIN_STEPS = 4
TARGET_DELAY = 100

# TODO: experiment with different values for different heuristics
MAX_EPSILON = 1
MIN_EPSILON = 0.01
DECAY_RATE = 0.01
DISCOUNT = 0.618
LEARNING_RATE = 0.001

if __name__ == '__main__':
    os.makedirs(PATH)
    train_scenarios = [{'FOOD_AMOUNT': 10, 'MAX_GRASS': 0.05, 'GRASS_GROWTH': 0.001},
                       {'FOOD_AMOUNT': 7, 'MAX_GRASS': 0.03, 'GRASS_GROWTH': 0.0005},
                       {'FOOD_AMOUNT': 5, 'MAX_GRASS': 0.02, 'GRASS_GROWTH': 0.0003},
                       {'FOOD_AMOUNT': 3, 'MAX_GRASS': 0.01, 'GRASS_GROWTH': 0.0001},
                       {'FOOD_AMOUNT': 1, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0}]
    eval_scenario = {'FOOD_AMOUNT': 1, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0}

    training_gif = imageio.get_writer(PATH + "/" + 'training.gif', mode='I')
    final_gif = imageio.get_writer(PATH + "/" + 'play.gif', mode='I')
    snake = snakeModel(BOARD_SHAPE, len(POLICY), Adam(lr=LEARNING_RATE), Huber())
    target_snake = snakeModel(BOARD_SHAPE, len(POLICY), Adam(lr=LEARNING_RATE), Huber())
    target_snake.set_weights(snake.get_weights())
    memory = deque(maxlen=MEMORY_LENGTH)

    epsilon = MAX_EPSILON
    game = None
    epochs_per_train_scenario = EPOCHS / len(train_scenarios)
    scenario = 0
    steps = 0
    train_rewards = []
    eval_rewards = []
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        if epoch % epochs_per_train_scenario == 0:
            game = generateGame(train_scenarios[scenario])
            scenario += 1
        board, reward, _, info = game.reset()
        total_reward = 0
        done = False
        while not done:
            if not done:
                file_name = PATH + "/" + str(epoch) + "_" + str(steps) + ".png"
                plot_board(file_name, board, str(epoch) + ", " + str(steps))
                training_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)
            steps += 1
            action = decide(snake, board, epsilon)
            next_board, reward, done, info = game.step(action)

            memory.append([board, action, reward, next_board, done])
            if len(memory) >= MIN_EPOCHS and (steps % TRAIN_STEPS == 0 or done):
                train(memory, snake, target_snake)

            board = next_board
            total_reward += reward
            if steps == MAX_STEPS:
                done = True
            if done:
               # print('Train Reward: total = {}, final = {}'.format(total_reward, reward))
                file_name = PATH + "/" + str(epoch) + "_" + str(steps) + ".png"
                plot_board(file_name, board, str(epoch) + ", " + str(steps))
                training_gif.append_data(imageio.imread(file_name))
                training_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)
                train_rewards.append(total_reward)
            if steps >= TARGET_DELAY:
                #print('Copying snake network weights to snake copy.')
                target_snake.set_weights(snake.get_weights())
                steps = 0
                # break
        total_reward = 0
        done = False
        eval_game = generateGame(eval_scenario)
        board, reward, _, info = eval_game.reset()
        while not done:
            if epoch + 1 == EPOCHS:
                file_name = PATH + "/" + str(epoch) + "_" + str(steps) + ".png"
                plot_board(file_name, board, str(epoch) + ", " + str(steps))
                final_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)
            action = decide(snake, board, epsilon)
            board, reward, done, info = eval_game.step(action)
            total_reward += reward
            if done:
                #print('Eval Reward: total = {}, final = {}'.format(total_reward, reward))
                if epoch + 1 == EPOCHS:
                    file_name = PATH + "/" + str(epoch) + "_" + str(steps) + ".png"
                    plot_board(file_name, board, str(epoch) + ", " + str(steps))
                    final_gif.append_data(imageio.imread(file_name))
                    final_gif.append_data(imageio.imread(file_name))
                    os.remove(file_name)
                eval_rewards.append(total_reward)
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)
    plot_rewards(PATH, 'Rewards.png', train_rewards, eval_rewards)
