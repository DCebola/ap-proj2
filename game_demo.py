#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 07:35:58 2021

"""
from snake_game import SnakeGame
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import imageio
from tqdm import tqdm
import os
from functools import partial
import pickle

tqdm = partial(tqdm, position=0, leave=True)


def plot_board(name, _board, text):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(_board)
    plt.gca().text(3, 3, text, fontsize=45, color='yellow')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)


def plot_apples_steps(path, name, rewards, apples, _steps):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].set_xlabel("Games")
    axs[0].set_ylabel("Apples")
    axs[0].plot(range(0, len(apples), 1), apples, linestyle='-', label='Apples', color='red')
    axs[1].set_xlabel("Games")
    axs[1].set_ylabel("Steps")
    axs[1].plot(range(0, len(_steps), 1), _steps, linestyle='-', label='Steps', color='cornflowerblue')
    axs[2].set_xlabel("Games")
    axs[2].set_ylabel("Total Reward")
    axs[2].plot(range(0, len(rewards), 1), rewards, linestyle='-', label='Steps', color='orange')

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
    return layer


def snakeModel(state_shape, action_shape, optimizer, loss, cnn=False):
    init = HeUniform()
    inputs = Input(shape=state_shape, name='inputs')
    layer = inputs
    if cnn:
        layer = CNNModel(inputs)
    layer = Flatten()(layer)
    layer = Dense(128, input_shape=state_shape, activation='relu', kernel_initializer=init)(layer)
    layer = Dense(64, activation='relu', kernel_initializer=init)(layer)
    outputs = Dense(action_shape, activation='softmax', kernel_initializer=init)(layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def decide(agent, state, _epsilon):
    if np.random.rand() <= _epsilon:
        return np.random.choice(POLICY, 1)
    predicted = agent.predict(state.reshape(BATCHED_SHAPE)).flatten()
    return POLICY[np.argmax(predicted)]


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


def generateGame(scenario, width=30, height=30, border=1):
    return SnakeGame(width, height, border=border,
                     food_amount=scenario.get('FOOD_AMOUNT'),
                     max_grass=scenario.get('MAX_GRASS'),
                     grass_growth=scenario.get('GRASS_GROWTH'))


def check_collisions(next_pos, tails):
    if (next_pos[0] == -1 or next_pos[0] == 33
            or next_pos[1] == -1 or next_pos[1] == 33
            or next_pos in tails):
        return True
    else:
        return False


def heuristic(_, apples, head, tails, direction, _type):
    act = 0
    if _type == RANDOM_PATTERN:
        return np.random.choice(POLICY, 1)  # Random move
    else:
        # Find Closest Apple
        closest_apple = apples[0]
        min_row = 32
        min_col = 32

        for apple in apples:
            d_row = apple[0] - head[0]
            d_col = apple[1] - head[1]
            if d_row <= min_row and d_col <= min_col:
                min_row = d_row
                min_col = d_col
                closest_apple = apple

        # Find Next Move
        d_row = closest_apple[0] - head[0]
        d_col = closest_apple[1] - head[1]

        if _type == L_PATTERN:  # L pattern heuristic
            if abs(d_row) > 0:
                if d_row < 0:
                    ddir = 0  # -x
                else:
                    ddir = 2  # +x
            else:
                if d_col < 0:
                    ddir = 3  # -y
                else:
                    ddir = 1  # +y
        else:
            if abs(d_row) > abs(d_col):  # Diagonal pattern heuristic
                if d_row < 0:
                    ddir = 0  # -x
                else:
                    ddir = 2  # +x
            else:
                if d_col < 0:
                    ddir = 3  # -y
                else:
                    ddir = 1  # +y

        if ddir == 0 and direction == 0:
            act = 0
        if ddir == 1 and direction == 0:
            act = 1
        if ddir == 2 and direction == 0:
            act = 1
        if ddir == 3 and direction == 0:
            act = -1

        if ddir == 0 and direction == 1:
            act = -1
        if ddir == 1 and direction == 1:
            act = 0
        if ddir == 2 and direction == 1:
            act = 1
        if ddir == 3 and direction == 1:
            act = 1

        if ddir == 0 and direction == 2:
            act = 1
        if ddir == 1 and direction == 2:
            act = -1
        if ddir == 2 and direction == 2:
            act = 0
        if ddir == 3 and direction == 2:
            act = 1

        if ddir == 0 and direction == 3:
            act = 1
        if ddir == 1 and direction == 3:
            act = 1
        if ddir == 2 and direction == 3:
            act = -1
        if ddir == 3 and direction == 3:
            act = 0

        # Check If Valid Move
        _done = False
        looping = False

        while not _done:
            if direction == 0:
                if act == -1:
                    next_pos = (head[0], head[1] - 1)
                    if check_collisions(next_pos, tails):
                        act = 0
                    else:
                        _done = True
                if looping:
                    _done = True
                if act == 0:
                    next_pos = (head[0] - 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = 1
                    else:
                        _done = True
                if act == 1:
                    next_pos = (head[0], head[1] + 1)
                    if check_collisions(next_pos, tails):
                        act = -1
                        looping = True
                    else:
                        _done = True
            if direction == 1:
                if act == -1:
                    next_pos = (head[0] - 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = 0
                    else:
                        _done = True
                if looping:
                    _done = True
                if act == 0:
                    next_pos = (head[0], head[1] + 1)
                    if check_collisions(next_pos, tails):
                        act = 1
                    else:
                        _done = True
                if act == 1:
                    next_pos = (head[0] + 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = -1
                        looping = True
                    else:
                        _done = True
            if direction == 2:
                if act == -1:
                    next_pos = (head[0], head[1] + 1)
                    if check_collisions(next_pos, tails):
                        act = 0
                    else:
                        _done = True
                if looping:
                    _done = True
                if act == 0:
                    next_pos = (head[0] + 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = 1
                    else:
                        _done = True
                if act == 1:
                    next_pos = (head[0], head[1] - 1)
                    if check_collisions(next_pos, tails):
                        act = -1
                        looping = True
                    else:
                        _done = True
            if direction == 3:
                if act == -1:
                    next_pos = (head[0] + 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = 0
                    else:
                        _done = True
                if looping:
                    _done = True
                if act == 0:
                    next_pos = (head[0], head[1] - 1)
                    if check_collisions(next_pos, tails):
                        act = 1
                    else:
                        _done = True
                if act == 1:
                    next_pos = (head[0] - 1, head[1])
                    if check_collisions(next_pos, tails):
                        act = -1
                        looping = True
                    else:
                        _done = True
        return act


def generateMemory(length, scenarios):
    mem = deque(maxlen=length)
    if os.path.isfile("memory.pkl"):
        with open(os.getcwd() + "/" + "memory.pkl", "rb") as f:
            try:
                print("Loaded memory.")
                mem = pickle.load(f)
            except Exception:
                pass
    else:
        print("Generating memory with examples.")
        mem_apples = []
        mem_steps = []
        mem_rewards = []
        not_filled = True
        example_count = 0
        games = 0
        heuristic_type = RANDOM_PATTERN
        interesting_examples = 0
        normal_examples = 0
        while not_filled:
            _game = generateGame(scenarios[games % (len(scenarios))])
            _board, _reward, _, _info = _game.reset()
            _done = False
            step = 0
            mem_apples_eaten = 0
            cumulative_rewards = 0
            while not _done:
                score, apples, head, tails, direction = _game.get_state()
                _action = heuristic(score, apples, head, tails, direction, heuristic_type)
                _next_board, _reward, _done, _info = _game.step(_action)
                cumulative_rewards += _reward

                if _reward >= 1:
                    mem_apples_eaten += 1

                if _done or _reward >= 1:
                    example_count += 1
                    mem.append([_board, _action, _reward, _next_board, _done])
                    interesting_examples += 1
                else:
                    if np.random.rand() <= 0.05:
                        example_count += 1
                        normal_examples += 1
                        mem.append([_board, _action, _reward, _next_board, _done])

                """
                file_name = PATH + "/" + str(games) + "_" + str(steps) + ".png"
                plot_board(file_name, board, str(games) + ", " + str(steps))
                memfill_gif.append_data(imageio.imread(file_name))
                memfill_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)
                """

                _board = _next_board
                step = step + 1
                if example_count == round(length / 5):
                    heuristic_type = L_PATTERN
                elif example_count == round(2 * length / 5):
                    heuristic_type = DIAGONAL_PATTERN
                if example_count == length:
                    not_filled = False
                if step == MAX_STEPS:
                    _done = True
            mem_rewards.append(cumulative_rewards)
            mem_apples.append(mem_apples_eaten)
            mem_steps.append(step)
            games += 1
        pickle.dump(mem, open(os.getcwd() + "/" + "memory.pkl", "wb"))
        plot_apples_steps(PATH, 'Heuristic.png', mem_rewards, mem_apples, mem_steps)
        print("Memory saved to pickle file. (" +
              str(interesting_examples * 100 / length) + "% interesting, " +
              str(normal_examples * 100 / length) + "% normal)")
    return mem


def generateProximityRewards(_board, _apples):
    odor_map = np.zeros((_board.shape[0], _board.shape[1]))
    for apple in _apples:
        _i = 0
        _j = 0
        for n in range(PROXIMITY_AREA_L, 0, -1):
            for i in range(_i, n):
                for j in range(_j, n):
                    x = apple[0] - round(n / 2) + i
                    y = apple[1] - round(n / 2) + j
                    try:
                        odor_map[x, y] += INITIAL_PROXIMITY_REWARD_BONUS
                    except Exception:
                        continue
            _i += 1
            _j += 1
        odor_map[apple[0], apple[1]] = INITIAL_APPLE_BONUS
    return odor_map


RANDOM_SEED = 10
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PATH = "logs/"
TRAIN_BORDER = 8
BORDER = 1
WIDTH = 30
HEIGHT = 30
BOARD_SHAPE = (WIDTH + 2 * BORDER, HEIGHT + 2 * BORDER, 3)
BATCHED_SHAPE = (1, WIDTH + 2 * BORDER, HEIGHT + 2 * BORDER, 3)
POLICY = [-1, 0, 1]

BATCH_SIZE = 64 * 2
MEMORY_LENGTH = 100000
MIN_EPOCHS = 1000
EPOCHS = 1000
MAX_STEPS = 500
MAX_EPSILON = 0.6
MIN_EPSILON = 0.01
DECAY_RATE = 0.01

ADD_BONUS = True
INITIAL_APPLE_BONUS = 10
INITIAL_PROXIMITY_REWARD_BONUS = 1
PROXIMITY_AREA_L = 5
BONUS_DECAY = 0.05

DISCOUNT = 0.6
LEARNING_RATE = 0.001
TARGET_DELAY = 100
TRAIN_STEPS = 4

RANDOM_PATTERN = 0
L_PATTERN = 1
DIAGONAL_PATTERN = 2
GENERATE_GIF = False

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    os.makedirs(PATH, exist_ok=True)
    if GENERATE_GIF:
        training_gif = imageio.get_writer(PATH + "/" + 'training.gif', mode='I')

    train_scenarios = [{'FOOD_AMOUNT': 3, 'MAX_GRASS': 0.2, 'GRASS_GROWTH': 0.001},
                       {'FOOD_AMOUNT': 3, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0},
                       {'FOOD_AMOUNT': 1, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0}]

    eval_scenario = {'FOOD_AMOUNT': 1, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0}

    memory = generateMemory(MEMORY_LENGTH,
                            [{'FOOD_AMOUNT': 3, 'MAX_GRASS': 0.2, 'GRASS_GROWTH': 0.001},
                             {'FOOD_AMOUNT': 3, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0},
                             {'FOOD_AMOUNT': 1, 'MAX_GRASS': 0, 'GRASS_GROWTH': 0}])

    snake = snakeModel(BOARD_SHAPE, len(POLICY), Adam(lr=LEARNING_RATE), Huber(), cnn=True)
    target_snake = snakeModel(BOARD_SHAPE, len(POLICY), Adam(lr=LEARNING_RATE), Huber(), cnn=True)
    target_snake.set_weights(snake.get_weights())

    epsilon = MAX_EPSILON
    game = None
    steps = 0
    train_apples = []
    train_steps = []
    train_rewards = []

    train_border = TRAIN_BORDER
    train_width = 32 - 2 * TRAIN_BORDER
    train_height = 32 - 2 * TRAIN_BORDER
    border_inc_interval = round(EPOCHS / (train_border - 1))
    max_apples = 0
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        if epoch % border_inc_interval == 0 and epoch > 0:
            train_border -= 1
            train_height += 2
            train_width += 2
        game = generateGame(train_scenarios[epoch % len(train_scenarios)], train_width, train_height, train_border)
        board, _, _, _ = game.reset()
        if ADD_BONUS:
            _, apples, _, _, _ = game.get_state()
            proximity_bonus_grid = generateProximityRewards(board, apples)
        total_apples = 0
        total_steps = 0
        total_rewards = 0
        reward = 0
        done = False
        while not done:
            if GENERATE_GIF:
                file_name = PATH + "/" + str(epoch) + "_" + str(total_steps) + ".png"
                plot_board(file_name, board, str(epoch) + ", " + str(total_steps))
                training_gif.append_data(imageio.imread(file_name))
                os.remove(file_name)

            total_rewards += reward
            steps += 1
            total_steps += 1
            action = decide(snake, board, epsilon)
            next_board, reward, done, info = game.step(action)
            if ADD_BONUS:
                _, apples, head, _, _ = game.get_state()
            if reward >= 1:
                if ADD_BONUS:
                    proximity_bonus_grid = generateProximityRewards(board, apples)
                total_apples += 1
                if total_apples > max_apples:
                    max_apples = total_apples + 1

            if ADD_BONUS:
                reward += proximity_bonus_grid[head]
            memory.append([board, action, reward, next_board, done])
            if len(memory) >= MIN_EPOCHS and (steps % TRAIN_STEPS == 0 or done):
                train(memory, snake, target_snake)

            board = next_board
            if done:
                train_apples.append(total_apples)
                train_steps.append(total_steps)
                train_rewards.append(total_rewards)
                total_steps = 0
                if total_apples >= max_apples and ADD_BONUS:
                    INITIAL_APPLE_BONUS -= INITIAL_APPLE_BONUS * BONUS_DECAY
                    INITIAL_PROXIMITY_REWARD_BONUS -= INITIAL_PROXIMITY_REWARD_BONUS * BONUS_DECAY
                if steps >= TARGET_DELAY:
                    target_snake.set_weights(snake.get_weights())
                    steps = 0
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)
plot_apples_steps(PATH, 'Train.png', train_rewards, train_apples, train_steps)
