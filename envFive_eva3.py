import numpy as np
import tensorflow as tf
import random
import os

from collections import deque
from random import randint
from envFive import envFive

import player.dqn as dqn


INPUT_SIZE = 12
OUTPUT_SIZE = 10

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 100000
BATCH_SIZE = 64

# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
#EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.2


def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:
    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)


def train_minibatch(DQN: dqn.DQN, train_batch: list) -> float:
    state_array = np.vstack([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    next_state_array = np.vstack([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])

    X_batch = state_array
    y_batch = DQN.predict(state_array)

    Q_target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), axis=1) * ~done_array
    y_batch[np.arange(len(X_batch)), action_array] = Q_target

    # Train our network using target and predicted Q values on each episode
    loss, _ = DQN.update(X_batch, y_batch)

    return loss


env = envFive()


if __name__ == '__main__':
    print("HELL WORLD")

    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)
    
    # 연쇄를 끊는다...크큭
    combo = 0
    max_combo = 17
    endgame = False

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        CHECK_POINT_DIR = "./save"

        for episode in range(MAX_EPISODE):
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            done = False
            state = env.reset()
            info = []
            
            step_count = 0
            while not done:

                if np.random.rand() < e:
                    action = env.getRandomCardA()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, info = env.step(action)

                if done:
                    reward = -1

                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                step_count += 1

                if len(replay_buffer) > BATCH_SIZE and not endgame:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    train_minibatch(mainDQN, minibatch)

            # 한 게임에서 승패여부 판정 및 출력
            #print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))
            result = "?"
            if info[0] > info[1]:
                result = "승리"
                combo += 1
            elif info[0] < info[1]:
                result = "패배"
                combo = 0
            else:
                result = "동점"

            if combo > 11:
                print(str(episode) + " : " + result + " : " + str(combo))

            # 최근 100 게임의 평균 측정
            last_100_game_reward.append(step_count)
            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)
                if avg_reward > 4.55:
                    print("Reached goal within {} episodes with avg reward {}, combo {}".format(episode, avg_reward, combo))

            if not endgame and combo >= max_combo:
                print(str(max_combo) + "분할!")
                endgame = True

            # 모델 저장
            if combo > 11:
                if not os.path.exists(CHECK_POINT_DIR):
                    os.makedirs(CHECK_POINT_DIR)
                saver.save(sess, CHECK_POINT_DIR, global_step=episode)