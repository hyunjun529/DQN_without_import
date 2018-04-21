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

MAX_EPISODE = 100000


env = envFive()


if __name__ == '__main__':
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        CHECK_POINT_DIR = "./"
        checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            except:
                print("Error on loading old network weights")
        else:
            print("Could not find old network weights")

        cnt_win = 0
        cnt_lose = 0
        cnt_draw = 0

        for episode in range(MAX_EPISODE):
            done = False
            state = env.reset()
            info = []
            
            step_count = 0
            while not done:
                action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, info = env.step(action)

                if done:
                    reward = -1

                state = next_state
                step_count += 1

            if info[0] > info[1]:
                cnt_win += 1
            elif info[0] < info[1]:
                cnt_lose += 1
            else:
                cnt_draw += 1

            if episode % 200 == 0:
                print(str(episode) + " : {}승 {}패 {}무 ({})".format(cnt_win,cnt_lose,cnt_draw,(cnt_win)/(cnt_win+cnt_lose+0.001)))
