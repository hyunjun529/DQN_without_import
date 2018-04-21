import numpy as np
import tensorflow as tf
import os
from random import randint
from envFive import envFive


# env
env = envFive()


# hyper
name = 'eva4'
CHECK_POINT_DIR = "./save"

input_size = 12
output_size = 1

max_num_episodes = 50000
learning_rate = 1e-4


# set uniform
ob_space = np.zeros((input_size,))
ac_space = np.zeros((output_size,))
batch_of_observations = [None] + list(ob_space.shape)

# input_size = state = observation
x = tf.placeholder(dtype=tf.float32, shape=batch_of_observations, name="ob")
ob = x

width = 128
dense1_w = tf.get_variable("dense1_w", [x.get_shape()[1], width])
dense1_b = tf.get_variable("dense1_b", [width])
x = tf.nn.relu(tf.matmul(x, dense1_w) + dense1_b)

'''
width = 128
dense2_w = tf.get_variable("dense2_w", [x.get_shape()[1], width])
dense2_b = tf.get_variable("dense2_b", [width])
x = tf.nn.relu(tf.matmul(x, dense2_w) + dense2_b)
'''

# action
assert(len(ac_space.shape)==1)
width = ac_space.shape[0]
final_w = tf.get_variable("final_w", [x.get_shape()[1], width])
final_b = tf.get_variable("final_b", [width])
x = tf.nn.sigmoid(tf.matmul(x, final_w) + final_b)
action_pred = x

# Y (fake) and advantages (rewards)
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function: log_likelihood * advantages
#log_lik = -tf.log(Y * action_pred + (1 - Y) * (1 - action_pred))     # using author(awjuliani)'s original cost function (maybe log_likelihood)
log_lik = -Y*tf.log(action_pred) - (1 - Y)*tf.log(1 - action_pred)    # using logistic regression cost function
loss = tf.reduce_sum(log_lik * advantages)

# learning (MAGIC)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


# tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Savor and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")


# replay
observation = env.reset()

cnt_win = 0
cnt_lose = 0
cnt_draw = 0

while True:
    x = np.reshape(observation, [1, input_size])
    action_prob = sess.run(action_pred, feed_dict={ob: x})
    action = action_prob[0][0] # only 1 action..
    observation, reward, done, info = env.step(action)
    if done:
        if info[0] > info[1]:
            cnt_win += 1
        elif info[0] < info[1]:
            cnt_lose += 1
        else:
            cnt_draw += 1
        if (cnt_win + cnt_lose + cnt_draw) % 200 == 0:
            print(" : {}승 {}패 {}무 ({})".format(cnt_win,cnt_lose,cnt_draw,(cnt_win)/(cnt_win+cnt_lose+0.001)))
        observation = env.reset()

sess.close()