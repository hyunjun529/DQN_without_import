import numpy as np
import tensorflow as tf
import os
from random import randint
from envFive import envFive


# env
env = envFive()


# hyper
name = 'eva4'
CHECK_POINT_DIR = "./"

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

width = 64
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

# it's danger!
np.set_printoptions(precision=5, suppress=True)

print(dense1_w)
print(dense1_b)
print(final_w)
print(final_b)

array = dense1_w.eval(sess)
print (array)
array = dense1_b.eval(sess)
print (array)

array = final_w.eval(sess)
print (array)
array = final_b.eval(sess)
print (array)