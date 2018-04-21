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


# run
EPISODE_100_REWARD_LIST = []
combo = 0
for step in range(max_num_episodes):
    xs = np.empty(shape=[0, input_size])
    ys = np.empty(shape=[0, 1])
    rewards = np.empty(shape=[0, 1])
    info = []

    reward_sum = 0
    observation = env.reset()

    while True:
        xr = np.reshape(observation, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={ob: xr})

        # Determine the output based on our net, allowing for some randomness
        action = action_prob[0][0] # only 1 action..

        # Append the observations and outputs for learning
        xs = np.vstack([xs, xr])
        ys = np.vstack([ys, action])  # Fake action

        # Determine the outcome of our action
        observation, reward, done, info = env.step(action)
        rewards = np.vstack([rewards, reward])
        reward_sum += reward
        
        if done:
            discounted_rewards = rewards
            # Normalization
            discounted_rewards = (discounted_rewards - discounted_rewards.mean())/(discounted_rewards.std() + 1e-7)
            l, _ = sess.run([loss, train],
                            feed_dict={ob: xs, Y: ys, advantages: discounted_rewards})

            EPISODE_100_REWARD_LIST.append(reward_sum)
            if len(EPISODE_100_REWARD_LIST) > 100:
                EPISODE_100_REWARD_LIST = EPISODE_100_REWARD_LIST[1:]
            break
        
    if info[0] > info[1]:
        combo += 1
    elif info[0] < info[1]:
        combo = 0

    if combo > 9:
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR, global_step=step)

    # Print status
    if step % 2500 == 0:
        print(f"[Episode {step:>5d}] Reward: {reward_sum:>4} Loss: {l:>10.5f}")
    
    if np.mean(EPISODE_100_REWARD_LIST) >= 5:
        print(f"Game Cleared within {step} steps with the average reward: {np.mean(EPISODE_100_REWARD_LIST)}")
        break
