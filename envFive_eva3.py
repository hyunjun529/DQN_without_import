import numpy as np
import tensorflow as tf
import random
from collections import deque
from random import randint

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


class envFive:
    def __init__(self):
        self.history = []
        self.hands_A = ['5', '4', '3', '2', '1', '!']
        self.hands_B = ['5', '4', '3', '2', '1', '!']
        self.old_games = []

        self.turn = 1
        self.score = 0 # A = - , B = +
        self.winner = 0 # A = -1, B = +1, Draw = 0

        self.record_win = 0
        self.record_lose = 0
        self.record_draw = 0

    def reset(self):
        self.history = []
        self.hands_A = ['5', '4', '3', '2', '1', '!']
        self.hands_B = ['5', '4', '3', '2', '1', '!']
        self.old_games = []

        self.turn = 1
        self.score = 0 # A = - , B = +
        self.winner = 0 # A = -1, B = +1, Draw = 0

        self.record_win = 0
        self.record_lose = 0
        self.record_draw = 0

        # 첫 손패는 랜덤으로 발생
        state, reward, done, info = self.step(self.getRandomCardA())
        return state
    
    def getRandomCardA(self):
        return randint(0, 9)

    def getRandomCardB(self):
        return self.hands_B[randint(1, len(self.hands_B)) - 1]

    def step(self, action):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        reward = 0
        done = False
        info = ["info"]

        action_idx = int(round(len(self.hands_A) / 10  * (action + 1) - 0.51))
        
        play_A = self.hands_A[action_idx]
        play_B = self.getRandomCardB()

        # 승패 판정
        winner = 0
        if play_A == '1' and play_B == '5':
            winner = -1
        elif play_B == '1' and play_A == '5':
            winner = 1
        elif play_A == '!' and (play_B == '2' or play_B == '4'):
            winner = -1
        elif play_B == '!' and (play_A == '2' or play_A == '4'):
            winner = 1
        elif play_A == play_B:
            winner = 0
        elif play_A > play_B:
            winner = -1
        elif play_B > play_A:
            winner = 1
        else :
            print("ERROR! ERROR! 주인님 나 망가졌어! 고쳐줘!")
            return -1

        # 승부 기록 및 다음 게임 준비
        self.score += winner
        self.hands_A.remove(play_A)
        self.hands_B.remove(play_B)
        self.history.append([play_A, play_B])

        # reward : 이겼으면 1, 졌으면 -1,
        reward = -winner

        # 승부 결과 출력 (A를 기준으로 승, 패)
        display = '?'
        if winner == 0:
            display = '무'
            self.record_draw += 1
        elif winner < 0:
            display = '승'
            self.record_win += 1
        elif winner > 0:
            display = '패'
            self.record_lose += 1
        # print(str(self.turn) + '턴 : ' + play_A + ", " + play_B + " : " + display + "reward : " + str(reward))

        # next_state : 현재 history로 갱신
        his_boku = [row[0] for row in self.history]
        his_teki = [row[1] for row in self.history]
        cnt = 1
        for i in his_boku:
            tmp_idx = 0
            if i != '!':
                tmp_idx = int(i)
            state[tmp_idx] = cnt
            cnt += 1
        cnt = 1
        for i in his_teki:
            tmp_idx = 0
            if i != '!':
                tmp_idx = int(i)
            state[tmp_idx + 6] = cnt
            cnt += 1

        # 게임 종료
        self.turn += 1
        if abs(self.score) >= 3 or self.turn >= 7:
            done = True

        return state, reward, done, [self.record_win, self.record_lose, self.record_draw]


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
        CHECK_POINT_DIR = "./save/"
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "00", global_step=global_step)