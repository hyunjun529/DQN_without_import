from random import randint
from random import uniform

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
        # ~eva3
        #return randint(0, 9)
        # eva4
        return uniform(0, 1)

    def getRandomCardB(self):
        return self.hands_B[randint(1, len(self.hands_B)) - 1]

    def step(self, action):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        reward = 0
        done = False
        info = ["info"]

        # ~eva3
        #action_idx = int(round(len(self.hands_A) / 10  * (action + 1) - 0.51))
        # eva4
        action_idx = int(round(action * (len(self.hands_A) - 1)))
        
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

