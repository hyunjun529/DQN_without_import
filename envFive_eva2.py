import numpy as np
from random import randint

import player.eva0 as eva0
import player.eva1 as eva1
import player.eva2 as eva2


def envFive(agent, B):
    history = []
    hands_A = ['5', '4', '3', '2', '1', '!']
    hands_B = ['5', '4', '3', '2', '1', '!']
    old_games = []
    
    turn = 1
    score = 0 # A = - , B = +
    winner = 0 # A = -1, B = +1, Draw = 0

    play_A = '?'
    play_B = '?'

    record_win = 0
    record_lose = 0
    record_draw = 0

    isEnd = True

    # rl
    action = 0
    action_idx = 0
    state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    reward = 0

    while(isEnd):
        # 플레이 받기
        # play_A = A(hands_A, history, old_games)
        '''
        if turn == 1:
            play_A = hands_A[randint(1, len(hands_A)) - 1]
        '''
        if turn == 6:
            play_A = hands_A[0]
        else:
            action = agent.get_action(str(state))
            #print("====================action : " + str(action))
            action_idx = int(round(len(hands_A) / 10  * (action + 1) - 0.51))
            play_A = hands_A[action_idx]
            '''
            print("====================action : " + str(action))
            print("====================act_idx : " + str(action_idx))
            print("====================play_A : " + play_A)
            '''

        play_B = B(hands_B, history, old_games)

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
        score += winner
        hands_A.remove(play_A)
        hands_B.remove(play_B)
        history.append([play_A, play_B])

        # 승부 결과 출력 (A를 기준으로 승, 패)
        display = '?'
        if winner == 0:
            display = '무'
            record_draw += 1
        elif winner < 0:
            display = '승'
            record_win += 1
        elif winner > 0:
            display = '패'
            record_lose += 1
        #print(str(turn) + '턴 : ' + play_A + ", " + play_B + " : " + display)        

         # next_state : 현재 history로 갱신
        his_boku = [row[0] for row in history]
        his_teki = [row[1] for row in history]
        cnt = 1
        for i in his_boku:
            tmp_idx = 0
            if i != '!':
                tmp_idx = int(i)
            next_state[tmp_idx] = cnt
            cnt += 1
        cnt = 1
        for i in his_teki:
            tmp_idx = 0
            if i != '!':
                tmp_idx = int(i)
            next_state[tmp_idx + 6] = cnt
            cnt += 1
        '''
        print("====================next_state : " + str(next_state))
        '''

        # reward : 이겼으면 1, 졌으면 -1,
        reward = -winner

        # learn : 지식이 늘었다
        '''
        print("====================state : " + str(state))
        print("====================action : " + str(action) + " (realplay : " + play_A + ")")
        print("====================reward : " + str(reward))
        print("====================next_state : " + str(next_state))
        '''
        agent.learn(str(state), action, reward, str(next_state))
        state = next_state

        # 게임 종료
        turn += 1
        if abs(score) >= 3 or turn >= 7:
            isEnd = False

    # 게임 결과
    '''
    print("A는 \""\
        + str(record_win) + "승 "\
        + str(record_lose) + "패 "\
        + str(record_draw) + "무\" 를 기록했다")
    '''
    #print("경기 기록 : " + str(history))

    return (record_win - record_lose)


if __name__ == '__main__':
    agent = eva2.QLearningAgent()
    memory = 0
    savepoint = 0
    rate = 1000000
    while(True):
        memory += envFive(agent, eva1.think)
        if savepoint == rate:
            agent.save_q_table()
            print(memory/rate) # 승률
            memory = 0
            savepoint = 0
        savepoint+=1
