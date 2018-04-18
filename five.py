import numpy as np

import player.eva0 as eva0
import player.eva1 as eva1

import learnDQN
import playDQN

def five(A, B):
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

    while(isEnd):
        # 플레이 받기
        play_A = A(hands_A, history, old_games)
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
        print(str(turn) + '턴 : ' + play_A + ", " + play_B + " : " + display)        

        # 게임 종료
        turn += 1
        if abs(score) >= 3 or turn >= 6:
            isEnd = False

    # 게임 결과
    print("A는 \""\
        + str(record_win) + "승 "\
        + str(record_lose) + "패 "\
        + str(record_draw) + "무\" 를 기록했다")
    print("경기 기록 : " + str(history))

    return 0

if __name__ == '__main__':
    print("HELL WORLD!")
    print("게임을 시작하지")
    for i in range(11):
        five(eva0.think, eva1.think)