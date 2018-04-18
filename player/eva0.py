### simple strategy

def think(hands, history, old_games):
    his_boku = [row[0] for row in history]
    his_teki = [row[1] for row in history]

    if len(history) == 0:
        return '5'
    if len(history) == 1:
        return '4'
    if len(history) == 2:
        return '3'

    if len(history) == 3 or len(history) == 4:
        if '!' in hands:
            if '4' in his_teki and '1' in hands:
                return '1'
            else :
                return '!'
        return '1'
        
    if len(history) == 5:
        return '2'

###############################################################

hands_all = ['5', '4', '3', '2', '1', '!']

history = [['5', '2'], ['4', '3'], ['3', '4'], ['1', '5']]

hands = hands_all

old_games = []

print(think(hands, history, old_games))

action = 1
action_idx = round(len(hands) / 10  * action - 0.51)

print(hands)
print(action_idx)
print(hands[action_idx])