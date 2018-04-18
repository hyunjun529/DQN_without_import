from random import randint

### random agent
def think(hands, history, old_games):
    return hands[randint(1, len(hands)) - 1]

##############################################################################

if __name__ == '__main__':
    hands_all = ['5', '4', '3', '2', '1', '!']
    history = [['5', '2'], ['4', '3'], ['3', '4'], ['1', '5']]
    hands = hands_all
    old_games = []
    print(think(hands, history, old_games))