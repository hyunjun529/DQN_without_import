import player.dqn

from random import randint

### random agent
def think(hands, history, old_games):
    return hands[randint(1, len(hands)) - 1]

##############################################################################



##############################################################################

if __name__ == '__main__':
   print("HELL WORLD")