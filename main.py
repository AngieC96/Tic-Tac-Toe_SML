from TicGame.tic_game import TicGame
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Training.network import Network
from Players.AIPlayer import AIPlayer

SAMPLE_SIZE = 500
CAPACITY = 1_000_000

GAMMA = 0.99
REWARD_INVALID_SCORE: float = 0
REWARD_WIN = 1
REWARD_LOSE = -2
EPS_MIN: float = 0.01
NUM_GAMES = 50_000
EPS_DECAY: float = 0.001
UPDATE_TARGET_EVERY = 100

FIXED_BATCH = False
only_valid_moves = True

STUPID_PLAYER_RANDOMNESS = 1


board_size = 3

aiplayer = AIPlayer(board_size)
aiplayer.network.load_weights(file="vincente.pt")
players = [RandomPlayer(board_size), aiplayer]
#players = [GreedyPlayer(board_size), GreedyPlayer(board_size)]
game = TicGame(players, board_size)

if not game.play():
    print("AI made an illegal move!")
else:
    if players[1].winner:
        print("The winner is ", players[1].__str__())
    elif players[0].winner:
        print("The winner is ", players[0].__str__())
    else:
        print("Draw")

game.reset()
