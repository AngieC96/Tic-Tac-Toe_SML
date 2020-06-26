from TicGame.tic_game import TicGame
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Training.network import Network
from Players.AIPlayer import AIPlayer

SAMPLE_SIZE = 500
CAPACITY = 1_000_000

GAMMA = 0.5
REWARD_INVALID_SCORE: float = -50
REWARD_WIN = 10
REWARD_LOSE = -1
FIXED_BATCH = False
EPS_MIN: float = 0.1
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 200


board_size = 3

net = Network(board_size)
net.load_weights(file="pesi_angela.pt")
aiplayer = AIPlayer(board_size, net)
players = [RandomPlayer(board_size), aiplayer]
game = TicGame(players, board_size)

if not game.play():
    print("AI made an illegal move!")
else:
    if players[1].winner:
        print("The winner is ", players[1].__str__())
    if players[0].winner:
        print("The winner is ", players[0].__str__())
    else:
        print("Draw")

game.reset()
