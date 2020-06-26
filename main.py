from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Players.randomPlayer import RandomPlayer

SAMPLE_SIZE = 200
CAPACITY = 1_000_000

GAMMA = 0.5
REWARD_INVALID_SCORE: float = -50
REWARD_WIN = 10
REWARD_LOSE = -1
FIXED_BATCH = False
EPS_MIN: float = 0.1
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20


board_size = 3

aiplayer = AITrainer(board_size, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
aiplayer.model_network.load_weights(file="pesi_angela.pt")
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
