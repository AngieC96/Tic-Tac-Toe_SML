from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Training.training import TicTraining
from Training.network import Network
from Players.randomPlayer import RandomPlayer

SAMPLE_SIZE = 200
CAPACITY = 1_000_000

HIDDEN = 30
GAMMA = 0.5

REWARD_INVALID_SCORE: float = -1
REWARD_WIN = 1
REWARD_LOSE = -1
FIXED_BATCH = False
EPS_MIN: float = 0.1
NUM_GAMES = 1500
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20

boardsize = 3


trainer = AITrainer(boardsize, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)

board_size = 3
players = [RandomPlayer(board_size), trainer]
game = TicTraining(players, board_size)
count = 0
for i in range(NUM_GAMES):
    count += game.play()
    if i % 10 == 0 and i > 0:
        print(i)
        print(count)
        print(trainer.model_network.loss)
        count = 0

trainer.model_network.save_weights("saved_weights")

aiplayer = AITrainer(boardsize, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
aiplayer.model_network.load_weights(file="saved_weights.pt")
players = [RandomPlayer(board_size), aiplayer]
game = TicGame(players, board_size)
game.play()

