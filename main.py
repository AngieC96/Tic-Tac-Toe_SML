from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Training.training import TicTraining
from Players.randomPlayer import RandomPlayer
<<<<<<< HEAD
from Players.greedyPlayer import GreedyPlayer
=======
from Players.randomMemory import RandomMemory
>>>>>>> 80788e52adbe2c22f60a7fb91081ca30326107e5

SAMPLE_SIZE = 200
CAPACITY = 1_000_000

HIDDEN = 30
GAMMA = 0.5

REWARD_INVALID_SCORE: float = -5
REWARD_WIN = 1
REWARD_LOSE = -1
FIXED_BATCH = False
only_valid_moves = True
EPS_MIN: float = 0.1
NUM_GAMES = 15_00
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20
STUPID_PLAYER_RANDOMNESS = 0

<<<<<<< HEAD
board_size = 3
players = [RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS), GreedyPlayer(board_size)]
game = TicGame(players, board_size)
game.play()
game.reset()


trainer = AITrainer(board_size, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
randomMemoryPlayer =RandomMemory(board_size,REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, trainer.replayMemory)
board_size = 3
players = [RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS), trainer]
game = TicTraining(players, board_size)
game_memory = TicTraining([RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS),randomMemoryPlayer],boardsize )
count_invalid = 0
count_win = 0
count_draws = 0
count_lose = 0

for i in range(NUM_GAMES):
    game_memory.play()
    game_memory.reset()
    if game.play():             #no invalid moves
        if game.winner == True:
            if trainer.winner == True: count_win += 1
            else: count_lose += 1
        else: count_draws += 1
    else: count_invalid += 1    #invalid move

    if i % 10 == 0 and i > 0:
        print(i)
        print("invalid: ", count_invalid)
        print("wins, draws, losses ", count_win, count_draws, count_lose)
        print("loss: ", trainer.model_network.loss)
        count_invalid = 0
        count_win = 0
        count_draws = 0
        count_lose = 0

    game.reset()
