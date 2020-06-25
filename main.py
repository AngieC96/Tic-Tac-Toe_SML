from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Training.training import TicTraining
from Players.randomPlayer import RandomPlayer
from Players.randomMemory import RandomMemory

SAMPLE_SIZE = 200 #1024 * 5
CAPACITY = 1_000_000

HIDDEN = 30
GAMMA = 0.5

REWARD_INVALID_SCORE: float = -50
REWARD_WIN = 10
REWARD_LOSE = -1
FIXED_BATCH = False
only_valid_moves = True
EPS_MIN: float = 0.1
NUM_GAMES = 15_000 #50_000
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20
STUPID_PLAYER_RANDOMNESS = 0

boardsize = 3


trainer = AITrainer(boardsize, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
randomMemoryPlayer =RandomMemory(boardsize,REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, trainer.replayMemory)
board_size = 3
players = [RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS), trainer]
game = TicTraining(players, board_size)
game_memory = TicTraining([RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS),randomMemoryPlayer],boardsize )
count_invalid = 0
count_win = 0
count_draws = 0
count_lose = 0

for i in range(200):
    game_memory.play()
    game_memory.reset()
    game_memory.players = [game_memory.players[1], game_memory.players[0]]

wins = []
invalids = []
num_wins = 0
for i in range(NUM_GAMES):

    game_memory.play()
    game_memory.reset()
    if game.play():             #no invalid moves
        if game.winner == True:
            if trainer.winner == True: count_win += 1
            else: count_lose += 1
        else: count_draws += 1
    else: count_invalid += 1    #invalid move
    wins.append(count_win)
    invalids.append(count_invalid)
    if i % 10 == 0 and i > 0:
        print(i)
        print("invalid: ", count_invalid)
        print("wins, draws, losses ", count_win, count_draws, count_lose)
        print("loss: ", trainer.model_network.loss)

        if count_win == 10:

            trainer.model_network.save_weights("vincente")

        count_invalid = 0
        count_win = 0
        count_draws = 0
        count_lose = 0

    game.reset()
    game.players=[game.players[i%2], game.players[(i+1)%2]]
    game_memory.players = [game_memory.players[i%2], game_memory.players[(i+1)%2]]

print(wins)
print(invalids)