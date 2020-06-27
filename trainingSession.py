from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Training.training import TicTraining
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Players.randomMemory import RandomMemory
from Players.greedyMemory import GreedyMemory

SAMPLE_SIZE = 200
CAPACITY = 1_000_000

GAMMA = 0.9
REWARD_INVALID_SCORE: float = -50
REWARD_WIN = 10
REWARD_LOSE = -1
FIXED_BATCH = False
EPS_MIN: float = 0.1
NUM_GAMES = 50_000
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20
STUPID_PLAYER_RANDOMNESS = 1

board_size = 3


trainer = AITrainer(board_size, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
trainer.model_network.load_weights("pesi_angela.pt")
trainer.target_network.load_weights("pesi_angela.pt")
#randomMemoryPlayer = RandomMemory(board_size, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, trainer.replayMemory)
#players = [RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS), trainer]
players = [GreedyPlayer(board_size), trainer]
greedyMemoryPlayer = GreedyMemory(board_size, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, trainer.replayMemory)

game = TicTraining(players, board_size)
#game_memory = TicTraining([RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS), randomMemoryPlayer], board_size)
game_memory = TicTraining([GreedyPlayer(board_size), greedyMemoryPlayer], board_size)
count_invalid = 0
count_win = 0
count_draws = 0
count_lose = 0

for i in range(100_000):
    game_memory.play()
    game_memory.reset()
    game_memory.players = [game_memory.players[1], game_memory.players[0]]

wins = []
invalids = []
num_wins = 0

for i in range(1, NUM_GAMES):
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
    if i % 10 == 0:
        print(i)
        print("invalid: ", count_invalid)
        print("wins, draws, losses ", count_win, count_draws, count_lose)
        print("loss: ", trainer.model_network.loss)

        if count_invalid == 0:
            # print(invalids)
            #print(wins)
            trainer.model_network.save_weights("pesi_angela")

        count_invalid = 0
        count_win = 0
        count_draws = 0
        count_lose = 0

    game.reset()
    game.players = [game.players[i % 2], game.players[(i+1) % 2]]
    game_memory.players = [game_memory.players[i % 2], game_memory.players[(i+1) % 2]]

print(wins)
print(invalids)
