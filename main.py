from TicGame.tic_game import TicGame
from Training.trainer import AITrainer
from Training.training import TicTraining
from Players.randomPlayer import RandomPlayer
from Players.randomMemory import RandomMemory
from csv import writer

SAMPLE_SIZE = 200 #1024 * 5
CAPACITY = 1_000_000


GAMMA = 0.5


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

boardsize = 3


trainer = AITrainer(boardsize, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, SAMPLE_SIZE, CAPACITY, GAMMA,
                    EPS_MIN, EPS_DECAY, fixed_batch=FIXED_BATCH, double_q_interval=UPDATE_TARGET_EVERY)
randomMemoryPlayer =RandomMemory(boardsize,REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE, trainer.replayMemory)
board_size = 3
players = [trainer,RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS)]
game = TicTraining(players, board_size)
game_memory = TicTraining([randomMemoryPlayer,RandomPlayer(board_size, STUPID_PLAYER_RANDOMNESS)], boardsize )
count_invalid = 0
count_win = 0
count_draws = 0
count_loose = 0

loose_min = 10000

# for i in range(200):
#     game_memory.play()
#     game_memory.reset()
#     game_memory.players = [game_memory.players[1], game_memory.players[0]]

wins = []
invalids = []
loose = []
draw = []
loss_list = []
num_wins = 0
for i in range(NUM_GAMES):
    # game_memory.play()
    # game_memory.reset()
    if game.play():             #no invalid moves
        if game.winner == True:
            if trainer.winner == True: count_win += 1
            else: count_loose += 1
        else: count_draws += 1
    else: count_invalid += 1    #invalid move

    loss_list.append(trainer.model_network.loss)
    if i % 100 == 0 and i > 0:
        wins.append(count_win)
        invalids.append(count_invalid)
        loose.append(count_loose)
        draw.append(count_draws)
        print(i)
        print("invalid: ", count_invalid,"eps ", trainer.eps_greedy_value, "w, d, l ",
              count_win, count_draws, count_loose, "loss: ", trainer.model_network.loss)

        if count_loose < loose_min:
            # print(invalids)
            trainer.model_network.save_weights("data/vincente")

        with open("data/wins.csv",'w', newline='') as w:
            csv_writer = writer(w)
            csv_writer.writerow(wins)
        with open("data/invalids.csv",'w',newline='') as w:
            csv_writer = writer(w)
            csv_writer.writerow(invalids)
        with open("data/loose.csv",'w',newline='') as w:
            csv_writer = writer(w)
            csv_writer.writerow(loose)
        with open("data/draw.csv",'w',newline='') as w:
            csv_writer = writer(w)
            csv_writer.writerow(draw)
        with open("data/loss_list.csv",'w',newline='') as w:
            csv_writer = writer(w)
            csv_writer.writerow(loss_list)
        loose_min = count_loose

        count_invalid = 0
        count_win = 0
        count_draws = 0
        count_loose = 0


    game.reset()
    trainer.update_eps(i)
    game.players=[game.players[i%2], game.players[(i+1)%2]]
    #game_memory.players = [game_memory.players[i%2], game_memory.players[(i+1)%2]]
