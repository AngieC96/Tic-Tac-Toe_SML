# from Players.AIPlayer import AIPlayer
from Players.player import Player
from Training.network import Network
import numpy as np
from Training.replayMemory import ReplayMemory


class AITrainer(Player):
    rewardInvalidMove: float
    rewardWinning: float
    rewardLosing: float
    state: np.array
    action: int
    invalid: bool
    model_network: Network
    target_network: Network
    replayMemory: ReplayMemory
    gamma: float
    fixed_batch: bool
    eps_greedy_value: float
    eps_min: float
    decay: float
    double_q_interval: int
    double_q_counter: int

    def __init__(self, board_size: int, rewardInvalidMove: float,
                 rewardWinning: float, rewardLosing: float, sample_size: int, capacity: int,
                 gamma: float, eps_min: float, eps_decay: float, fixed_batch: bool = False,
                 double_q_interval: int = 0):

        super().__init__(board_size)
        self.rewardNoScore = 0
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardWinning = rewardWinning
        self.rewardLosing = rewardLosing
        self.model_network = Network(board_size)
        self.target_network = Network(board_size)
        self.state = None
        self.final_state = np.ones(board_size ** 2) * 5
        self.action = None
        self.invalid = False
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.gamma = gamma
        self.fixed_batch = fixed_batch
        self.eps_greedy_value = 1.
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.double_q_interval = double_q_interval
        self.double_q_counter = 0
        self.winner = False

    def get_random_valid_move(self, state: np.array) -> int:
        self.invalid = False
        validMoves = np.flatnonzero(state == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def get_move(self, state: np.array) -> int:
        self.state = state.copy()
        if np.random.rand() > self.eps_greedy_value:
            self.action = self.model_network.get_action(self.state)
            return self.action
        else:
            return self.get_random_valid_move(state)

    def update_eps(self, iteration: int):
        self.eps_greedy_value = self.eps_min + (1 - self.eps_min) * np.exp(- self.eps_decay * iteration)

    def invalidMove(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardInvalidMove, done=True)
        self.train_model_network()

    def train_model_network(self):
        if self.replayMemory.size < self.replayMemory.sampleSize:
            return
        for i in range(2):
            self.model_network.update_weights(self.replayMemory.get_sample(), self.gamma, self.target_network)
        self.double_q_counter += 1

        if self.double_q_interval == 0:
            return
        if self.double_q_counter % self.double_q_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.take_weights(self.model_network)

    # def get_trained_player(self, id_number: int) -> AIPlayer:
    #     trained_network = Network(self.boardsize, self.model_network.hidden,
    #                               self.model_network.only_valid_actions, self.model_network.softmax)
    #     trained_network.take_weights(self.model_network)
    #     return AIPlayer(id_number, self.boardsize, trained_network)

    def win(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardWinning, done=True)
        self.train_model_network()
        self.winner = True

    def lose(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardLosing, done=True)
        self.train_model_network()
        self.winner = False

    def add_record(self, next_game_state: np.array, done: bool):
        self.replayMemory.add_record(self.state, self.action, next_game_state, reward=0, done=done)

    def __str__(self):
        return "AI player [id: "+str(self.id)+"]"