# from Players.AIPlayer import AIPlayer
from Players.player import Player
from Training.network import Network
import numpy as np
from Training.replayMemory import ReplayMemory


class RandomMemory(Player):
    rewardInvalidMove: float
    rewardWinning: float
    rewardLosing: float
    state: np.array
    action: int
    replayMemory: ReplayMemory


    def __init__(self, board_size: int, rewardInvalidMove: float,
                 rewardWinning: float, rewardLosing: float, rep_memory):

        super().__init__(board_size)
        self.rewardNoScore = 0
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardWinning = rewardWinning
        self.rewardLosing = rewardLosing
        self.state = None
        self.final_state = np.ones(board_size ** 2) * 5
        self.action = None
        self.invalid = False
        self.replayMemory = rep_memory
        self.winner = False


    def get_move(self, board: np.array) -> int:
        self.state = board.copy()
        validMoves = np.flatnonzero(board == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def invalidMove(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardInvalidMove, done=True)

    def win(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardWinning, done=True)
        self.winner = True

    def lose(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardLosing, done=True)
        self.winner = False

    def add_record(self, next_game_state: np.array, done: bool):
        self.replayMemory.add_record(self.state, self.action, next_game_state.copy(), reward=0, done=done)
