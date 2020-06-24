import numpy as np
import abc


class Player:
    id: int  # player identifier
    boardsize: int  # grid size

    def __init__(self, board_size: int):
        self.id = None
        self.boardSize = board_size
        self.winner = False

    @abc.abstractmethod
    def get_move(self, state: np.array):
        pass

    def invalidMove(self):
        pass

    def add_record(self, next_game_state: np.array, done: bool) -> None:
        pass

    def endGameReward(self, win: bool):
        pass

    def update_eps(self, i: int):
        pass

    def train_model_network(self):
        pass

    def update_target_network(self):
        pass

    def win(self):
        self.winner = True

    def lose(self):
        self.winner = False