from Players.player import Player
import numpy as np


class RandomPlayer(Player):

    def __init__(self, boardsize: int, randomness: float = 0):
        super().__init__(boardsize)
        self.randomness = randomness

    def get_move(self, board: np.array) -> int:
        validMoves = np.flatnonzero(board == 0)
        if np.random.rand() > 1-self.randomness:
            return np.random.choice(validMoves)
        return min(validMoves)

    def __str__(self):
        return "Random player"+str(self.id)