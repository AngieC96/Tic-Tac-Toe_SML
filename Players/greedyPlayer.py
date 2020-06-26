from Players.player import Player
import numpy as np


class GreedyPlayer(Player):

    def __init__(self, boardsize: int):
        super().__init__(boardsize)

    def get_move(self, board: np.array) -> int:
        v = board.reshape([self.boardSize, self.boardSize])
        for i in range(self.boardSize):
            if np.sum(v[i]) == self.id * 2:
                return i * self.boardSize + np.flatnonzero(v[i] == 0)[0]
            if np.sum(v[:, i]) == self.id * 2:
                return i + self.boardSize * np.flatnonzero(v[:, i] == 0)[0]
            if np.sum(np.diagonal(v)) == self.id * 2:
                return 4 * np.flatnonzero(np.diagonal(v) == 0)[0]
            if np.sum(np.fliplr(v).diagonal()) == self.id * 2:
                return 2 + 2 * np.flatnonzero(np.fliplr(v).diagonal() == 0)[0]

        validMoves = np.flatnonzero(board == 0)
        return np.random.choice(validMoves)

    def __str__(self):
        return "Random player"+str(self.id)