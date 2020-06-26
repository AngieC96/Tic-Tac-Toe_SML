import numpy as np


class Board:
    vectorBoard: np.ndarray
    rows: int
    cols: int

    def __init__(self, N: int):
        self.vectorBoard = np.zeros(N**2).astype(int)
        self.size = N

    def set_board(self, idx: int, player):
        self.vectorBoard[idx] = player.id

    def winner(self) -> int:
        """Returns 0 if player 0 wins, 1 if Player 1 wins and -1 if No one won"""
        if self.player_0_wins():
            return 0
        if self.player_1_wins():
            return 1
        return -1

    def player_0_wins(self) -> bool:
        """Returns True if player 0 (symbol 1) won"""
        return self.search_winning(np.ones(3) * (-1.))

    def player_1_wins(self)-> bool:
        """Returns True if player 1 (symbol -1) won"""
        return self.search_winning(np.ones(3)*(-1))


    def search_winning(self, input: np.array) -> bool:
        v = self.vectorBoard.reshape([self.size, self.size])
        rows = any([np.equal(v[i], input).all() for i in range(self.size)])
        cols = any([np.equal(v[:, i], input).all() for i in range(self.size)])
        diag = np.equal(np.diagonal(v), input).all()
        other_diag = np.equal(np.fliplr(v).diagonal(), input).all()
        return rows or cols or diag or other_diag

    def print_board(self) -> str:
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                box = self.vectorBoard[i * self.size + j]
                if box == 0:
                    row += " *"
                if box == 1:
                    row += " x"
                if box == -1:
                    row += " o"
            print(row)
        print("\n")

    def reset(self):
        self.vectorBoard = np.zeros(self.size**2).astype(int)

