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
        if self.vectorBoard.zerowins(): return 0
        if self.vectorBoard.twowins(): return 1
        return -1

    def zerowins(self) -> bool:
        """Returns True if player 0 (symbol 1) won"""
        return self.search_winning(np.ones(3).astype(self.vectorBoard))

    def onewins(self)-> bool:
        """Returns True if player 1 (symbol -1) won"""
        return self.search_winning(np.ones(3).astype(self.vectorBoard) * (-1))


    def search_winning(self, input: np.array) -> bool:
        v = self.vectorBoard.reshape([self.size, self.size])
        rows = any([v[i].equal(input).all() for i in range(self.size)])
        cols = any([v[:, i].equal(input).all() for i in range(self.size)])
        diag = np.diagonal(v).equal(input)
        other_diag = np.fliplr(v).diagonal().equal(input).all()
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

