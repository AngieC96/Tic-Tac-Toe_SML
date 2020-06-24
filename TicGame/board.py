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

    def win_condition(self) -> int:
        pass

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

