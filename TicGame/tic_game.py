from TicGame.board import Board

class TicGame:

    def __init__(self, players: list, board_size: int = 4):
        self.boardSize = board_size
        self.board = Board(board_size)
        self.players = players
        self.players[0].id = 1
        self.players[1].id = -1

    def is_valid(self, idx: int) -> bool:
        return self.board.vectorBoard[idx] == 0

    def play(self):
        currentPlayer = self.players[0]
        otherPlayer = self.players[1]
        turn = 0

        self.board.print_board()

        while turn < self.boardSize**2:

            move = currentPlayer.get_move(self.board.vectorBoard)

            #while not self.is_valid(move):
            #    currentPlayer.invalidMove()
            #    move = currentPlayer.get_move(self.board.vectorBoard)
                # print("Invalid Move")

            self.board.set_board(move, currentPlayer)
            turn += 1

            currentPlayer = self.players[turn % 2]
            otherPlayer = self.players[(turn + 1) % 2]

            self.board.print_board()
        self.board.reset()

        print("Winner : ")