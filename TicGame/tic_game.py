from TicGame.board import Board

class TicGame:

    def __init__(self, players: list, board_size: int = 4):
        self.boardSize = board_size
        self.board = Board(board_size)
        self.players = players
        self.players[0].id = 1
        self.players[1].id = -1
        self.winner = False

    def is_valid(self, idx: int) -> bool:
        return self.board.vectorBoard[idx] == 0

    def play(self):
        currentPlayer = self.players[0]
        otherPlayer = self.players[1]
        turn = 0

        self.board.print_board()

        while turn < self.boardSize**2:

            move = currentPlayer.get_move(self.board.vectorBoard)

            if not self.is_valid(move):
                return False

            self.board.set_board(move, currentPlayer)
            winner = self.board.winner()    # Returns 0 if player 0 wins, 1 if Player 1 wins and -1 if No one won

            if winner > - 0.1:  # if it is 0 or 1
                self.players[winner].win()
                self.players[(winner + 1) % 2].lose()
                self.winner = True
                self.board.print_board()
                return True

            turn += 1
            currentPlayer = self.players[turn % 2]
            otherPlayer = self.players[(turn + 1) % 2]

            self.board.print_board()

        return True


    def reset(self):
        self.board.reset()
        self.winner = None