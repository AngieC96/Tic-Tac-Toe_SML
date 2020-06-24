from TicGame.board import Board

class TicTraining:

    def __init__(self, players: list, board_size: int = 3):
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

        while turn < self.boardSize**2:

            move = currentPlayer.get_move(self.board.vectorBoard)

            if not self.is_valid(move):
                currentPlayer.invalidMove()
                return False

            self.board.set_board(move, currentPlayer)
            winner = self.board.winner()    # Returns 0 if player 0 wins, 1 if Player 1 wins and -1 if No one won

            if winner > - 0.1:
                self.players[winner].win()
                self.players[(winner + 1) % 2].lose()
                self.winner = True
                return True

            if turn > 1:
                otherPlayer.add_record(self.board.vectorBoard, done=False)
                otherPlayer.train_model_network()
            turn += 1
            currentPlayer = self.players[turn % 2]
            otherPlayer = self.players[(turn + 1) % 2]

        currentPlayer.add_record(self.board.vectorBoard, done=True)
        currentPlayer.train_model_network()
        otherPlayer.add_record(self.board.vectorBoard, done=True)
        otherPlayer.train_model_network()
        return True

    def reset(self):
        self.board.reset()
        self.winner = None