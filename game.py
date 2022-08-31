import numpy as np
import numpy.random as random
import os

class Action:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    MOVES = {'w': UP, 'd': RIGHT, 's': DOWN, 'a': LEFT}

    @classmethod
    def get_action(cls, keyboard_key): # For human play
        # wtf python, static references?
        if keyboard_key not in Action.MOVES:
            raise Exception
        return Action.MOVES[keyboard_key]

class Game:
    def __init__(self) -> None:
        self.board = np.zeros(shape=(4,4))
        self.generate_new_tile(2)

    def generate_new_tile(self, n=1):
        # By the rules of 2048, a 4 is generated 10% of the time.
        for _ in range(n):
            a = random.randint(0, 3)
            b = random.randint(0, 3)
            while(self.board[a][b] != 0):
                a = random.randint(0, 3)
                b = random.randint(0, 3)

            self.board[a][b] = 4 if np.random.uniform(0, 10) < 1 else 2 

            # if sum([cell for row in self.board for cell in row]) in (0, 2):
            #     self.board[a][b] = 2
            # else:
            #     self.board[a][b] = random.choice((2, 4))

    def is_done(self):
    
        # If there are at least one cell with 0, then the game is not over
        if not np.all(self.board):
            return False
        
        # If all cells are filled, we need to check if there are any possible moves
        else:
            # Check if there are any equal adjacent cells across horisontal and vertical axes
            for row in self.board:
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            for row in np.transpose(self.board):
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            # There are no equal adjacent cells, the game is over
            return True

    
    def stack(self, board):
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                k = i
                while board[k][j] == 0:
                    if k == len(board) - 1:
                        break
                    k += 1
                if k != i:
                    board[i][j], board[k][j] = board[k][j], 0

    def sum_up(self, board):
        for i in range(0, len(board) - 1):
            for j in range(0, len(board)):
                if board[i][j] != 0 and board[i][j] == board[i + 1][j]:
                    board[i][j] += board[i + 1][j]
                    board[i + 1][j] = 0


    # Slick move mechanics from https://gist.github.com/wbars/88df9704306629c40c7929e691b48b98
    def move(self, action):
        self.drop_elem()
        rotated_board = np.rot90(self.board, action)
        self.stack(rotated_board)
        self.sum_up(rotated_board)
        self.stack(rotated_board)
        self.board = np.rot90(rotated_board, len(self.board) - action) # Rotate back

    def get_score(self) -> None:
        return np.max(self.board)

    def show(self) -> None:
        print(np.array_str(self.board))

    def reset(self) -> None:
        self.__init__()

    def drop_elem(self):
        elem = random.choice([2, len(self.board)], 1, False, [0.9, 0.1])[0]
        zeroes_flatten = np.where(self.board == 0)
        zeroes_indices = [(x, y) for x, y in zip(zeroes_flatten[0], zeroes_flatten[1])]
        random_index = zeroes_indices[random.choice(len(zeroes_indices), 1)[0]]
        self.board[random_index] = elem


# board = Game()
# while True:
#     board.show()
#     board.move(Action.get_action(input(f"Score: {board.get_score()}\nNext move?\n")))
#     if board.is_done():
#         break
#     os.system('cls' if os.name == 'nt' else 'clear')