from .data_loader import DataLoader
from .game_model import GameTurn, GameModel
Player = GameTurn.Player


class Game:
    '''
    Game class that simulates the Mathable board game.
    '''

    def __init__(self):
        self.board = [[None for _ in range(14)] for _ in range(14)]
        # Set initial values
        self.board[6][6] = 1
        self.board[6][7] = 2
        self.board[7][6] = 3
        self.board[7][7] = 4

        self.bonus_positions = self.__get_bonus_positions()
        self.constraints_positions = self.__get_constraints_positions()
        self.scores = []

        self.__player = Player.PLAYER1
        self.current_turn = 1
        self.__starting_position = 1 # The starting position of the current player
        self.__score = 0
    

    def play(self, position: str, value: int):
        '''
        Plays a move on the board.
        This method updates the board, increases the score and advances the turn.

        Parameters:
        -----------
        position (str): The position to play the move. The position is a string with the row number and the column letter.
        value (int): The value of the piece to play on the board.
        '''

        position = self.__convert_position_to_coordinates(position)
        self.board[position[0]][position[1]] = value
        self.__score += self.__get_score(position)
        self.current_turn += 1

    
    def change_turn(self):
        '''
        Changes the turn of the game.
        '''

        self.scores.append((self.__player, self.__starting_position, self.__score))
        self.__player = Player.PLAYER1 if self.__player == Player.PLAYER2 else Player.PLAYER2
        self.__score = 0
        self.__starting_position = self.current_turn


    def __get_score(self, position: tuple[int, int]) -> int:
        '''
        Calculates the score for a given position on the board.

        A score is calculated by first checking the 4 directions (top, right, bottom, left) of the position
        for a valid operation between current position and the two positions in the direction.

        If a valid operation is found, the score is calculated by adding the value of the current position to the score.
        This can be done for each direction.

        The score is then multiplied by the bonus of the position, if any.

        Parameters:
        -----------
        position (tuple[int, int]): The position on the board to calculate the score for.

        Returns:
        --------
        int: The score for the given position.
        '''

        row, col = position
        score = 0
        
        # define direction offsets: top, right, bottom, left
        directions = [
            ((-2, 0), (-1, 0)),  # top: check 2 positions above
            ((0, 2), (0, 1)),    # right: check 2 positions to right
            ((2, 0), (1, 0)),    # bottom: check 2 positions below
            ((0, -2), (0, -1))   # left: check 2 positions to left
        ]
        
        for (d2_row, d2_col), (d1_row, d1_col) in directions:
            # calculate positions to check
            pos1_row, pos1_col = row + d1_row, col + d1_col
            pos2_row, pos2_col = row + d2_row, col + d2_col
            
            # check bounds and if positions are occupied
            if (0 <= pos2_row < 14 and 0 <= pos2_col < 14 and
                self.board[pos1_row][pos1_col] is not None and
                self.board[pos2_row][pos2_col] is not None):
                
                # get operation and constraint
                operation = self.__get_operation_for(
                    self.board[pos1_row][pos1_col],
                    self.board[pos2_row][pos2_col],
                    self.board[row][col]
                )
                
                # check if operation is valid
                if operation is not None:
                    # check if there is a constraint
                    constraint = self.__get_constraint_for(self.__convert_coordinates_to_position((row, col)))
                    # if there is no constraint or the constraint is the same as the operation
                    if constraint is None or constraint == operation:
                        score += self.board[row][col]

        # multiply score by bonus
        position_str = self.__convert_coordinates_to_position((row, col))
        score *= self.__get_bonus_for(position_str)
        return score
                    

    def __get_bonus_for(self, position: str) -> int:
        '''
        Returns the bonus for a given position on the board.

        Returns:
        --------
        int: The bonus for the given position. If no bonus is found, 1 is returned.
        '''

        if position in self.bonus_positions:
            return self.bonus_positions[position]
        return 1


    def __get_constraint_for(self, position):
        '''
        Returns the constraint for a given position on the board.

        Returns:
        --------
        str|None: The constraint for the given position. If no constraint is found, None is returned.
        '''

        if position in self.constraints_positions:
            return self.constraints_positions[position]
        return None
    

    def __get_operation_for(self, elem1: int, elem2: int, result: int) -> str|None:
        '''
        Returns the operation that can be applied to elem1 and elem2 to get the result.

        Returns:
        --------
        str|None: The operation that can be applied to elem1 and elem2 to get the result.
        If no operation is found, None is returned.
        '''

        if elem1 + elem2 == result:
            return "+"
        
        if abs(elem1 - elem2) == result:
            return "-"
        
        if elem1 * elem2 == result:
            return "x"
        
        # TODO: check for division by zero
        if elem1 == 0 and elem2 == 0: # 0 / 0 - impossible
            return None
        elif elem1 == 0 and result == 0: # 0 / elem2 = 0
            return "/"
        elif elem2 == 0 and result == 0: # 0 / elem1 = 0
            return "/"
        elif elem1 != 0 and elem2 != 0 and (elem1 // elem2 == result or elem2 // elem1 == result):
            return "/"
        
        return None


    def __convert_position_to_coordinates(self, position: str) -> tuple[int, int]:
        '''
        Returns the coordinates of a given position on the board as a tuple of integers.
        '''
        return (int(position[:-1]) - 1, ord(position[-1]) - ord('A'))
    

    def __convert_coordinates_to_position(self, coordinates: tuple[int, int]) -> str:
        '''
        Returns the position of a given coordinates on the board as a string.
        '''
        return f"{coordinates[0]+1}{chr(coordinates[1] + ord('A'))}"
    

    def print_board(self):
        '''
        Prints the board to the console as a grid.
        '''

        for row in self.board:
            for col in row:
                if col is None:
                    print(" _", end=" ")
                else:
                    print(f"{col:>2}", end=" ")
            print()


    def __get_bonus_positions(self) -> dict[str, int]:
        return {
            # 2x positions (purple squares)
            "2B": 2,
            "3C": 2,
            "4D": 2,
            "5E": 2,
            "2M": 2,
            "3L": 2,
            "4K": 2,
            "5J": 2,
            "10E": 2,
            "11D": 2,
            "12C": 2,
            "13B": 2,
            "10J": 2,
            "11K": 2,
            "12L": 2,
            "13M": 2,

            # 3x positions (orange squares)
            "1A": 3,
            "1G": 3,
            "1H": 3,
            "1N": 3,
            "7A": 3,
            "8A": 3,
            "14A": 3,
            "7N": 3,
            "8N": 3,
            "14G": 3,
            "14H": 3,
            "14N": 3,
        }
    

    def __get_constraints_positions(self) -> dict[str, str]:
        return {
            # top
            "2E": "/",
            "3F": "-",
            "4G": "+",
            "4H": "x",
            "5G": "x",
            "5H": "+",
            "3I": "-",
            "2J": "/",

            # right
            "5M": "/",
            "6L": "-",
            "7K": "+",
            "7J": "x",
            "8K": "x",
            "8J": "+",
            "9L": "-",
            "10M": "/",

            # bottom
            "10G": "+",
            "10H": "x",
            "11G": "x",
            "11H": "+",
            "12F": "-",
            "13E": "/",
            "12I": "-",
            "13J": "/",

            # left
            "5B": "/",
            "6C": "-",
            "7D": "x",
            "7E": "+",
            "8D": "+",
            "8E": "x",
            "9C": "-",
            "10B": "/",
        }


if __name__ == "__main__":
    game = Game()
    game_number = 1
    # moves = DataLoader(f"../data/train/game_{game_number}").load_moves()
    # turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"
    # scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"

    moves = DataLoader("../data/validation").load_moves()
    turns_path = "../data/validation/1_turns.txt"
    scores_path = "../data/validation/1_scores.txt"

    game_model = GameModel(moves, turns_path, scores_path)

    for move in moves[1:]:
        game.play(move.move, move.value)
    
        for turn in game_model.game_turns:
            if int(turn.starting_position) == game.current_turn:
                game.change_turn()
                break
    game.change_turn()
    
    # print(*game.scores, sep="\n")
    for turn, score in zip(game_model.game_turns, game.scores):
        print(f"{turn.player:>3} :", end=" ")
        print(f"{turn.score:>3} - ", end=" ")
        print(score[2], end=" - ")
        print("diff:", int(turn.score) - score[2])

    