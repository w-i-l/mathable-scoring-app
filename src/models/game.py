from .game_model import GameTurn, GameModel
Player = GameTurn.Player

from .data_loader import DataLoader

class Game:
    def __init__(self):
        self.board = [[None for _ in range(14)] for _ in range(14)]
        self.board[6][6] = 1
        self.board[6][7] = 2
        self.board[7][6] = 3
        self.board[7][7] = 4

        self.bonus_positions = self.__get_bonus_positions()
        self.constraints_positions = self.__get_constraints_positions()
        self.scores = []

        self.__player = Player.PLAYER1
        self.current_turn = 1
        self.__starting_position = 1
        self.__score = 0
    

    def play(self, position, value):
        position = self.__convert_position_to_coordinates(position)
        self.board[position[0]][position[1]] = value
        self.__score += self.__get_score(position)
        self.current_turn += 1

    
    def change_turn(self):
        self.scores.append((self.__player, self.__starting_position, self.__score))
        self.__player = Player.PLAYER1 if self.__player == Player.PLAYER2 else Player.PLAYER2
        self.__score = 0
        self.__starting_position = self.current_turn


    def __get_score(self, position):
        row, col = position
        score = 0
        
        # Define direction offsets: top, right, bottom, left
        directions = [
            ((-2, 0), (-1, 0)),  # top: check 2 positions above
            ((0, 2), (0, 1)),    # right: check 2 positions to right
            ((2, 0), (1, 0)),    # bottom: check 2 positions below
            ((0, -2), (0, -1))   # left: check 2 positions to left
        ]
        
        for (d2_row, d2_col), (d1_row, d1_col) in directions:
            # Calculate positions to check
            pos1_row, pos1_col = row + d1_row, col + d1_col
            pos2_row, pos2_col = row + d2_row, col + d2_col
            
            # Check bounds and if positions are occupied
            if (0 <= pos2_row < 14 and 0 <= pos2_col < 14 and
                self.board[pos1_row][pos1_col] is not None and
                self.board[pos2_row][pos2_col] is not None):
                
                # Get operation and constraint
                operation = self.__get_operation_for(
                    self.board[pos1_row][pos1_col],
                    self.board[pos2_row][pos2_col],
                    self.board[row][col]
                )
                
                if operation is not None:
                    constraint = self.__get_constraint_for(self.__convert_coordinates_to_position((row, col)))
                    if constraint is None or constraint == operation:
                        score += self.board[row][col]

        score *= self.__get_bonus_for(self.__convert_coordinates_to_position((row, col)))
        return score
                    


    def __get_bonus_for(self, position):
        if position in self.bonus_positions:
            return self.bonus_positions[position]
        return 1


    def __get_constraint_for(self, position):
        if position in self.constraints_positions:
            return self.constraints_positions[position]
        return None
    

    def __get_operation_for(self, elem1, elem2, result):
        if elem1 + elem2 == result:
            return "+"
        
        if abs(elem1 - elem2) == result:
            return "-"
        
        if elem1 * elem2 == result:
            return "x"
        
        if elem1 == 0 and elem2 == 0:
            return None
        elif elem1 == 0 and result == 0:
            return "/"
        elif elem2 == 0 and result == 0:
            return "/"
        elif elem1 != 0 and elem2 != 0 and (elem1 // elem2 == result or elem2 // elem1 == result):
            return "/"
        
        return None


    def __convert_position_to_coordinates(self, position):
        return (int(position[:-1]) - 1, ord(position[-1]) - ord('A'))
    

    def __convert_coordinates_to_position(self, coordinates):
        return f"{coordinates[0]+1}{chr(coordinates[1] + ord('A'))}"
    

    def print_board(self):
        for row in self.board:
            for col in row:
                if col is None:
                    print(" _", end=" ")
                else:
                    print(f"{col:>2}", end=" ")
            print()


    def __get_bonus_positions(self):
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
    

    def __get_constraints_positions(self):
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

    