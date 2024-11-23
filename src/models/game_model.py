from enum import Enum
from utils.helper_functions import format_path

class GameMove:
    
    def __init__(self, image_path, position, value=None):
        self.image_path = image_path
        self.move = position
        self.value = value


    def __repr__(self):
        return f"GameMove(image_path: {self.image_path}, move: {self.move}, value: {self.value})"


class GameTurn:

    class Player(Enum):
        PLAYER1 = 1
        PLAYER2 = 2

        def __str__(self):
            return str(self.value)
        

    def __init__(self, player: Player, starting_position, score=None):
        self.player = player
        self.starting_position = starting_position
        self.score = score
        
    
    def __repr__(self):
        return f"GameTurn(player: {self.player}, statring_position: {self.starting_position}, score: {self.score})"


class GameModel:

    def __init__(self, moves: list[GameMove], turns_path, scores_path=None):
        self.moves = moves
        self._scores_path = format_path(scores_path)
        self._turns_path = format_path(turns_path)
        self.game_turns = self.__zip_scores_and_turns()


    @staticmethod
    def available_pieces():
        return [
            # 0-9 (10 pieces)
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            
            # 10-19 (10 pieces)
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            
            # 20-29 (6 pieces)
            20, 21, 24, 25, 27, 28,
            
            # 30-39 (4 pieces)
            30, 32, 35, 36,
            
            # 40-49 (5 pieces)
            40, 42, 45, 48, 49,
            
            # 50-59 (3 pieces)
            50, 54, 56,
            
            # 60-69 (3 pieces)
            60, 63, 64,
            
            # 70-79 (2 pieces)
            70, 72,
            
            # 80-89 (2 pieces)
            80, 81,
            
            # 90-99 (1 piece)
            90
        ]


    def __zip_scores_and_turns(self):
        game_turns = []
        turns = self.__load_turns(self._turns_path)
        if self._scores_path == None:
            player = GameTurn.Player.PLAYER1 if len(game_turns) % 2 == 0 else GameTurn.Player.PLAYER2
            for turn in turns:
                game_turn = GameTurn(player, turn)
                game_turns.append(game_turn)
            return game_turns

        scores = self.__load_scores(self._scores_path)
        
        for score, turn in zip(scores, turns):
            player = GameTurn.Player.PLAYER1 if len(game_turns) % 2 == 0 else GameTurn.Player.PLAYER2
            game_turn = GameTurn(player, turn, score)
            game_turns.append(game_turn)

        return game_turns


    def __load_scores(self, scores_path):
        with open(scores_path, 'r') as file:
            lines = file.readlines()
            scores = []
            for line in lines:
                player, position, score = line.split(' ')
                score = score.strip()
                scores.append(score)
            return scores


    def __load_turns(self, turns_path):
        with open(turns_path, 'r') as file:
            lines = file.readlines()
            turns = []
            for line in lines:
                player, position = line.split(' ')
                position = position.strip()

                turns.append(position)
            return turns


if __name__ == "__main__":
    moves = [GameMove("image1.jpg", "e4"), GameMove("image2.jpg", "e5")]
    game = GameModel(moves, "../data/train/game_1/1_turns.txt", "../data/train/game_1/1_scores.txt")
    print(game.game_turns)