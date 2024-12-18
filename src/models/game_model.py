from enum import Enum
from utils.helper_functions import format_path

class GameMove:
    '''
    A class which represents a move in the game.
    '''

    def __init__(self, image_path: str, position: str, value=None):
        self.image_path = image_path
        self.move = position
        self.value = value


    def __repr__(self):
        return f"GameMove(image_path: {self.image_path}, move: {self.move}, value: {self.value})"


class GameTurn:
    '''
    A class which represents a turn in the game.
    '''

    class Player(Enum):
        '''
        An enum class which represents the players in the game.
        '''
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
    '''
    A class which represents all states of the game.
    '''

    def __init__(self, moves: list[GameMove], turns_path: str, scores_path=None):
        self.moves = moves
        self._scores_path = format_path(scores_path)
        self._turns_path = format_path(turns_path)
        self.game_turns = self.__zip_scores_and_turns()


    @staticmethod
    def available_pieces() -> list[int]:
        '''
        Returns a list of the available pieces in the game.
        '''
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


    def __zip_scores_and_turns(self) -> list[GameTurn]:
        '''
        Zips the scores and turns into a list of GameTurn objects.
        '''

        game_turns = []
        turns = self.__load_turns(self._turns_path)
        if self._scores_path == None:
            player = self.__get_first_player()
            for turn in turns:
                game_turn = GameTurn(player, turn)
                game_turns.append(game_turn)
                player = GameTurn.Player.PLAYER1 if player == GameTurn.Player.PLAYER2 else GameTurn.Player.PLAYER2
            return game_turns

        if self._scores_path == None:
            scores = [None] * len(turns)
        else:
            scores = self.__load_scores(self._scores_path)
        
        player = self.__get_first_player()
        for score, turn in zip(scores, turns):
            game_turn = GameTurn(player, turn, score)
            game_turns.append(game_turn)
            player = GameTurn.Player.PLAYER1 if player == GameTurn.Player.PLAYER2 else GameTurn.Player.PLAYER2

        return game_turns


    def __load_scores(self, scores_path: str) -> list[str]:
        '''
        Loads and returns the scores from the scores file.
        '''

        with open(scores_path, 'r') as file:
            lines = file.readlines()
            scores = []
            for line in lines:
                player, position, score = line.split(' ')
                score = score.strip()
                scores.append(score)
            return scores


    def __load_turns(self, turns_path: str) -> list[str]:
        '''
        Loads and returns the turns from the turns file.
        '''

        with open(turns_path, 'r') as file:
            lines = file.readlines()
            turns = []
            for line in lines:
                player, position = line.split(' ')
                position = position.strip()

                turns.append(position)
            return turns
        

    def __get_first_player(self) -> GameTurn.Player:
        '''
        Returns the first player in the game.
        '''

        with open(self._turns_path, 'r') as file:
            line = file.readline()
            player, _ = line.split(' ')
            return GameTurn.Player.PLAYER1 if player == 'Player1' else GameTurn.Player.PLAYER2
            


if __name__ == "__main__":
    moves = [GameMove("image1.jpg", "e4"), GameMove("image2.jpg", "e5")]
    game = GameModel(moves, "../data/train/game_1/1_turns.txt", "../data/train/game_1/1_scores.txt")
    print(game.game_turns)