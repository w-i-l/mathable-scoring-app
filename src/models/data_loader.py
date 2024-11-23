import os
from utils.helper_functions import format_path
from models.game_model import GameMove

class DataLoader:
    '''
    A class which provides functionality to load moves from a folder.
    '''

    def __init__(self, folder_path: str):
        self.folder_path = format_path(folder_path)


    def load_moves(self) -> list[GameMove]:
        '''
        Loads the moves from the folder path and returns a list of GameMove objects.
        '''

        files = os.listdir(self.folder_path)
        valid_files = [file for file in files if self.__is_valid_file(file)]
        moves = []

        # read the empty board image and add it as the first move
        empty_board_path = "../data/empty_board.jpg"
        empty_board_move = GameMove(empty_board_path, "empty")
        moves.append(empty_board_move)

        images = [file for file in valid_files if file.endswith('.jpg')]
        positions = [file for file in valid_files if file.endswith('.txt')]

        # sort the files to match the order of the moves
        images.sort()
        positions.sort()

        for image, position_path in zip(images, positions):
            image_path = os.path.join(self.folder_path, image)

            # read the ground truth from the file
            with open(os.path.join(self.folder_path, position_path), 'r') as file:
                position, value = file.read().strip().split(' ')
                value = int(value)

            move = GameMove(image_path, position, value)
            moves.append(move)

        return moves

    
    def __is_valid_file(self, file: str) -> bool:
        '''
        Checks if the file is a valid image or position file.
        '''

        if os.path.isdir(file):
            return False
        
        # ignore the scores and turns files
        if file.endswith("scores.txt") or file.endswith("turns.txt"):
            return False

        return (file.endswith('.jpg') or file.endswith('.txt'))


if __name__ == "__main__":
    loader = DataLoader("../data/train/game_1")
    moves = loader.load_moves()
    print(*moves, sep='\n')