import os
from util import format_path
from game_model import GameMove

class DataLoader():
    def __init__(self, folder_path):
        self.folder_path = format_path(folder_path)


    def load_moves(self) -> list[GameMove]:
        files = os.listdir(self.folder_path)
        valid_files = [file for file in files if self.__is_valid_file(file)]
        moves = []

        # read the empty board image
        empty_board_path = "../data/empty_board.jpg"
        empty_board_move = GameMove(empty_board_path, "empty")
        moves.append(empty_board_move)

        images = [file for file in valid_files if file.endswith('.jpg')]
        positions = [file for file in valid_files if file.endswith('.txt')]
        images.sort()
        positions.sort()

        for image, position_path in zip(images, positions):
            image_path = os.path.join(self.folder_path, image)
            with open(os.path.join(self.folder_path, position_path), 'r') as file:
                position, value = file.read().strip().split(' ')
                value = int(value)
            move = GameMove(image_path, position, value)
            moves.append(move)

        return moves

    
    def __is_valid_file(self, file):
        if os.path.isdir(file):
            return False
        
        if file.endswith("scores.txt") or file.endswith("turns.txt"):
            return False

        return (file.endswith('.jpg') or file.endswith('.txt'))
    

if __name__ == "__main__":
    loader = DataLoader("../data/train/game_1")
    moves = loader.load_moves()
    print(*moves, sep='\n')