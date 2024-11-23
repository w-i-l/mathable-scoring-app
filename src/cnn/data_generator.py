from models.game_model import GameModel
from models.data_loader import DataLoader
from utils.image_processing import ImageProcessing
from models.game_model import GameModel
from utils.helper_functions import format_path
from tqdm import tqdm
import cv2 as cv
import os

class DataGenerator:
    '''
    Data generator for the CNN model from the images provided as training data
    '''

    def __init__(self):
        self.image_processing = ImageProcessing()
        self.game = None
        self.__generate_folders()

    
    def __generate_folders(self):
        '''
        Generates the folders for the training and validation data
        '''

        for i in GameModel.available_pieces():
            folder_path = f"../data/cnn/train/{i}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path = f"../data/cnn/validation/{i}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


    def generate_data(self, moves_path: str, scores_path: str, turns_path: str, is_validation=False, game_number=1):
        '''
        Generates the training or validation data for the CNN model from the images provided in the moves_path folder.
        It crops the board from the images, finds the added piece, crops it and saves it in the corresponding folder, 
        using as label the ground truth value of the piece.

        Parameters:
        -----------
        moves_path (str): The path to the folder containing the images and the positions files.
        scores_path (str): The path to the scores file.
        turns_path (str): The path to the turns file.
        is_validation (bool): If True, the data will be saved in the validation folder.
        game_number (int): The number of the game.
        '''

        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        self.game = GameModel(moves, turns_path, scores_path)

        for i in tqdm(range(1, len(self.game.moves)), desc=f"Game {game_number}"):
            # find the board contours
            board_contor_1 = self.image_processing._find_board_contour(self.game.moves[i-1].image_path)
            board_contor_2 = self.image_processing._find_board_contour(self.game.moves[i].image_path)

            # crop the board
            board_1 = self.image_processing.crop_board(self.game.moves[i-1].image_path, board_contor_1)
            board_2 = self.image_processing.crop_board(self.game.moves[i].image_path, board_contor_2)

            # find the difference between the two boards
            diff = self.image_processing.find_difference_between_images(board_1, board_2)
            
            # get original dimensions before any resizing
            original_height, original_width = diff.shape[:2]

            # find the added piece coordinates
            x, y = self.image_processing.find_added_piece_coordinates(diff)
            w = 105
            h = 105
            
            # scale coordinates for board_2
            scale_x = board_2.shape[1] / original_width
            scale_y = board_2.shape[0] / original_height
            
            # compute the added piece contour relative to the board_2
            added_piece_contour = [
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + w) * scale_x), 
                int((y + h) * scale_y)
            ]
            
            # computing the grid rectangles
            board_2, grid_rectangles = self.image_processing.split_board_in_blocks(board_2)
            
            # extracting the added piece
            added_piece = board_2[added_piece_contour[1]:added_piece_contour[3], added_piece_contour[0]:added_piece_contour[2]]
            added_piece = cv.resize(added_piece, (105, 105))
            
            # save the added piece
            subfolder = "validation" if is_validation else "train"
            game_number = game_number if not is_validation else "validation"
            piece_path = f"../data/cnn/{subfolder}/{self.game.moves[i].value}/{game_number}_{i}.png"
            cv.imwrite(piece_path, added_piece)


    def move_from_folder(self, folder_path: str, name="template"):
        '''
        Moves the files from a folder to the training folder, renaming them with the given name.
        It is used to move the template pieces to the training folder.
        '''
        
        folder_path = format_path(folder_path)
        for file in os.listdir(folder_path):
            value = file.split('.')[0]
            source_path = os.path.join(folder_path, file)
            destination_path = f"../data/cnn/train/{value}/{name}_{file}"
            os.rename(source_path, destination_path)       


if __name__ == "__main__":
    generator = DataGenerator()
    # moves_path = "../data/validation"
    # scores_path = "../data/validation/1_scores.txt"
    # turns_path = "../data/validation/1_turns.txt"
    # generator.generate_data(moves_path, scores_path, turns_path, is_validation=True)

    # for game_number in range(1, 5):
    #     moves_path = f"../data/train/game_{game_number}"
    #     scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
    #     turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"
    #     generator.generate_data(moves_path, scores_path, turns_path)

    # generator.move_from_folder("../data/templates/board", name="template_board")
    # generator.move_from_folder("../data/templates/togheter", name="template_togheter")