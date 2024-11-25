from models.data_loader import DataLoader
from models.game_model import GameModel, GameMove
from models.game import Game
from utils.image_processing import ImageProcessing
from utils.helper_functions import format_path
from tqdm import tqdm
import cv2 as cv
from multiprocessing import Pool
from cnn.cnn_model import CNNModel
from pipelines.base_pipeline import BasePipeline
import os

class TestingPipeline(BasePipeline):
    '''
    Pipeline for testing the moves of a game using a CNN model
    '''

    def __init__(self, moves_path: str, scores_path: str, turns_path: str):
        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        self.game = GameModel(moves, turns_path, scores_path)
        self.image_processing = ImageProcessing()


    def test_game(self, title="Processing moves"):
        '''
        Tests the moves of a game using a CNN model and prints the results.
        '''
        predicted = 0

        model = CNNModel()
        model.load("../models/cnn_model")

        for i in tqdm(range(1, len(self.game.moves)), desc=title):
            board_contor_1 = self.image_processing._find_board_contour(self.game.moves[i-1].image_path)
            board_contor_2 = self.image_processing._find_board_contour(self.game.moves[i].image_path)

            board_1 = self.image_processing.crop_board(self.game.moves[i-1].image_path, board_contor_1)
            board_2 = self.image_processing.crop_board(self.game.moves[i].image_path, board_contor_2)

            diff = self.image_processing.find_difference_between_images(board_1, board_2)
            
            # get original dimensions before any resizing
            original_height, original_width = diff.shape[:2]

            x, y = self.image_processing.find_added_piece_coordinates(diff)
            w = 105
            h = 105
            
            # scale coordinates for board_2
            scale_x = board_2.shape[1] / original_width
            scale_y = board_2.shape[0] / original_height
            
            added_piece_contour = [
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + w) * scale_x), 
                int((y + h) * scale_y)
            ]
            
            # computing the grid rectangles
            board_2, grid_rectangles = self.image_processing.split_board_in_blocks(board_2)
            
            # find the matching block
            matching_block_idx, overlap_percentage = self.find_matching_block(added_piece_contour, grid_rectangles)
            
            if matching_block_idx != -1:
                expected_move = self.game.moves[i].move
                predicted_move = self.convert_index_to_coordinates(matching_block_idx)

                added_piece = board_2[added_piece_contour[1]:added_piece_contour[3], added_piece_contour[0]:added_piece_contour[2]]
                added_piece = cv.resize(added_piece, (105, 105))
                predicted_value = model.predict(added_piece)
                expected_value = self.game.moves[i].value

                if predicted_move == expected_move and predicted_value == expected_value:
                    predicted += 1
                elif expected_move != predicted_move:
                    print(f"Expected: {expected_move} -- Actual: {predicted_move} -- Overlap: {overlap_percentage:.2f}%")
                elif expected_value != predicted_value:
                    print(f"Expected: {expected_value} -- Predicted: {predicted_value}")

        print(f"Predicted: {predicted}/{len(self.game.moves)-1} ({predicted/(len(self.game.moves)-1)*100:.2f}%)")
        return predicted / (len(self.game.moves)-1)


if __name__ == "__main__":
    game_number = 1
    moves_path = f"../data/train/game_{game_number}"
    scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
    turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

    pipeline = TestingPipeline(moves_path, scores_path, turns_path)
    # pipeline.test_game(debug=True)
    # pipeline.test_game_parallel()

