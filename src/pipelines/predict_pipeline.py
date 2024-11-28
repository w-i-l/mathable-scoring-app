from pipelines.base_pipeline import BasePipeline
from models.game import Game
from utils.helper_functions import format_path
from utils.image_processing import ImageProcessing
from tqdm import tqdm
import os
import cv2 as cv

from models.game_model import GameModel
from models.data_loader import DataLoader
from cnn.cnn_model import CNNModel

class PredictPipeline(BasePipeline):
    '''
    Pipeline for predicting the moves of a game using a CNN model
    '''

    def __init__(self, moves_path: str, turns_path: str):
        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        self.game = GameModel(moves, turns_path)
        self.image_processing = ImageProcessing()
    
    def play_game(self, game_model: GameModel, model: CNNModel, game_number=1):
        '''
        Plays the game using the CNN model to predict the moves and saves the results in the output folder.
        '''

        game = Game()
        moves = game_model.moves

        if not os.path.exists(format_path(f"../data/output/game_{game_number}")):
            os.makedirs(format_path(f"../data/output/game_{game_number}"))

        for i in tqdm(range(1, len(moves)), desc=f"Playing game {game_number}"):
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
            
            # compute the added piece contour relative to the board_2
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
            
            # if a matching block is found, predict the move
            if matching_block_idx != -1:
                predicted_move = self.convert_index_to_coordinates(matching_block_idx)

                added_piece = board_2[added_piece_contour[1]:added_piece_contour[3], added_piece_contour[0]:added_piece_contour[2]]
                added_piece = cv.resize(added_piece, (105, 105))
                predicted_value = model.predict(added_piece)

                # format the index
                if i < 10:
                    index = f"0{i}"
                else:
                    index = str(i)
                with open(format_path(f"../data/output/game_{game_number}/{game_number}_{index}.txt"), "w") as file:
                    file.write(f"{predicted_move} {predicted_value}\n")

                # process the move
                game.play(predicted_move, predicted_value)

                # check if the turn is over
                for turn in game_model.game_turns:
                    if int(turn.starting_position) == game.current_turn:
                        game.change_turn()
                        break

        # add the last score
        game.change_turn()

        # save the scores
        scores = game.scores
        with open(format_path(f"../data/output/game_{game_number}/{game_number}_scores.txt"), "w+") as file:
            for score in scores:
                file.write(f"Player{str(score[0])} {str(score[1])} {str(score[2])}")
                if score != scores[-1]:
                    file.write("\n")
    

if __name__ == "__main__":
    for game_number in range(1, 5):
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        moves = DataLoader(moves_path).load_moves()
        game_model = GameModel(moves, turns_path, scores_path)

        model = CNNModel()
        model.load("../models/cnn_model")
        pipeline = PredictPipeline(moves_path, scores_path, turns_path)
        pipeline.play_game(game_model, model, game_number)