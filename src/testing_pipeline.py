from data_loader import DataLoader
from game_model import GameModel, GameMove
from game import Game
from image_processing import ImageProcessing
from util import *
from tqdm import tqdm
import cv2 as cv
from multiprocessing import Pool
from cnn.cnn_model import CNNModel

class TestingPipeline:

    def __init__(self, moves_path, scores_path, turns_path):
        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        self.game = GameModel(moves, turns_path, scores_path)
        self.image_processing = ImageProcessing()
    

    def process_move(self, i):
        """Process a single move and return if it was predicted correctly"""
        board_contor_1 = self.image_processing._find_board_contour(self.game.moves[i-1].image_path)
        board_contor_2 = self.image_processing._find_board_contour(self.game.moves[i].image_path)

        board_1 = self.image_processing.crop_board(self.game.moves[i-1].image_path, board_contor_1)
        board_2 = self.image_processing.crop_board(self.game.moves[i].image_path, board_contor_2)

        diff = self.image_processing.find_difference_between_images(board_1, board_2)
        
        original_height, original_width = diff.shape[:2]

        x, y = self.image_processing.find_added_piece_coordinates(diff)
        w = 105
        h = 105
        
        scale_x = board_2.shape[1] / original_width
        scale_y = board_2.shape[0] / original_height
        
        board_contour = [
            int(x * scale_x), 
            int(y * scale_y), 
            int((x + w) * scale_x), 
            int((y + h) * scale_y)
        ]
        
        _, grid_rectangles = self.image_processing.split_board_in_blocks(board_2)
        matching_block_idx, overlap_percentage = self.__find_matching_block(board_contour, grid_rectangles)
        
        if matching_block_idx != -1:
            expected_move = self.game.moves[i].move
            actual_move = self.__convert_index_to_coordinates(matching_block_idx)

            if actual_move == expected_move:
                return True, None
            else:
                return False, f"Move {i}: Expected: {expected_move} -- Actual: {actual_move} -- Overlap: {overlap_percentage:.2f}%"
        
        return False, f"Move {i}: No matching block found"


    @staticmethod
    def process_single_move(move_data):
        """Wrapper function for multiprocessing"""
        pipeline, i = move_data
        return pipeline.process_move(i)


    def test_game_parallel(self, title="Processing moves"):
        # Create list of (pipeline, index) tuples for each move
        move_data = [(self, i) for i in range(1, len(self.game.moves))]
        
        # Create a process pool and use the static method
        cores_to_use = os.cpu_count() - 1
        with Pool(cores_to_use) as pool:
            # Process all moves in parallel
            results = list(tqdm(
                pool.imap(TestingPipeline.process_single_move, move_data),
                total=len(self.game.moves)-1,
                desc=title
            ))

        # Count correct predictions and print errors
        predicted = sum(1 for result, error in results if result)
        
        # Print all errors
        for result, error in results:
            if not result:
                print(error)

        print(f"Predicted: {predicted}/{len(self.game.moves)-1} ({predicted/(len(self.game.moves)-1)*100:.2f}%)")
        
        return predicted / (len(self.game.moves)-1)
    

    def test_game(self, debug=False, title="Processing moves"):
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
            
            board_contour = [
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + w) * scale_x), 
                int((y + h) * scale_y)
            ]
            
            # computing the grid rectangles
            board_2, grid_rectangles = self.image_processing.split_board_in_blocks(board_2)
            
            # find the matching block
            matching_block_idx, overlap_percentage = self.__find_matching_block(board_contour, grid_rectangles)
            
            if matching_block_idx != -1:
                expected_move = self.game.moves[i].move
                predicted_move = self.__convert_index_to_coordinates(matching_block_idx)

                added_piece = board_2[board_contour[1]:board_contour[3], board_contour[0]:board_contour[2]]
                added_piece = cv.resize(added_piece, (105, 105))
                predicted_value = model.predict(added_piece)
                expected_value = self.game.moves[i].value

                if predicted_move == expected_move and predicted_value == expected_value:
                    predicted += 1
                elif expected_move != predicted_move:
                    print(f"Expected: {expected_move} -- Actual: {predicted_move} -- Overlap: {overlap_percentage:.2f}%")
                elif expected_value != predicted_value:
                    print(f"Expected: {expected_value} -- Predicted: {predicted_value}")
                    
                # if debug:
                #     # draw the expected block
                #     matched_rect = grid_rectangles[matching_block_idx]
                #     cv.rectangle(board_2, matched_rect[0], matched_rect[1], (0, 0, 255), 2)
                    
                #     # draw the actual block
                #     board_2 = self.image_processing.draw_rect(board_2, board_contour, color=(0, 255, 0), thickness=2)

                #     add_piece = board_2[board_contour[1]:board_contour[3], board_contour[0]:board_contour[2]]
                #     add_piece = cv.resize(add_piece, (300, 300))
                #     add_piece = cv.cvtColor(add_piece, cv.COLOR_BGR2HSV)
                #     mask = cv.inRange(add_piece, (55, 0, 0), (255, 255, 255))
                #     add_piece = cv.bitwise_and(add_piece, add_piece, mask=mask)
                #     cv.imwrite(f"../data/transformed_pieces/{i}.png", add_piece)


                #     add_piece = cv.GaussianBlur(add_piece, (3, 3), 0)
                #     add_piece = cv.Canny(add_piece, 20, 255)
                #     number_contours, _ = cv.findContours(add_piece, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                #     add_piece = cv.drawContours(add_piece, number_contours, -1, (0, 255, 0), 2)

                #     add_piece = cv.rectangle(add_piece, (0, 0), (add_piece.shape[1], add_piece.shape[0]), (0, 0, 255), 2)
                #     cv.imshow("Added Piece", add_piece)
                    
                #     board_2 = cv.resize(board_2, (800, 800))
                #     diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
                    
                #     # draw the difference
                #     diff = self.image_processing.draw_rect(diff, [x, y, x+w, y+h])
                #     diff = cv.resize(diff, (800, 800))

                #     cv.imshow("Difference", diff)
                #     cv.imshow("Board 2", board_2)
                #     cv.waitKey(0)
                #     cv.destroyAllWindows()

        print(f"Predicted: {predicted}/{len(self.game.moves)-1} ({predicted/(len(self.game.moves)-1)*100:.2f}%)")
        return predicted / (len(self.game.moves)-1)
    

    def play_game(self, game_model, model, game_number=1):
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
            
            board_contour = [
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + w) * scale_x), 
                int((y + h) * scale_y)
            ]
            
            # computing the grid rectangles
            board_2, grid_rectangles = self.image_processing.split_board_in_blocks(board_2)
            
            # find the matching block
            matching_block_idx, overlap_percentage = self.__find_matching_block(board_contour, grid_rectangles)
            
            if matching_block_idx != -1:
                predicted_move = self.__convert_index_to_coordinates(matching_block_idx)

                added_piece = board_2[board_contour[1]:board_contour[3], board_contour[0]:board_contour[2]]
                added_piece = cv.resize(added_piece, (105, 105))
                predicted_value = model.predict(added_piece)

                if i < 10:
                    index = f"0{i}"
                else:
                    index = str(i)
                with open(format_path(f"../data/output/game_{game_number}/{game_number}_{index}.txt"), "w") as file:
                    file.write(f"{predicted_move} {predicted_value}\n")

                game.play(predicted_move, predicted_value)

                for turn in game_model.game_turns:
                    if int(turn.starting_position) == game.current_turn:
                        game.change_turn()
                        break

        game.change_turn()
        return game.scores
                    
    
    def __calculate_intersection_area(self, rect1, rect2):
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0
        
        return (x_right - x_left) * (y_bottom - y_top)


    def __find_matching_block(self, contour_rect, grid_rectangles) -> tuple:
        max_overlap = 0
        best_block_index = -1
        
        # convert grid rectangles to [x1, y1, x2, y2] format
        formatted_grid_rects = [[r[0][0], r[0][1], r[1][0], r[1][1]] for r in grid_rectangles]
        
        # calculate contour rectangle area
        contour_area = (contour_rect[2] - contour_rect[0]) * (contour_rect[3] - contour_rect[1])
        
        for idx, grid_rect in enumerate(formatted_grid_rects):
            intersection = self.__calculate_intersection_area(contour_rect, grid_rect)
            
            overlap_percentage = (intersection / contour_area) * 100 if contour_area > 0 else 0
            
            if overlap_percentage > max_overlap:
                max_overlap = overlap_percentage
                best_block_index = idx
        
        return best_block_index, max_overlap


    def __convert_index_to_coordinates(self, index):
        divider = 14
        row = index // divider
        col = index % divider

        return f"{row+1}{chr(col+65)}"


if __name__ == "__main__":
    game_number = 1
    moves_path = f"../data/train/game_{game_number}"
    scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
    turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

    pipeline = TestingPipeline(moves_path, scores_path, turns_path)
    # pipeline.test_game(debug=True)
    moves = DataLoader(moves_path).load_moves()
    game_model = GameModel(moves, turns_path, scores_path)

    model = CNNModel()
    model.load("../models/cnn_model")

    scores = pipeline.play_game(game_model, model)
    # with open(format_path(f"../data/output/{game_number}_scores.txt"), "w") as file:
    #     for score in scores:
    #         file.write(f"Player{str(score[0])} {str(score[1])} {str(score[2])}\n")


    files = os.listdir(format_path("../data/output"))
    outputs = []
    for file in files:
        if file.endswith(".txt"):
            with open(format_path(f"../data/output/{file}"), "r") as f:
                outputs.append(tuple(f.read().strip().split()))

    outputs = set(outputs)
    moves = set([(move.move, str(move.value)) for move in moves[1:]])
    print()
    print(f"Diff: {outputs.difference(moves)}")

    # pipeline.test_game_parallel()

