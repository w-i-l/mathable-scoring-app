from data_loader import DataLoader
from game_model import GameModel
from image_processing import ImageProcessing

import cv2 as cv

if __name__ == "__main__":
    game_number = 1
    moves_path = f"../data/train/game_{game_number}"
    scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
    turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

    loader = DataLoader(moves_path)
    moves = loader.load_moves()
    game = GameModel(moves, turns_path, scores_path)

    image_processing = ImageProcessing()

    if __name__ == "__main__":
        game_number = 1
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        game = GameModel(moves, turns_path, scores_path)

        image_processing = ImageProcessing()

        for i in range(10, 15):
            board_contor_1 = image_processing._find_board_contour(moves[i-1].image_path)
            board_contor_2 = image_processing._find_board_contour(moves[i].image_path)

            board_1 = image_processing.crop_board(moves[i-1].image_path, board_contor_1)
            board_2 = image_processing.crop_board(moves[i].image_path, board_contor_2)

            diff = image_processing.find_difference_between_images(board_1, board_2)
            diff_piece = image_processing.find_largest_contour(diff)
            
            # Get original dimensions before any resizing
            original_height, original_width = diff.shape[:2]
            
            x, y, w, h = cv.boundingRect(diff_piece[0])
            w = 105
            h = 105
            
            # Scale coordinates for board_2
            scale_x = board_2.shape[1] / original_width
            scale_y = board_2.shape[0] / original_height
            
            board_contour = [
                int(x * scale_x), 
                int(y * scale_y), 
                int((x + w) * scale_x), 
                int((y + h) * scale_y)
            ]
            
            # Draw on diff (original coordinates)
            diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
            diff = image_processing.draw_rect(diff, [x, y, x+w, y+h])
            diff = cv.resize(diff, (800, 800))
            
            # Draw on board_2 (scaled coordinates)
            board_2 = image_processing.draw_rect(board_2, board_contour)
            board_2 = cv.resize(board_2, (800, 800))
            
            cv.imshow("Difference", diff)
            cv.imshow("Board 2", board_2)
            cv.waitKey(0)
            cv.destroyAllWindows()




        