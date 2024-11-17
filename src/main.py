from data_loader import DataLoader
from game_model import GameModel
from image_processing import ImageProcessing
from util import format_path
from tqdm import tqdm
import numpy as np

import cv2 as cv


def calculate_intersection_area(rect1, rect2):
    """Calculate the intersection area between two rectangles
    
    Args:
        rect1: First rectangle coordinates [x1, y1, x2, y2]
        rect2: Second rectangle coordinates in the same format
    """
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0
    
    return (x_right - x_left) * (y_bottom - y_top)

def find_matching_block(contour_rect, grid_rectangles) -> tuple:
    """Find the grid block that best matches the contour
    
    Args:
        contour_rect: Contour rectangle coordinates [x1, y1, x2, y2]
        grid_rectangles: List of grid block coordinates [((x1,y1), (x2,y2)), ...]
    
    Returns:
        tuple: (block_index, intersection_percentage)
    """
    max_overlap = 0
    best_block_index = -1
    
    # Convert grid rectangles to [x1, y1, x2, y2] format
    formatted_grid_rects = [[r[0][0], r[0][1], r[1][0], r[1][1]] for r in grid_rectangles]
    
    # Calculate contour rectangle area
    contour_area = (contour_rect[2] - contour_rect[0]) * (contour_rect[3] - contour_rect[1])
    
    for idx, grid_rect in enumerate(formatted_grid_rects):
        intersection = calculate_intersection_area(contour_rect, grid_rect)
        
        # Calculate overlap as percentage of contour area
        overlap_percentage = (intersection / contour_area) * 100 if contour_area > 0 else 0
        
        if overlap_percentage > max_overlap:
            max_overlap = overlap_percentage
            best_block_index = idx
    
    return best_block_index, max_overlap


def convert_index_to_coordinates(index):
    ''' returns a string with the row being a number and the column being a letter '''
    divider = 14
    row = index // divider
    col = index % divider

    return f"{row+1}{chr(col+65)}"



# Modified main code:
if __name__ == "__main__":
    for game_number in range(1, 2):
        moves_path = f"../data/train/game_{game_number}"
        scores_path = f"../data/train/game_{game_number}/{game_number}_scores.txt"
        turns_path = f"../data/train/game_{game_number}/{game_number}_turns.txt"

        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        game = GameModel(moves, turns_path, scores_path)

        image_processing = ImageProcessing()


        # ################
        # for i in range(1, len(moves)):
        #     board_contor_1 = image_processing._find_board_contour(moves[i-1].image_path)
        #     board_contor_2 = image_processing._find_board_contour(moves[i].image_path)

        #     board_1 = image_processing.crop_board(moves[i-1].image_path, board_contor_1)
        #     board_2 = image_processing.crop_board(moves[i].image_path, board_contor_2)

        #     diff = image_processing.find_difference_between_images(board_1, board_2)
            
        #     cv.imwrite(format_path(f"../data/diff/board_{i}-board_{i+1}.jpg"), diff)
        # exit(0)
        # ################


        predicted = 0

        for i in tqdm(range(1, len(moves)), desc=f"Game {game_number}"):
            board_contor_1 = image_processing._find_board_contour(moves[i-1].image_path)
            board_contor_2 = image_processing._find_board_contour(moves[i].image_path)

            board_1 = image_processing.crop_board(moves[i-1].image_path, board_contor_1)
            board_2 = image_processing.crop_board(moves[i].image_path, board_contor_2)

            cv.imwrite(format_path(f"../data/cropped/board_{i}.jpg"), board_1)

            # debug_board_1 = board_1.copy()
            # debug_board_2 = board_2.copy()
            # debug_board_1 = cv.resize(debug_board_1, (800, 800))
            # debug_board_2 = cv.resize(debug_board_2, (800, 800))
            # cv.imshow("Board 1", debug_board_1)
            # cv.imshow("Board 2", debug_board_2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()


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
            
            # Get grid blocks
            board_2, grid_rectangles = image_processing.split_board_in_blocks(board_2)
            
            # Find matching block
            matching_block_idx, overlap_percentage = find_matching_block(board_contour, grid_rectangles)
            
            if matching_block_idx != -1:
                # Draw matching block in different color
                matched_rect = grid_rectangles[matching_block_idx]
                cv.rectangle(board_2, matched_rect[0], matched_rect[1], (0, 0, 255), 2)  # Red color for matched block
                
                # Draw the contour rectangle
                board_2 = image_processing.draw_rect(board_2, board_contour, color=(0, 255, 0), thickness=2)  # Green color for contour
                
                if convert_index_to_coordinates(matching_block_idx) == moves[i].move:
                    predicted += 1
                else:
                    print(f"Found matching block {convert_index_to_coordinates(matching_block_idx)} with {overlap_percentage:.2f}% overlap -- matched piece: {moves[i].move} -- matched {convert_index_to_coordinates(matching_block_idx) == moves[i].move}")
                board_2 = cv.resize(board_2, (800, 800))
                diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
                diff = image_processing.draw_rect(diff, [x, y, x+w, y+h])
                diff = cv.resize(diff, (800, 800))
                cv.imshow("Difference", diff)
                cv.imshow("Board 2", board_2)
                cv.waitKey(0)
                cv.destroyAllWindows()

        
            
            # # Resize for display
            # diff = cv.resize(diff, (800, 800))
            # board_2 = cv.resize(board_2, (800, 800))
            
            # cv.imshow("Difference", diff)
            # cv.imshow("Board 2", board_2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        print(f"Predicted: {predicted}/{len(moves)-1} ({predicted/(len(moves)-1)*100:.2f}%)")