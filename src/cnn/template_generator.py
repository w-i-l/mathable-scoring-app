import cv2 as cv
import numpy as np
from utils.image_processing import ImageProcessing
from models.game_model import GameModel
from utils.helper_functions import format_path
from time import sleep
from tqdm import tqdm
import os

class TemplateGenerator:

    def __init__(self, board_path):
        self.board_path = format_path(board_path)
        self.image_processing = ImageProcessing()


    def generate_templates_from_board(self):
        # Read and preprocess the board image
        board = cv.imread(self.board_path)
        if board is None:
            raise FileNotFoundError(f"Image not found at {self.board_path}")
            
        # Resize to exactly 1470x1470 (14 * 105) to ensure perfect 105x105 grid
        board = cv.resize(board, (1470, 1470))
        
        # Create empty list to store pieces
        pieces = []
        
        # Extract each 105x105 piece
        for i in range(14):  # rows
            for j in range(14):  # columns
                if i % 2 == 0 and j % 2 == 0:
                    if i < 12 or j <= 6:
                        # Calculate piece coordinates
                        x = j * 105
                        y = i * 105
                        # Extract piece
                        piece = board[y:y+105, x:x+105]
                        pieces.append(piece)  # Store piece with its position
                        # # Draw rectangle for visualization
                        # cv.rectangle(board, (x, y), (x+105, y+105), (0, 255, 255), 8)
            
        # Display results
        # board = cv.resize(board, (800, 800))
        # cv.imshow("Grid", board)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return pieces
    

    def generate_templates_from_all_together(self):
        board = cv.imread(self.board_path)
        if board is None:
            raise FileNotFoundError(f"Image not found at {self.board_path}")
        
        # Calculate grid dimensions
        num_cols = 8
        num_rows = 6
        cell_size = 105
        
        # Calculate total grid size
        grid_width = num_cols * cell_size
        grid_height = num_rows * cell_size
        
        # Resize to exact grid dimensions
        board = cv.resize(board, (grid_width, grid_height))
        
        pieces = []
        for i in range(num_rows):
            for j in range(num_cols):
                # Skip last two slots in bottom row
                if i == num_rows - 1 and j >= 6:
                    continue
                    
                # Calculate piece coordinates with border offset
                x = j * cell_size
                y = i * cell_size
                
                # Extract piece with exact dimensions
                piece = board[y:y+cell_size, x:x+cell_size]
                # cv.imshow("Piece", piece)
                # cv.waitKey(1)
                # sleep(0.5)
                
                piece = cv.resize(piece, (cell_size, cell_size))
                
                pieces.append(piece)
        # cv.destroyAllWindows()

        return pieces

if __name__ == "__main__":
    if not os.path.exists("../data/templates"):
        os.makedirs("../data/templates")
    if not os.path.exists("../data/templates/board"):
        os.makedirs("../data/templates/board")
    if not os.path.exists("../data/templates/togheter"):
        os.makedirs("../data/templates/togheter")

    generator = TemplateGenerator("../data/all_pieces_board.jpg")
    pieces = generator.generate_templates_from_board()
    available_pieces = GameModel.available_pieces()
    for available_piece, piece in tqdm(zip(available_pieces, pieces)):
        # cv.imshow(f"Piece {available_piece}", piece)
        # cv.waitKey(1)
        # sleep(0.5)
        # cv.destroyAllWindows()
        cv.imwrite(f"../data/templates/board/{available_piece}.jpg", piece)


    generator = TemplateGenerator("../data/all_pieces_togheter.png")
    pieces = generator.generate_templates_from_all_together()
    available_pieces = GameModel.available_pieces()
    for available_piece, piece in tqdm(zip(available_pieces, pieces)):
        piece = cv.cvtColor(piece, cv.COLOR_BGR2RGB)
        piece = cv.resize(piece, (105, 105))
        # cv.imshow(f"Piece {available_piece}", piece)
        # cv.waitKey(1)
        # sleep(0.5)
        # cv.destroyAllWindows()
        piece = cv.cvtColor(piece, cv.COLOR_RGB2BGR)
        cv.imwrite(f"../data/templates/togheter/{available_piece}.jpg", piece)
