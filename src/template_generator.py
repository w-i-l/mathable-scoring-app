import cv2 as cv
import numpy as np
from image_processing import ImageProcessing
from game_model import GameModel
from number_reader import NumberReader
from util import format_path
from time import sleep

class TemplateGenerator:
    def __init__(self, board_path):
        self.board_path = format_path(board_path)
        self.image_processing = ImageProcessing()

    def generate_template(self):
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
        board = cv.resize(board, (800, 800))
        cv.imshow("Grid", board)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return pieces

if __name__ == "__main__":
    generator = TemplateGenerator("../data/all_pieces_board.jpg")
    pieces = generator.generate_template()
    available_pieces = GameModel.available_pieces()
    for available_piece, piece in zip(available_pieces, pieces):
        image = NumberReader().extract_number(piece) 
        cv.imshow(f"Piece {available_piece}", image)
        cv.waitKey(1)
        sleep(0.5)
        cv.destroyAllWindows()
        cv.imwrite(f"../data/templates/{available_piece}.jpg", piece)
