from game_model import GameModel
from data_loader import DataLoader
from image_processing import ImageProcessing
from game_model import GameModel
from util import format_path
from tqdm import tqdm
import cv2 as cv
from time import sleep
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

class DataGenerator:
    def __init__(self):
        self.image_processing = ImageProcessing()
        self.game = None
        self.__generate_folders()

    
    def __generate_folders(self):
        for i in GameModel.available_pieces():
            folder_path = f"../data/cnn/train/{i}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path = f"../data/cnn/validation/{i}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


    def generate_data(self, moves_path, scores_path, turns_path, is_validation=False, game_number=1):
        loader = DataLoader(moves_path)
        moves = loader.load_moves()
        self.game = GameModel(moves, turns_path, scores_path)

        for i in tqdm(range(1, len(self.game.moves)), desc=f"Game {game_number}"):
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
            

            added_piecce = board_2[board_contour[1]:board_contour[3], board_contour[0]:board_contour[2]]
            added_piecce = cv.resize(added_piecce, (105, 105))
            
            subfolder = "validation" if is_validation else "train"
            game_number = game_number if not is_validation else "validation"
            piece_path = f"../data/cnn/{subfolder}/{self.game.moves[i].value}/{game_number}_{i}.png"
            # print(piece_path)
            cv.imwrite(piece_path, added_piecce)


    def plot_distribution(self, is_validation=False):
        pieces = GameModel.available_pieces()
        counts = {str(piece): 0 for piece in pieces}
        subfolder = "validation" if is_validation else "train"
        
        for piece in pieces:
            folder_path = f"../data/cnn/{subfolder}/{piece}"
            counts[str(piece)] = len(os.listdir(folder_path))
        
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(list(counts.keys()), list(counts.values()), 
                    linewidth=2.0,
                    edgecolor='black',
                    width=0.7)
        
        mean_value = sum(counts.values()) / len(counts)
        plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
        
        plt.xticks(range(len(counts)), list(counts.keys()), rotation=0)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Distribution of Pieces in Dataset')
        plt.xlabel('Piece Number')
        plt.ylabel('Frequency')
        plt.margins(x=0.02)
        plt.legend()
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def plot_statistical_functions(self, is_validation=False):
        pieces = GameModel.available_pieces()
        counts = {str(piece): 0 for piece in pieces}
        subfolder = "validation" if is_validation else "train"
        
        for piece in pieces:
            folder_path = f"../data/cnn/{subfolder}/{piece}"
            counts[str(piece)] = len(os.listdir(folder_path))
        
        values = list(counts.values())
        x = np.linspace(min(values), max(values), 100)
        
        # Calculate statistics
        mean = np.mean(values)
        median = np.median(values)
        trimmed_mean = stats.trim_mean(values, 0.1)  # 10% trimmed mean
        
        # Calculate normal distribution
        std = np.std(values)
        pdf = stats.norm.pdf(x, mean, std)
        pdf = pdf * len(values) * (max(values) - min(values)) / 5  # Scale for visibility
        
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.hist(values, bins=20, density=False, alpha=0.6, color='gray', label='Histogram')
        
        # Plot distributions
        plt.plot(x, pdf, 'r-', lw=2, label=f'Normal Distribution (μ={mean:.2f}, σ={std:.2f})')
        
        # Plot statistical measures
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(x=median, color='g', linestyle='--', label=f'Median: {median:.2f}')
        plt.axvline(x=trimmed_mean, color='b', linestyle='--', label=f'Trimmed Mean (10%): {trimmed_mean:.2f}')
        
        plt.title('Statistical Distribution of Piece Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()      


    def move_from_folder(self, folder_path, name="template"):
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
    
    generator.plot_distribution(is_validation=False)
    generator.plot_statistical_functions(is_validation=False)
    # generator.move_from_folder("../data/templates/board", name="template_board")
    # generator.move_from_folder("../data/templates/togheter", name="template_togheter")