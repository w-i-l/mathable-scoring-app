from models.game_model import GameModel
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

class StatisticalPlots:
    '''
    A class which provides functionality to plot statistical distributions
    '''

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
        
        # calculate statistics
        mean = np.mean(values)
        median = np.median(values)
        trimmed_mean = stats.trim_mean(values, 0.1)  # 10% trimmed mean
        
        # calculate normal distribution
        std = np.std(values)
        pdf = stats.norm.pdf(x, mean, std)
        pdf = pdf * len(values) * (max(values) - min(values)) / 5  # Scale for visibility
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(values, bins=20, density=False, alpha=0.6, color='gray', label='Histogram')
        
        plt.plot(x, pdf, 'r-', lw=2, label=f'Normal Distribution (μ={mean:.2f}, σ={std:.2f})')
        
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


if __name__ == "__main__":
    plots = StatisticalPlots()
    plots.plot_distribution(is_validation=False)
    plots.plot_statistical_functions(is_validation=False)