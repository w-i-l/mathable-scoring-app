from utils.helper_functions import format_path
import os
from models.game_model import GameModel

class DataLoader:
    '''
    A class which provides functionality to load images
    for CNN training and validation.
    '''

    def __init__(self, folder_path: str):
        '''
        Parameters:
        -----------
        folder_path (str): The path to the folder containing the images. It should have subfolders for each piece.
        '''
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} does not exist.")
        
        self.folder_path = format_path(folder_path)
        self.__load_files()

    
    def __load_files(self):
        '''
        Loads pieces from the folder path and stores them in a dictionary.\n

        Key: piece number -
        Value: list of file paths
        '''
        
        classes_folders = os.listdir(self.folder_path)
        self.classes = [i for i in GameModel.available_pieces()]
        self.files = {i: [] for i in self.classes}

        for folder in classes_folders:
            if not os.path.isdir(os.path.join(self.folder_path, folder)):
                continue

            files = os.listdir(os.path.join(self.folder_path, folder))
            piece = int(folder)

            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    relative_path = os.path.join(self.folder_path, folder, file)
                    self.files[piece].append(relative_path)