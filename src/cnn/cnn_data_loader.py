from util import format_path
import os
from game_model import GameModel


class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = format_path(folder_path)
        self.__load_files()

    
    def __load_files(self):
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
