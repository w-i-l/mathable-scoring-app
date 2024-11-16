import os

class DataOrganizer():
    def __init__(self, folder_path):
        self.folder_path = self.__format_path(folder_path)
        self.files = self.__load_files()

    
    def move_files(self):
        self.__create_folders()
        for file in self.files:
            game_number = file.split('_')[0]
            source_path = os.path.join(self.folder_path, file)
            destination_path = os.path.join(self.folder_path,f"game_{game_number}", file)
            os.rename(source_path, destination_path)
        

    def undo_move_files(self):
        folders = [os.path.join(self.folder_path, f"game_{i+1}") for i in range(self.__get_number_of_games())]
        
        for folder in folders:
            files = os.listdir(folder)
            for file in files:
                source_path = os.path.join(folder, file)
                destination_path = os.path.join(self.folder_path, file)
                os.rename(source_path, destination_path)
            
            os.rmdir(folder)


    def __format_path(self, path):
        if os.name == 'nt':
            path = path.replace('/', '\\')
        else:
            path = path.replace('\\', '/')
        return path


    def __load_files(self):
        files = os.listdir(self.folder_path)
        valid_files = [files for files in files if self.__is_valid_file(files)]
        return valid_files
    
    
    def __is_valid_file(self, file):
        if os.path.isdir(file):
            print(f"{file} is a directory")
            return False
        return file.endswith('.jpg') or file.endswith('.txt')
    
    
    def __create_folders(self):
        number_of_folders = self.__get_number_of_games()
        for i in range(number_of_folders):
            folder_name = f"game_{i+1}"
            folder_path = os.path.join(self.folder_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


    def __get_number_of_games(self):
        if self.files == None:
            raise Exception("Files not loaded yet.")
        
        number_of_games = 0

        for file in self.files:
            game_number = int(file.split('_')[0])
            if game_number > number_of_games:
                number_of_games = game_number

        if number_of_games == 0:
            number_of_games = len(os.listdir(self.folder_path))
        
        return number_of_games
    
if __name__ == "__main__":
    folder_path = "../data/train"
    data_organizer = DataOrganizer(folder_path)
    data_organizer.move_files()