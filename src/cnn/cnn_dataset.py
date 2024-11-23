import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from .cnn_data_loader import DataLoader
from utils.helper_functions import format_path


class DataSet:
    '''
    Class to generate the dataset for the CNN model

    It generates the dataset by augmenting the images and splitting them into training and validation sets
    '''

    # reduced the image size for faster training and less memory usage
    image_size = 40

    def __init__(self, data_loader: DataLoader, images_per_class=1000):
        '''
        Parameters:
        -----------
        data_loader (DataLoader): An instance of DataLoader class as the data source
        images_per_class (int): The number of images to generate for each class
        '''

        self.data_loader = data_loader
        self.images_per_class = images_per_class


    def __augment_image(
            self,
            img: np.ndarray,
            brightness_steps: int, 
            saturation_steps: int, 
            no_of_cropped_images: int
    ) -> list[np.ndarray]:
        '''
        Augments the image by generating brightness variations, saturation variations, and cropping the image
        
        Parameters:
        -----------
        img (np.ndarray): The image to augment
        brightness_steps (int): The number of brightness variations to generate
        saturation_steps (int): The number of saturation variations to generate
        no_of_cropped_images (int): The number of cropped images to generate

        Returns:
        --------
        list[np.ndarray]: A list of augmented images
        '''

        augmented_images = []
        img_tensor = tf.convert_to_tensor(img)
        
        # store the original image
        augmented_images.append(img)
        
        # compute the variations
        brightnesses = np.linspace(0.6, 1.4, brightness_steps)
        saturations = np.linspace(0.6, 1.4, saturation_steps)

        for brightness in brightnesses:
            for saturation in saturations:
                for _ in range(no_of_cropped_images):
                    bright = tf.image.adjust_brightness(img_tensor, delta=brightness-1)
                    saturated = tf.image.adjust_saturation(bright, saturation)

                    min_crop_size = int(self.image_size * 0.7)
                    crop_size = tf.random.uniform([], min_crop_size, self.image_size, dtype=tf.int32)
                    cropped = tf.image.random_crop(saturated, [crop_size, crop_size, 3])

                    # resize the image to the desired size
                    resized = tf.image.resize(cropped, [self.image_size, self.image_size])
                    augmented_images.append(resized.numpy())
                
        return augmented_images
    

    def generate_dataset(self, should_load=False, save_images=False) -> tuple[list[np.ndarray], list[int]]:
        '''
        Generates the dataset by augmenting the images provided by the DataLoader

        Parameters:
        -----------
        should_load (bool): If True, the function will load the images from the `augmented_images` folder
        save_images (bool): If True, the function will save the augmented images generated to the `augmented_images` folder

        Returns:
        --------
        list[np.ndarray], list[int]: The training data and labels
        '''

        train_data = []
        train_labels = []
        
        # get the unique pieces and assign a class to each piece
        unique_pieces = sorted(list(self.data_loader.files.keys()))
        piece_to_class = {piece: idx for idx, piece in enumerate(unique_pieces)}

        # load data from augmented_images folder
        if should_load:
            train_data, train_labels = self.__load_dataset()
            return train_data, train_labels
        
        
        for piece, files in tqdm(self.data_loader.files.items(), desc="Loading data"):
            if save_images:
                self.__handle_folder(f"../data/augmented_images/{piece}")

            brightness_steps, saturation_steps, no_of_cropped_images = self.get_no_of_attributes_for_class(len(files))
            for file in tqdm(files, desc=f"Loading {piece}"):
                img = cv.imread(file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                if img is not None:
                    img = cv.resize(img, (self.image_size, self.image_size))
                    img = img.astype(np.float32) / 255.0
                    
                    # generate augmented images
                    augmented = self.__augment_image(img, brightness_steps, saturation_steps, no_of_cropped_images)
                    train_data.extend(augmented)
                    train_labels.extend([piece_to_class[piece]] * len(augmented))

                    # save the images to their respective folders
                    if save_images:
                        piece_folder_path = f"../data/augmented_images/{piece}"
                        images_number = len(os.listdir(format_path(piece_folder_path)))
                        for idx, img in enumerate(augmented):
                            idx = images_number + idx
                            img_path = f"{piece_folder_path}/{idx}.png"
                            img = tf.keras.utils.save_img(format_path(img_path), img)

        return train_data, train_labels
    

    def __handle_folder(self, folder_path: str):
        '''
        Creates a folder if it does not exist else empties the folder

        Parameters:
        -----------
        folder_path (str): The path to the folder
        '''

        folder_path = format_path(folder_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            for file in os.listdir(folder_path):
                os.remove(f"{folder_path}/{file}")

    
    def __load_dataset(self) -> tuple[list[np.ndarray], list[int]]:
        '''
        Loads the dataset from the `augmented_images` folder

        Returns:
        --------
        list[np.ndarray], list[int]: The training data and labels
        '''

        train_data = []
        train_labels = []

        # get the unique pieces and assign a class to each piece
        unique_pieces = sorted(list(self.data_loader.files.keys()))
        piece_to_class = {piece: idx for idx, piece in enumerate(unique_pieces)}

        # load data from augmented_images folder
        for piece in unique_pieces:
            files = os.listdir(f"../data/augmented_images/{piece}")
            for file in tqdm(files, desc=f"Loading {piece}"):
                img = cv.imread(f"../data/augmented_images/{piece}/{file}")
                if img is not None:
                    img = cv.resize(img, (self.image_size, self.image_size))
                    # normalize the image
                    img = img.astype(np.float32) / 255.0

                    train_data.append(img)
                    train_labels.append(piece_to_class[piece])

        return train_data, train_labels
    

    def get_no_of_attributes_for_class(self, no_of_templates: int) -> tuple[int, int, int]:
        '''
        Computes the number of brightness steps, saturation steps, and number of cropped images to generate
        in order to have the desired number of images per class based on the number of templates that
        are available for that class

        The function will find those quantities such that the total number of images generated is bigger
        than the number of images per class, then it will adjust the quantities to be as close as possible
        to the desired number of images per class, having a maximum of 20% difference

        It will prioritize reducing the number of cropped images, then the saturation steps in order to
        have a smaller number of images generated

        Example:
        --------
        If the number of images per class is 1000 and the number of templates is 10, the function will return
        [5, 5, 4] as the number of brightness steps, saturation steps, and number of cropped images to generate
        resulting in a total of 10 * (1 + 5 * 5 * 4) = 1010 images

        Parameters:
        -----------
        no_of_templates (int): The number of templates for the class
        '''

        brightness_steps = 0
        saturation_steps = 0
        no_of_cropped_images = 0

        # loop until the total number of images is bigger than the desired number of images per class
        total_images = 0
        while True:
            brightness_steps += 1
            saturation_steps += 1
            no_of_cropped_images += 1

            total_images = no_of_templates * (1 + brightness_steps * saturation_steps * no_of_cropped_images)

            if total_images >= self.images_per_class:
                break
        
        # adjust the quantities to be as close as possible to the desired number of images per class
        if total_images >= self.images_per_class * 1.2:
            no_of_cropped_images -= 1
        
        total_images = no_of_templates * (1 + brightness_steps * saturation_steps * no_of_cropped_images)
        if total_images >= self.images_per_class * 1.2:
            saturation_steps -= 1

        return brightness_steps, saturation_steps, no_of_cropped_images
    

    def split_dataset(self, train_data: list[np.ndarray], train_labels: list[int], split_ratio=0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Splits the dataset into training and validation sets based on the split ratio
        It will split the data for each class rather than splitting the whole dataset

        Parameters:
        -----------
        train_data (list[np.ndarray]): The training data
        train_labels (list[int]): The training labels
        split_ratio (float): The ratio to split the data into training and validation sets. Default is 0.8 which means 80% training and 20% validation

        Returns:
        --------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: The training data, training labels, validation data, validation labels
        '''

        # Split the data for each class
        X_train, y_train, X_val, y_val = [], [], [], []
        unique_classes = np.unique(train_labels)
        
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        
        for cls in unique_classes:
            cls_indices = np.where(train_labels == cls)[0]
            np.random.shuffle(cls_indices)
            
            split = int(split_ratio * len(cls_indices))
            cls_train_indices = cls_indices[:split]
            cls_val_indices = cls_indices[split:]
            
            X_train.extend(train_data[cls_train_indices])
            y_train.extend(train_labels[cls_train_indices])
            X_val.extend(train_data[cls_val_indices])
            y_val.extend(train_labels[cls_val_indices])

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)


if __name__ == "__main__":
    loader = DataLoader("../data/cnn/train")
    data_set = DataSet(loader, images_per_class=1000)
    for i in range(2, 20):
        brightness_steps, saturation_steps, no_of_cropped_images = data_set.get_no_of_attributes_for_class(i)
        print(f"Templates: {i} - brightness_steps: {brightness_steps}, saturation_steps: {saturation_steps}, no_of_cropped_images: {no_of_cropped_images}", end=" - ")
        print(f"Total images: {i * (1 + brightness_steps * saturation_steps * no_of_cropped_images)}")