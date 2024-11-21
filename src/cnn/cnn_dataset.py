import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from game_model import GameModel
from .cnn_data_loader import DataLoader


class DataSet:
    image_size = 40

    def __init__(self, data_loader, images_per_class=1000):
        self.data_loader = data_loader
        self.images_per_class = images_per_class


    def __augment_image(self, img, brightness_steps, saturation_steps, no_of_cropped_images):
        augmented_images = []
        img_tensor = tf.convert_to_tensor(img)
        
        # Original image
        augmented_images.append(img)
        
        # Brightness variations
        factors = np.linspace(0.6, 1.4, brightness_steps)
        saturations = np.linspace(0.6, 1.4, saturation_steps)

        for factor in factors:
            for saturation in saturations:
                for _ in range(no_of_cropped_images):
                    bright = tf.image.adjust_brightness(img_tensor, delta=factor-1)
                    saturated = tf.image.adjust_saturation(bright, saturation)
                    min_crop_size = int(self.image_size * 0.7)
                    crop_size = tf.random.uniform([], min_crop_size, self.image_size, dtype=tf.int32)
                    cropped = tf.image.random_crop(saturated, [crop_size, crop_size, 3])
                    resized = tf.image.resize(cropped, [self.image_size, self.image_size])
                    augmented_images.append(resized.numpy())
                
        return augmented_images
    

    def generate_dataset(self, should_load=False, save_images=False):
        train_data = []
        train_labels = []
        
        unique_pieces = sorted(list(self.data_loader.files.keys()))
        piece_to_class = {piece: idx for idx, piece in enumerate(unique_pieces)}

        if should_load:
            # load data from augmented_images
            for piece in unique_pieces:
                files = os.listdir(f"../data/augmented_images/{piece}")
                for file in tqdm(files, desc=f"Loading {piece}"):
                    img = cv.imread(f"../data/augmented_images/{piece}/{file}")
                    if img is not None:
                        img = cv.resize(img, (self.image_size, self.image_size))
                        img = img.astype(np.float32) / 255.0
                        train_data.append(img)
                        train_labels.append(piece_to_class[piece])

            return train_data, train_labels
        
        
        for piece, files in tqdm(self.data_loader.files.items(), desc="Loading data"):

            if save_images:
                if not os.path.exists(f"../data/augmented_images/{piece}"):
                    os.makedirs(f"../data/augmented_images/{piece}")
                else:
                    # empty the folder
                    for file in os.listdir(f"../data/augmented_images/{piece}"):
                        os.remove(f"../data/augmented_images/{piece}/{file}")

            brightness_steps, saturation_steps, no_of_cropped_images = self.get_no_of_attributes_for_class(len(files))
            for file in tqdm(files, desc=f"Loading {piece}"):
                img = cv.imread(file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if img is not None:
                    img = cv.resize(img, (self.image_size, self.image_size))
                    img = img.astype(np.float32) / 255.0
                    
                    # Generate augmented images
                    augmented = self.__augment_image(img, brightness_steps, saturation_steps, no_of_cropped_images)
                    train_data.extend(augmented)
                    train_labels.extend([piece_to_class[piece]] * len(augmented))

                    if save_images:
                        images_number = len(os.listdir(f"../data/augmented_images/{piece}"))
                        for idx, img in enumerate(augmented):
                            idx = images_number + idx
                            img = tf.keras.utils.save_img(f"../data/augmented_images/{piece}/{idx}.png", img)

        return train_data, train_labels
    

    def get_no_of_attributes_for_class(self, no_of_templates):
        brightness_steps = 0
        saturation_steps = 0
        no_of_cropped_images = 0

        total_images = 0
        while True:
            brightness_steps += 1
            saturation_steps += 1
            no_of_cropped_images += 1

            total_images = no_of_templates * (1 + brightness_steps * saturation_steps * no_of_cropped_images)

            if total_images >= self.images_per_class:
                break
        
        if total_images >= self.images_per_class * 1.2:
            no_of_cropped_images -= 1
        
        total_images = no_of_templates * (1 + brightness_steps * saturation_steps * no_of_cropped_images)
        if total_images >= self.images_per_class * 1.2:
            saturation_steps -= 1

        return brightness_steps, saturation_steps, no_of_cropped_images
    

    def split_dataset(self, train_data, train_labels, split_ratio=0.8):
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

        # plot the distribution
        train_counts = {piece: 0 for piece in unique_classes}
        val_counts = {piece: 0 for piece in unique_classes}
        for piece in y_train:
            train_counts[piece] += 1
        for piece in y_val:
            val_counts[piece] += 1

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)


if __name__ == "__main__":
    loader = DataLoader("../data/cnn/train")
    data_set = DataSet(loader, images_per_class=1000)
    for i in range(2, 20):
        brightness_steps, saturation_steps, no_of_cropped_images = data_set.get_no_of_attributes_for_class(i)
        print(f"Templates: {i} - brightness_steps: {brightness_steps}, saturation_steps: {saturation_steps}, no_of_cropped_images: {no_of_cropped_images}", end=" - ")
        print(f"Total images: {i * (1 + brightness_steps * saturation_steps * no_of_cropped_images)}")
        

