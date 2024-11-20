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
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


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

class CNNModel:
    def __init__(self):
        self.model = self.__create_model()
        
    def __create_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(105, 105, 3), padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),   
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            
            layers.Flatten(),
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.5),
            layers.Dense(46, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def augment_image(self, img):
        augmented_images = []
        img_tensor = tf.convert_to_tensor(img)
        
        # Original image
        augmented_images.append(img)
        
        # Brightness variations
        quantity = 3
        factors = np.linspace(0.7, 1.3, quantity)
        no_of_cropped_images = quantity + 1
        saturations = np.linspace(0.7, 1.3, quantity)

        for factor in factors:
            for saturation in saturations:
                for _ in range(no_of_cropped_images):
                    bright = tf.image.adjust_brightness(img_tensor, delta=factor-1)
                    saturated = tf.image.adjust_saturation(bright, saturation)
                    crop_size = tf.random.uniform([], 80, 105, dtype=tf.int32)
                    cropped = tf.image.random_crop(saturated, [crop_size, crop_size, 3])
                    resized = tf.image.resize(cropped, [105, 105])
                    augmented_images.append(resized.numpy())
                
        return augmented_images

    def __generate_dataset(self, data_loader, should_load=False):
        train_data = []
        train_labels = []
        
        unique_pieces = sorted(list(data_loader.files.keys()))
        piece_to_class = {piece: idx for idx, piece in enumerate(unique_pieces)}

        if should_load:
            # load data from augmented_images
            for piece in unique_pieces:
                files = os.listdir(f"../data/augmented_images/{piece}")
                for file in tqdm(files, desc=f"Loading {piece}"):
                    img = cv.imread(f"../data/augmented_images/{piece}/{file}")
                    if img is not None:
                        img = cv.resize(img, (105, 105))
                        img = img.astype(np.float32) / 255.0
                        train_data.append(img)
                        train_labels.append(piece_to_class[piece])

            return train_data, train_labels
        
        
        for piece, files in tqdm(data_loader.files.items(), desc="Loading data"):
            if not os.path.exists(f"../data/augmented_images/{piece}"):
                os.makedirs(f"../data/augmented_images/{piece}")
            else:
                # empty the folder
                for file in os.listdir(f"../data/augmented_images/{piece}"):
                    os.remove(f"../data/augmented_images/{piece}/{file}")


            if len(files) >= 7:
                for index, file in enumerate(files[:7]):
                    img = cv.imread(file)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    if img is not None:
                        img = cv.resize(img, (105, 105))
                        img = img.astype(np.float32) / 255.0

                        train_data.append(img)
                        train_labels.append(piece_to_class[piece])
                        
                        img = tf.keras.utils.save_img(f"../data/augmented_images/{piece}/{index}.png", img)

                files = files[:7]

            for file in tqdm(files, desc=f"Loading {piece}"):
                img = cv.imread(file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if img is not None:
                    img = cv.resize(img, (105, 105))
                    img = img.astype(np.float32) / 255.0
                    
                    # Generate augmented images
                    augmented = self.augment_image(img)
                    train_data.extend(augmented)
                    train_labels.extend([piece_to_class[piece]] * len(augmented))

                    images_number = len(os.listdir(f"../data/augmented_images/{piece}"))
                    for idx, img in enumerate(augmented):
                        idx = images_number + idx
                        img = tf.keras.utils.save_img(f"../data/augmented_images/{piece}/{idx}.png", img)

        return train_data, train_labels
    

    def train(self, data_loader, epochs=50, batch_size=32):
        
        train_data, train_labels = self.__generate_dataset(data_loader, should_load=True)
        
        X_train = np.array(train_data)
        y_train = np.array(train_labels)
        
        # Shuffle the data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Create validation split
        split = int(0.8 * len(X_train))
        X_val = X_train[split:]
        y_val = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history

    def predict(self, img):
        img = cv.resize(img, (105, 105))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        return GameModel.available_pieces()[class_idx]
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)

if __name__ == "__main__":
    model = CNNModel()

    data_loader = DataLoader("../data/cnn/train")
    try:
        model.train(data_loader, epochs=200, batch_size=32)
    except KeyboardInterrupt:
        print("Training interrupted")
    model.model.save("../models/cnn_model")
    print("Model saved")

    print("Testing model")
    model.load("../models/cnn_model")
    print(model.model.summary())
    acc = 0
    total = 0 
    for i in GameModel.available_pieces():
        for test_image in DataLoader("../data/cnn/train").files[i]:
            img = cv.imread(test_image)
            prediction = model.predict(img)
            if prediction == i:
                acc += 1
            total += 1
    print(f"Accuracy: {acc/total:.2f}")

    print("Validate model")
    model.load("../models/cnn_model")
    acc = 0
    total = 0 
    data_loader = DataLoader("../data/cnn/validation")
    for i in GameModel.available_pieces():
        if data_loader.files[i] == []:
            continue
        for test_image in data_loader.files[i]:
            img = cv.imread(test_image)
            prediction = model.predict(img)
            if prediction == i:
                acc += 1
            print(f"Expected: {i} -- Predicted: {prediction}")
            total += 1
    print(f"Accuracy: {acc/total:.2f}")





    
