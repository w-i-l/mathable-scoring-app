import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
from game_model import GameModel
from .cnn_dataset import DataSet
from .cnn_data_loader import DataLoader

class CNNModel:
    def __init__(self):
        self.model = self.__create_model()
        
    
    def __create_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(DataSet.image_size, DataSet.image_size, 3), padding='same'),
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
    

    def train(self, dataset, should_load=False, should_save=False, epochs=50, batch_size=32):
        
        train_data, train_labels = dataset.generate_dataset(should_load, should_save)
        X_train, y_train, X_val, y_val = dataset.split_dataset(train_data, train_labels)
        
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
        img = cv.resize(img, (DataSet.image_size, DataSet.image_size))
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
    dataset = DataSet(data_loader)

    try:
        print(model.model.summary())
        model.train(dataset, should_load=False, should_save=True, epochs=200, batch_size=128)
    except KeyboardInterrupt:
        print("Training interrupted")
    model.model.save("../models/cnn_model")
    print("Model saved")

    print("Testing model")
    model.load("../models/cnn_model")
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
            else:
                print(f"Expected: {i} -- Predicted: {prediction}")
            total += 1
    print(f"Accuracy: {acc/total:.2f}")





    
