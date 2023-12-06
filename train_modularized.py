import os
import random
import json
import numpy as np
import tensorflow as tf
import cv2
import optuna
from prnet import ResFcn256
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.model_selection import train_test_split

# Configuration Management
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Data Handling Class
class TrainData(object):
    def __init__(self, train_data_file):
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.index = 0
        self.num_data = 0
        self.readTrainData()

    def readTrainData(self):
        with open(self.train_data_file, 'r') as fp:
            temp = fp.readlines()
            for line in temp:
                items = line.strip().split()  # Split line into items
                if len(items) != 2:  # Check if there are exactly two items
                    print(f"Warning: Skipping invalid line: {line.strip()}")
                    continue
                self.train_data_list.append(items)

            random.shuffle(self.train_data_list)
            self.num_data = len(self.train_data_list)

    def getBatch(self, batch_list):
        imgs = []
        labels = []
        for item in batch_list:
            img_path, label_path = item

            if not os.path.exists(img_path) or not os.path.exists(label_path):
                print(f"Warning: Skipping missing file pair: {img_path}, {label_path}")
                continue  # Skip this iteration if file not found

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (256, 256))  # Resize image to 256x256

                # Load labels
                if label_path.endswith('.npy'):
                    label = np.load(label_path)
                elif label_path.endswith('.npz'):
                    label_data = np.load(label_path)
                    label = label_data[list(label_data.keys())[0]]
                label = cv2.resize(label, (512, 512))  # Resize label to 512x512
                label = label.reshape((512, 512, 1))  # Ensure single channel

                img_array = np.array(img, dtype=np.float32) / 256.0 / 1.1
                label_array = np.array(label, dtype=np.float32)
                label_array = label_array / np.max(label_array)  # Normalize to 0 to 1 values

                imgs.append(img_array)
                labels.append(label_array)
            except Exception as e:
                print(f"Error processing file pair: {img_path}, {label_path}: {e}")
                continue  # Skip this iteration if there is an error

        return np.array(imgs), np.array(labels)

    def __call__(self, batch_num):
        num_data = len(self.train_data_list)

        if (self.index + batch_num) > num_data:
            self.index = 0
            random.shuffle(self.train_data_list)

        batch_list = self.train_data_list[self.index:(self.index + batch_num)]
        batch_data = self.getBatch(batch_list)
        self.index += batch_num
        return batch_data

# Model Building
def build_model():
    model = ResFcn256(256, 512)
    return model

# Loss Function with Weight Map
def weighted_mse(weight_map, labels, predictions):
    error = labels - predictions
    weighted_error = error * weight_map
    mse = tf.reduce_mean(tf.square(weighted_error), axis=[1, 2, 3])
    return tf.reduce_mean(mse)

# Custom Callback for Early Stopping
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience):
        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_val_loss = float('inf')
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                self.model.set_weights(self.best_weights)
                self.stopped_epoch = epoch
                self.model.stop_training = True

# Main Function
def main(config_path, trial=None):
    args = load_config(config_path)

    # Load data and split into train and validation sets
    data = TrainData(args['train_data_file'])
    X_train, X_val = train_test_split(data.train_data_list, test_size=args['validation_split'], random_state=42)

    # Define model
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'])
    weight_map = load_weight_map(args['weight_map_path'])

    # Define custom early stopping callback
    early_stopping = EarlyStoppingCallback(patience=args['patience'])

    # Define model checkpoint callback to save the best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('checkpoints', 'ResFcn256_20231005', 'best_model.h5'),
        monitor='val_loss', save_best_only=True)

    # Define callbacks for model.fit
    callbacks = [early_stopping, model_checkpoint]

    # Train the model using model.fit
    history = model.fit(
        batch_size=args['batch_size'],
        x=data,
        epochs=args['epochs'],
        validation_data=X_val,
        callbacks=callbacks,
        verbose=1)

    # Return the best validation loss for Optuna optimization
    return min(history.history['val_loss'])

if __name__ == '__main__':
    main('model_config/config.json')