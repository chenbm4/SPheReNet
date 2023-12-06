import os
import math
import random
import json
import numpy as np
import tensorflow as tf
import cv2
import optuna
from prnet import ResFcn256
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError


# Configuration Management
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Data Handling Class
class TrainData(object):
    def __init__(self, train_data_file, validation_split=0.2):
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.validation_data_list = []
        self.validation_split = validation_split
        self.index = 0
        self.num_data = 0
        self.num_validation_data = 0
        self.readTrainData()
        self.split_train_validation()

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

    def split_train_validation(self):
        num_validation = int(self.validation_split * self.num_data)
        self.validation_data_list = self.train_data_list[:num_validation]
        self.train_data_list = self.train_data_list[num_validation:]
        self.num_validation_data = len(self.validation_data_list)
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

    def __call__(self, batch_num, is_validation=False):
        data_list = self.validation_data_list if is_validation else self.train_data_list
        num_data = len(data_list)

        if (self.index + batch_num) > num_data:
            self.index = 0
            random.shuffle(data_list)

        batch_list = data_list[self.index:(self.index + batch_num)]
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

# Training Step
@tf.function
def train_step(model, optimizer, weight_map, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = weighted_mse(weight_map, labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Validation Step
@tf.function
def validation_step(model, weight_map, images, labels):
    predictions = model(images, training=False)
    loss = weighted_mse(weight_map, labels, predictions)
    return loss

# Training Loop
def train_model(model, optimizer, weight_map, train_data, args, trial=None):
    best_val_loss = float('inf')

    # Early stopping callback
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)

    # Metrics for logging
    train_loss_results = []
    val_loss_results = []
    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()

    for epoch in range(args.epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        mse_metric.reset_states()
        mae_metric.reset_states()

        for _ in range(math.ceil(train_data.num_data / args.batch_size)):
            batch = train_data(args.batch_size)
            loss_value = train_step(model, optimizer, weight_map, batch[0], batch[1])
            epoch_loss_avg.update_state(loss_value)
            mse_metric.update_state(batch[1], model(batch[0], training=True))
            mae_metric.update_state(batch[1], model(batch[0], training=True))

        for _ in range(math.ceil(train_data.num_validation_data / args.batch_size)):
            val_batch = train_data(args.batch_size, is_validation=True)
            val_loss_value = validation_step(model, weight_map, val_batch[0], val_batch[1])
            epoch_val_loss_avg.update_state(val_loss_value)

        train_loss = epoch_loss_avg.result()
        val_loss = epoch_val_loss_avg.result()
        train_mse = mse_metric.result()
        train_mae = mae_metric.result()

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Validation Loss: {val_loss}, MSE: {train_mse}, MAE: {train_mae}")

        train_loss_results.append(train_loss)
        val_loss_results.append(val_loss)

        # Early Stopping Check
        early_stopping_callback.on_epoch_end(epoch, logs={'val_loss': val_loss})
        if early_stopping_callback.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Optuna Pruning Check
        if trial and trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model weights manually
            model.save_weights(os.path.join('checkpoints', 'ResFcn256_20231005', 'best_model.h5'))

    return best_val_loss  # Return the best validation loss for Optuna optimization

# Main Function
def main(config_path):
    args = load_config(config_path)
    data = TrainData(args['train_data_file'], args['validation_split'])

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'])
    weight_map = load_weight_map(args['weight_map_path'])

    train_model(model, optimizer, weight_map, data, args)

# Load Weight Map
def load_weight_map(weight_map_path):
    weight_map = cv2.imread(weight_map_path, cv2.IMREAD_GRAYSCALE)
    weight_map = cv2.resize(weight_map, (512, 512))
    weight_map = np.expand_dims(weight_map, axis=-1)
    weight_map = weight_map / np.max(weight_map)
    return tf.cast(weight_map, tf.float32)

if __name__ == '__main__':
    main('config.json')