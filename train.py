import numpy as np
import os
import argparse
import tensorflow as tf
import cv2
import random
from prnet import ResFcn256 # Ensure this is compatible with TensorFlow 2.x
import math
from datetime import datetime

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
                label_array = np.array(label, dtype=np.float32) / 256 / 1.1  # Normalize if necessary

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

def main(args):
    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        raise SystemError("GPU device not found")
    print(f"GPUs available: {gpus}")

    try:
        # Set the GPU to be used
        tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
    except RuntimeError as e:
        # Catch the runtime error if GPU setup fails
        print(e)
        raise


    # Load and prepare data
    data = TrainData(args.train_data_file, validation_split=args.validation_split)

    # Build the model
    model = ResFcn256(256, 512)

    # Load weights if they exist
    checkpoint_path = os.path.join(args.checkpoint, 'resfcn256')
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Define the loss function and the optimizer
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Load the weight map
    weight_map = cv2.imread("weighted_map.png", cv2.IMREAD_GRAYSCALE)
    weight_map = cv2.resize(weight_map, (512, 512))
    weight_map = np.expand_dims(weight_map, axis=-1)
    weight_map = weight_map / np.max(weight_map)

    # Define training and validation steps
    @tf.function
    def train_step(images, labels):
        if images.shape[0] == 0:
            print("Skipping empty batch")
            return 0.0  # Return a default loss value

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            # Calculate the error (shape: [batch_size, 512, 512, 1])
            error = labels - predictions

            # Convert weight_map to float32 and ensure it has the right shape
            weight_map_float32 = tf.cast(weight_map, tf.float32)  # Shape: [512, 512, 1]

            # Expand dimensions of weight_map to match the batch size
            weight_map_expanded = tf.expand_dims(weight_map_float32, axis=0)  # Shape: [1, 512, 512, 1]
            weight_map_batch = tf.tile(weight_map_expanded, [tf.shape(images)[0], 1, 1, 1])  # Shape: [batch_size, 512, 512, 1]

            # Apply the weight map to the error
            weighted_error = error * weight_map_batch

            # Compute the mean squared error across spatial dimensions
            mse = tf.reduce_mean(tf.square(weighted_error), axis=[1, 2, 3])  # Shape: [batch_size]

            # Compute the mean of mse across the batch
            loss = tf.reduce_mean(mse)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss




    @tf.function
    def validation_step(images, labels):
        predictions = model(images, training=False)
        # Calculate the mean squared error for each element
        mse = tf.square(labels - predictions)
        # Apply the weight map
        weighted_mse = mse * weight_map
        # Calculate the mean over the batch
        v_loss = tf.reduce_mean(weighted_mse)
        return v_loss


    # Training loop
    for epoch in range(args.epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        for _ in range(math.ceil(data.num_data / args.batch_size)):
            batch = data(args.batch_size)
            loss_value = train_step(batch[0], batch[1])
            epoch_loss_avg.update_state(loss_value)

        for _ in range(math.ceil(data.num_validation_data / args.batch_size)):
            val_batch = data(args.batch_size, is_validation=True)
            val_loss_value = validation_step(val_batch[0], val_batch[1])
            epoch_val_loss_avg.update_state(val_loss_value)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result()}, Validation Loss: {epoch_val_loss_avg.result()}")

        model.save_weights(checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spherical Position Map Regression Network for Accurate 3D Facial Geometry Estimation')
    parser.add_argument('--train_data_file', default='', type=str, help='The training data file')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='The learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='Total epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch sizes')
    parser.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    parser.add_argument('--model_path', default='checkpoint/resfcn256_weight', type=str, help='The path of pretrained model')
    parser.add_argument('--gpu', default='0', type=str, help='The GPU ID')
    parser.add_argument('--validation_split', default=0.2, type=float, help='Fraction of data to be used for validation')

    args = parser.parse_args()
    main(args)