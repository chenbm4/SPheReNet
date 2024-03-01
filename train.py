import numpy as np
import os
import argparse
import tensorflow as tf
import cv2
import random
from prnet import ResFcn256 # Ensure this is compatible with TensorFlow 2.x
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.spatial import cKDTree

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
                # print(f"Label shape: {label_array.shape}, Max value in label: {np.max(label_array)}, Min value in label: {np.min(label_array)}")

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
    print(f"Number of training data: {data.num_data}")

    # Build the model
    model = ResFcn256(256, 512)

    dummy_input = tf.random.normal([1, 256, 256, 3])
    model(dummy_input)

    best_checkpoint_path = os.path.join(args.checkpoint, 'best')
    recent_checkpoint_path = os.path.join(args.checkpoint, 'recent')

    # Load weights if they exist
    if os.path.exists(recent_checkpoint_path):
        model.load_weights(recent_checkpoint_path)
        print(f"Loaded weights from the most recent checkpoint: {recent_checkpoint_path}")
    else:
        print("No recent checkpoint found. Starting training from scratch.")

    # Load the weight map
    weight_map = cv2.imread("model_config/weighted_map.png", cv2.IMREAD_GRAYSCALE)
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


    # @tf.function
    def validation_step(images, labels):
        # Function to convert posmap to 3D point cloud
        def convert_to_point_cloud(posmap):
            # Ensure posmap is two-dimensional
            if posmap.ndim == 3 and posmap.shape[2] == 1:
                posmap = posmap[:, :, 0]  # Convert 3D tensor to 2D by removing the channel dimension

            phi_values = (np.arange(512) / (512 - 1)) * np.pi - np.pi / 2
            y_max = 512
            y_min = 0
            y_values = y_max - (np.arange(512) / (512 - 1)) * (y_max - y_min)
            phi_grid, y_grid = np.meshgrid(phi_values, y_values)
            
            valid_indices = posmap != 0
            r_flat = posmap[valid_indices]
            phi_flat = phi_grid[valid_indices]
            y_flat = y_grid[valid_indices]
            
            x = r_flat * np.sin(phi_flat)
            z = r_flat * np.cos(phi_flat)
            point_cloud = np.vstack((x, y_flat, z)).T

            valid_indices = r_flat != 0
            return point_cloud

        # Initialize the NME calculation
        predictions = model(images, training=False)
        total_nme = 0
        count = 0

        for i in range(predictions.shape[0]):
            true_posmap = labels[i]
            predicted_posmap = predictions[i]
            
            # Convert ground truth and predictions to point clouds
            reconstructed_gt = convert_to_point_cloud(true_posmap)
            reconstructed_prediction = convert_to_point_cloud(predicted_posmap)

            # Create KD-Trees for efficient nearest neighbor search
            tree_gt = cKDTree(reconstructed_gt)

            # For each point in one cloud, find the nearest in the other
            distances, _ = tree_gt.query(reconstructed_prediction)
            me = np.mean(distances) * 100
            total_nme += me
            count += 1

        # Average NME over the batch
        avg_me = total_nme / count if count > 0 else 0
        return avg_me

    # Early stopping setup
    best_val_loss = float('inf')
    early_stopping_patience = 5
    early_stopping_counter = 0

    # Learning rate decay
    initial_learning_rate = args.learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=5*math.ceil(data.num_data / args.batch_size),
        decay_rate=0.9,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    training_loss_history = []
    validation_loss_history = []

    # Perform initial validation before starting the training loop
    print("Performing initial validation...")
    initial_val_batch = data(args.batch_size, is_validation=True)
    initial_val_loss = validation_step(initial_val_batch[0], initial_val_batch[1])
    print(f"Initial Validation Loss: {initial_val_loss}")

    # Training loop
    for epoch in range(args.epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        # Training step
        for _ in range(math.ceil(data.num_data / args.batch_size)):
            batch = data(args.batch_size)
            loss_value = train_step(batch[0], batch[1])
            epoch_loss_avg.update_state(loss_value)

        # Validation loop with tqdm progress bar
        num_validation_steps = math.ceil(data.num_validation_data / args.batch_size)
        for _ in tqdm(range(num_validation_steps), desc=f'Epoch {epoch+1} Validation'):
            val_batch = data(args.batch_size, is_validation=True)
            val_loss_value = validation_step(val_batch[0], val_batch[1])
            epoch_val_loss_avg.update_state(val_loss_value)

        # Get the average losses
        train_loss = epoch_loss_avg.result()
        val_loss = epoch_val_loss_avg.result()

        print(f"Epoch {epoch+1}, Loss: {train_loss}, Validation Avg NME: {val_loss}")

        # Append losses to history lists
        training_loss_history.append(train_loss.numpy())
        validation_loss_history.append(val_loss.numpy())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            model.save_weights(os.path.join(best_checkpoint_path, 'best_model'))
            print(f"Improved Validation Avg NME: {val_loss}. Best model checkpoint saved.")
        else:
            early_stopping_counter += 1
            print(f"No improvement in Validation Avg NME for {early_stopping_counter} epoch(s)")
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Checkpointing every epoch
        model.save_weights(os.path.join(recent_checkpoint_path, f'model_epoch_{epoch+1}'))
        print(f"Checkpoint for epoch {epoch+1} saved.")
    
        # Convert the loss history from NumPy float32 to native Python float
        training_loss_history = [float(loss) for loss in training_loss_history]
        validation_loss_history = [float(loss) for loss in validation_loss_history]

        # Save loss history to a file
        loss_history = {
            'training_loss': training_loss_history,
            'validation_nme': validation_loss_history
        }
        with open('loss_history.json', 'w') as f:
            json.dump(loss_history, f)
    
        # Plot loss history
        plt.figure(figsize=(10, 5))
        plt.plot(training_loss_history, label='Training Loss')
        plt.plot(validation_loss_history, label='Validation Avg NME')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_history_plot.png')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spherical Position Map Regression Network for Accurate 3D Facial Geometry Estimation')
    parser.add_argument('--train_data_file', default='train_data_file.txt', type=str, help='The training data file')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Total epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
    parser.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    parser.add_argument('--model_path', default='checkpoint/resfcn256_weight', type=str, help='The path of pretrained model')
    parser.add_argument('--gpu', default='0', type=str, help='The GPU ID')
    parser.add_argument('--validation_split', default=0.2, type=float, help='Fraction of data to be used for validation')

    args = parser.parse_args()
    main(args)
    
