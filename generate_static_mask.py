import numpy as np
import cv2
import tensorflow as tf
from prnet import ResFcn256
from train import TrainData
import os

def load_image_label(img_path, label_path):
    try:
        # Load and process image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img_array = np.array(img, dtype=np.float32) / 256.0 / 1.1

        # Load and process label
        if label_path.endswith('.npy'):
            label = np.load(label_path)
        elif label_path.endswith('.npz'):
            label_data = np.load(label_path)
            label = label_data[list(label_data.keys())[0]]
        label = cv2.resize(label, (512, 512))
        label = label.reshape((512, 512, 1))
        label_array = np.array(label, dtype=np.float32)
        label_array = label_array / np.max(label_array)

        return img_array, label_array
    except Exception as e:
        print(f"Error processing file pair: {img_path}, {label_path}: {e}")
        return None, None

def create_error_mask(data):
    error_accumulator = np.zeros((512, 512))  # Assuming labels are resized to 512x512
    prediction_count = 0

    for img_path, label_path in data.train_data_list:
        prediction_file = os.path.join("predictions", os.path.basename(img_path) + "_prediction.npz")

        if not os.path.exists(prediction_file) or not os.path.exists(label_path):
            print(f"Warning: Missing prediction or label file for: {img_path}")
            continue

        label = load_label(label_path)
        if label is None:
            continue

        prediction = np.load(prediction_file)['arr_0'][0]  # Load prediction

        error = np.abs(prediction - label)
        error_accumulator += error.squeeze()
        prediction_count += 1

    if prediction_count == 0:
        raise ValueError("No predictions were processed. Check your files.")

    error_mask = error_accumulator / prediction_count
    error_mask_normalized = error_mask / np.max(error_mask)
    cv2.imwrite("error_mask.png", error_mask_normalized * 255)  # Save as an image

def load_label(label_path):
    try:
        if label_path.endswith('.npy'):
            label = np.load(label_path)
        elif label_path.endswith('.npz'):
            label_data = np.load(label_path)
            label = label_data[list(label_data.keys())[0]]
        label = cv2.resize(label, (512, 512))
        label = label.reshape((512, 512, 1))
        label_array = np.array(label, dtype=np.float32)
        label_array = label_array / np.max(label_array)

        return label_array
    except Exception as e:
        print(f"Error processing label file: {label_path}: {e}")
        return None

if __name__ == '__main__':
    data_file = 'train_data_file.txt'
    data = TrainData(data_file)
    print(f"Number of training data: {data.num_data}")

    create_error_mask(data)
