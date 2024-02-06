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
        label_array = np.array(label, dtype=np.float32) / np.max(label_array)

        return img_array, label_array
    except Exception as e:
        print(f"Error processing file pair: {img_path}, {label_path}: {e}")
        return None, None

def create_error_mask(model, data):
    error_accumulator = np.zeros((512, 512))  # Assuming labels are resized to 512x512

    for img_path, label_path in data.train_data_list:
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"Warning: Skipping missing file pair: {img_path}, {label_path}")
            continue

        img, label = load_image_label(img_path, label_path)
        if img is None or label is None:
            continue

        prediction = model.predict(np.expand_dims(img, axis=0))
        error = np.abs(prediction[0] - label)
        error_accumulator += error.squeeze()

    error_mask = error_accumulator / len(data.train_data_list)
    error_mask_normalized = error_mask / np.max(error_mask)
    cv2.imwrite("error_mask.png", error_mask_normalized * 255)  # Save as an image

if __name__ == '__main__':
    model = ResFcn256(256, 512)
    checkpoint_path = 'path_to_checkpoint'
    model.load_weights(checkpoint_path)

    data_file = 'path_to_train_data_file'
    data = TrainData(data_file)
    create_error_mask(model, data)
