import argparse
import tensorflow as tf
import cv2
import numpy as np
import os
import re
from prnet import ResFcn256
from scipy.spatial import cKDTree

def preprocess_image(image_path):
    """Preprocess the input image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float32) / 256.0 / 1.1
    img = np.expand_dims(img, axis=0)
    return img

def infer(model, image_path):
    """Perform inference using the model on the given image."""
    processed_img = preprocess_image(image_path)
    predictions = model(processed_img, training=False)  # Get model predictions
    return predictions

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

def get_epochs_from_checkpoints(checkpoint_dir):
    epochs = []
    for filename in os.listdir(checkpoint_dir):
        match = re.match(r'model_epoch_(\d+).index', filename)
        if match:
            epochs.append(int(match.group(1)))
    return epochs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform inference with trained models.')
    parser.add_argument('checkpoint_dir', type=str, help='Path to the checkpoint directory.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()

    # Load the model
    model = ResFcn256(256, 512)

    # Load and preprocess label
    label = np.load(args.image_path + '.npz')[list(np.load(args.image_path + '.npz').keys())[0]]
    label = cv2.resize(label, (512, 512))  # Resize label to 512x512
    label = label.reshape((512, 512, 1))
    label_array = np.array(label, dtype=np.float32)
    label_array = label_array / np.max(label_array)

    print(f"Label shape: {label_array.shape}, Max value in label: {np.max(label_array)}, Min value in label: {np.min(label_array)}")

    # Load the weight map for the specific image
    weight_map = cv2.imread("model_config/weighted_map.png", cv2.IMREAD_GRAYSCALE)
    weight_map = cv2.resize(weight_map, (512, 512))
    weight_map = np.expand_dims(weight_map, axis=-1)
    weight_map = weight_map / np.max(weight_map)

    print(f"Weight map shape: {weight_map.shape}, Max value in weight map: {np.max(weight_map)}")

    # Load model weights
    try:
        model.load_weights(args.weights_path)
        print(f"Loaded weights from: {args.weights_path}")
    except Exception as e:
        print("Error loading weights:", e)
        return

    prediction = infer(model, args.image_path)

    mse = tf.square(label_array - prediction)
    weight_map_float32 = tf.cast(weight_map, dtype=tf.float32)
    weighted_mse = tf.multiply(mse, weight_map_float32)
    loss = tf.reduce_mean(weighted_mse)

    reconstructed_gt = convert_to_point_cloud(label_array)
    reconstructed_prediction = convert_to_point_cloud(prediction[0])
    tree_gt = cKDTree(reconstructed_gt)

    distances, _ = tree_gt.query(reconstructed_prediction)

    nme = np.mean(distances)

    print(f"Loss: {loss}")
    print(f"Validaiton NME%: {nme * 100}")

    # Save the results
    output_filename = os.path.join('predictions', f'{args.image_path.split("/")[-1]}_prediction.npy')
    os.makedirs('predictions', exist_ok=True)
    np.save(output_filename, prediction)
    print(f'Saved prediction to {output_filename}.')

if __name__ == '__main__':
    main()
