import argparse
import tensorflow as tf
import cv2
import numpy as np
from prnet import ResFcn256  # Import your model definition

def preprocess_image(image_path):
    """Preprocess the input image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize to the expected input size
    img = np.array(img, dtype=np.float32) / 256.0 / 1.1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def infer(model, image_path):
    """Perform inference using the model on the given image."""
    processed_img = preprocess_image(image_path)
    predictions = model(processed_img, training=False)  # Get model predictions
    return predictions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform inference with a trained model.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights file.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()

    # Load the model
    model = ResFcn256(256, 512)
    
    # Build the model by passing a dummy input
    dummy_input = np.zeros((1, 256, 256, 3))  # Adjust the shape according to your model's input shape
    model(dummy_input, training=False)

    model.load_weights(args.weights_path)  # Load the trained weights

    # Perform inference
    prediction = infer(model, args.image_path)

    # Save prediction to file
    output_filename = args.image_path.rsplit('.', 1)[0] + '_prediction.npy'  # Creates a file name based on the image path
    np.save(output_filename, prediction)
    print(f"Saved prediction to {output_filename}")

if __name__ == '__main__':
    main()
