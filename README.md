# Handwritten-Digit-Recognition-using-CNN-MNIST-
🖊️ Handwritten Digit Recognition using CNN (MNIST)

This project builds a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from the popular MNIST dataset.
The model is trained for 2 epochs, evaluated on test data, and visualizes predictions for sample images.

📌 Project Features

Loads MNIST dataset (60,000 training + 10,000 testing images)

Preprocesses images (reshaping + normalization)

Builds a deep CNN using TensorFlow/Keras

Trains the model for 2 epochs

Evaluates test accuracy

Displays predictions for sample images

Saves the trained model as mnist_cnn_model.h5

📁 Requirements

Install the required libraries:

pip install tensorflow numpy matplotlib


Make sure your TensorFlow installation supports tensorflow.keras.

📜 Code Overview
1. Load & Preprocess Data

MNIST is loaded from TensorFlow datasets.

Images are reshaped to (28, 28, 1) for CNN input.

Pixel values are normalized to range 0–1.

Labels are one-hot encoded.

2. CNN Model Architecture

The model contains:

3 Convolution layers (32, 64, 64 filters)

2 MaxPooling layers

Flatten layer

Dense layer (64 units)

Output layer (10 units with softmax)

3. Training

The model is compiled with:

Adam optimizer

Categorical crossentropy loss

Accuracy metric

Training is done for 2 epochs with 10% validation split.

4. Evaluation

The model is evaluated on the test set and prints accuracy.

5. Visualization

Displays first 5 test images with:

Predicted label

Actual label

6. Saving Model

The trained model is saved as:

mnist_cnn_model.h5

▶️ Running the Project

Run the script:

python mnist_cnn.py


(or whatever filename you saved)

📷 Sample Output (Example)
Test accuracy: 0.9765


Five plots will appear showing prediction vs actual label.

💾 Model File
mnist_cnn_model.h5


After training, a model file is created:

mnist_cnn_model.h5
