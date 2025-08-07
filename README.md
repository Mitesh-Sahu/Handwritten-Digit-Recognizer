Handwritten Digit Recognizer using TensorFlow
Python TensorFlow License

This project is a classic introduction to deep learning and computer vision. It uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits (0-9) from the famous MNIST dataset.

Problem Statement
Manual recognition of handwritten digits is time-consuming and error-prone, especially when processing large volumes of data like scanned forms or checks. This project automates the task by building a deep learning model that can accurately and efficiently classify digits from images.

Key Features
Data Loading & Preprocessing: Loads the MNIST dataset directly from TensorFlow and prepares it for training (normalization, reshaping).
CNN Architecture: Implements a simple yet powerful CNN with Conv2D, MaxPooling2D, and Dense layers.
Model Training: Trains the model on 60,000 labeled images.
Performance Evaluation: Achieves high accuracy on a test set of 10,000 unseen images.
Prediction & Visualization: Predicts digits from new images and visualizes the results.
Technology Stack
Python
TensorFlow / Keras for building and training the neural network.
NumPy for numerical operations.
Matplotlib for visualizing images and results.
Results
The model was trained for 5 epochs and achieved the following performance on the unseen test dataset:

Test Accuracy: ~98.98%
Test Loss: ~0.0323
Here is a screenshot of the final training and evaluation output:
