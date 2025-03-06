# AI-ML
Project Overview
This project implements a fully connected neural network from scratch using NumPy and Pandas to classify images in the CIFAR-10 dataset into 10 categories. The goal is to build a simple deep learning model without using TensorFlow or PyTorch.

Technologies Used
Python
NumPy
Pandas
Pickle (for loading dataset)
Google Colab (for training and testing)
Implementation Details
Dataset Handling:

The CIFAR-10 dataset is downloaded and extracted automatically.
Data is normalized by dividing pixel values by 255.
One-hot encoding is applied to labels.
Neural Network Architecture:

Input layer: 3072 neurons (for 32x32 images with 3 channels).
Hidden layer: 256 neurons with ReLU activation.
Output layer: 10 neurons with softmax activation.
Training Process:

Uses forward propagation with ReLU and softmax.
Backpropagation applies gradient descent to optimize weights.
Loss function: Cross-entropy.
Prediction:

The trained model predicts labels for given images using the highest probability output.
Usage Instructions
Run the script in Google Colab.
The model will download and preprocess CIFAR-10.
Training progress is displayed every 10 epochs.
The script prints final predictions for a sample of images.
