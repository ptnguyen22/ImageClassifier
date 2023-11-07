# ImageClassifier
A Python machine learning program using the CIFAR-10 dataset (Python version) to classify 32x32 images into 10 different classifications. The dataset is divided into five training batches and one test batch, each with 10000 images.
Results in a ~73% accuracy rate.

This model is a convolution neural network. The CNN architecture consisted of three convolutional layers followed by max-pooling layers to capture hierarchical features from the input images. The convolutional layers used Rectified Linear. Unit (ReLU) activation functions to introduce non-linearity. Dropout layers were incorporated after the convolutional layers to prevent overfitting. Two fully connected layers were added at the end, followed by a softmax activation function to produce class probabilities for the 10 categories.

Built with python3 on MacOS and Ubuntu

Dependencies:
Tensorflow
NumPy
Matplotlib
