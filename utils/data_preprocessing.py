import os
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming MNIST_Dataset_Loader is a sibling directory to the project root or handled via sys.path
# For better reusability, we can pass project_root_path as an argument or set it up differently
# For now, let's assume MNIST_Dataset_Loader is accessible from the script's execution context.

# Adjust import based on the project structure to ensure MNIST_Dataset_Loader is found.
# Since it's relative to the main project directory, we need to ensure sys.path is set correctly
# before this module is imported.
# For now, let's assume the calling script (e.g., knn.py) has already set the project_root_path in sys.path.
from MNIST_Dataset_Loader.mnist_loader import MNIST

def load_and_preprocess_mnist_data(project_root_path):
    """
    加载和预处理MNIST数据集，适用于KNN, SVM, RFC等需要扁平化图像数据的方法。
    """
    print('\nLoading MNIST Data...')
    mnist = MNIST(project_root_path)
    img_train, labels_train, _ = mnist.load_training() # Unpack 3 values, discard the third one
    img_test, labels_test, _ = mnist.load_testing()   # Unpack 3 values, discard the third one

    # Reshape image data from 28x28 to a 1D array of 784 pixels
    train_img = np.array(img_train).reshape((len(img_train), -1))
    train_labels = np.array(labels_train)

    test_img = np.array(img_test).reshape((len(img_test), -1))
    test_labels = np.array(labels_test)

    # Features and Labels for training/validation split
    X = train_img
    y = train_labels

    print('\nPreparing Classifier Training and Validation Data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test, test_img, test_labels 