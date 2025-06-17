import numpy as np
import argparse
import cv2
import sys
import os
# from cnn.neural_network import CNN # Moved into run_cnn_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Added for plotting
from sklearn.metrics import confusion_matrix # Added for confusion matrix
import seaborn as sns # Added for heatmap plotting

# Calculate project root path dynamically based on the current script's location
# This path will be used for relative imports and file saving/loading
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_path = os.path.abspath(os.path.join(_current_file_dir, '..'))
if _project_root_path not in sys.path:
    sys.path.append(_project_root_path)

# Import utility functions
from utils.evaluation_metrics import display_prediction_examples, print_classification_report

# Removed global logging.basicConfig - logging will be configured within run_cnn_model
import logging

# Removed argparse parsing from here, it will be handled by main.py
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--save_model", type=int, default=-1)
# ap.add_argument("-l", "--load_model", type=int, default=-1)
# ap.add_argument("-w", "--save_weights", type=str)
# args = vars(ap.parse_args())

def run_cnn_model(project_root_path=_project_root_path, log_file_name=None, save_model=0, load_model=0, save_weights=None):
    """
    Runs the Convolutional Neural Network model for handwritten digit recognition.

    Args:
        project_root_path (str): The absolute path to the project root directory.
        log_file_name (str): The name of the log file, relative to the model's directory. (Optional, defaults to Results/CNN/summary.log)
        save_model (int): Set to 1 to save the trained model weights. Default is 0.
        load_model (int): Set to 1 to load pre-trained model weights. Default is 0.
        save_weights (str): The path to save/load model weights (e.g., cnn_weights.hdf5).
    """
    # Define the output directory for CNN results
    cnn_results_dir = os.path.join(project_root_path, 'Results', 'CNN')
    os.makedirs(cnn_results_dir, exist_ok=True)

    # Configure logging for this specific run
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    if log_file_name is None:
        log_file_path = os.path.join(cnn_results_dir, 'summary.log')
    else:
        log_file_path = os.path.join(cnn_results_dir, log_file_name)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logging.info("\n--- Running Convolutional Neural Network Model ---")
    # old_stdout = sys.stdout # Removed - replaced by logging
    # current_log_file = os.path.join(_current_file_dir, log_file_name) # Removed - replaced by logging
    # log_file = open(current_log_file, "w") # Removed - replaced by logging
    # sys.stdout = log_file # Removed - replaced by logging

    try:
        # Local imports for CNN specific components
        from MNIST_Dataset_Loader.mnist_loader import MNIST
        # Dynamically add CNN_Keras to sys.path to import cnn.neural_network
        cnn_keras_path = os.path.join(project_root_path, 'models', 'CNN_Keras')
        logging.debug(f"project_root_path: {project_root_path}")
        logging.debug(f"Calculated cnn_keras_path: {cnn_keras_path}")
        if cnn_keras_path not in sys.path:
            sys.path.append(cnn_keras_path)
        logging.debug(f"sys.path after adding cnn_keras_path: {sys.path}")
        from cnn.neural_network import CNN

        # Read/Download MNIST Dataset
        logging.info('\nLoading MNIST Dataset...')
        mnist_loader = MNIST(project_root_path)
        mnist_data_train, mnist_labels_train, _ = mnist_loader.load_training()
        mnist_data_test, mnist_labels_test, _ = mnist_loader.load_testing()

        # Combine train and test data for a single split as original code did with fetch_openml
        mnist_data = np.concatenate((mnist_data_train, mnist_data_test), axis=0)
        mnist_labels = np.concatenate((mnist_labels_train, mnist_labels_test), axis=0)

        # Reshape and add channel dimension for NHWC format (Height, Width, Channels)
        mnist_data = mnist_data[:, :, :, np.newaxis]

        # Divide data into testing and training sets.
        train_img, test_img, train_labels, test_labels = train_test_split(mnist_data, mnist_labels, test_size=0.1, random_state=42)

        # Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
        total_classes = 10
        train_labels = to_categorical(train_labels, total_classes)
        test_labels = to_categorical(test_labels, total_classes)

        logging.info('\nCNN Model Parameters:')
        logging.info(f'  Input Shape: (Width={28}, Height={28}, Depth={1})')
        logging.info(f'  Total Classes: {total_classes}')
        logging.info(f'  Optimizer: SGD(learning_rate=0.01, momentum=0.9, nesterov=True)')

        logging.info('\n Compiling model...')
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        
        # Define full_weights_path relative to cnn_results_dir
        full_weights_path = os.path.join(cnn_results_dir, save_weights) if save_weights else None
        clf = CNN.build(width=28, height=28, depth=1, total_classes=total_classes, Saved_Weights_Path=full_weights_path if load_model > 0 else None)
        clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        b_size = 256
        num_epoch = 20
        verb = 0

        history = None

        # If weights not loaded, train the model
        if load_model < 0:
            logging.info('\nTraining the Model...')
            history = clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch, verbose=verb)

        logging.info('\nEvaluating Accuracy and Loss Function...')
        loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=256, verbose=0)
        logging.info('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

        # Convert one-hot encoded test_labels back to single labels for classification_report and confusion matrix
        test_labels_decoded = np.argmax(test_labels, axis=1)
        # Get predictions for all test images
        test_predictions = clf.predict(test_img)
        test_predictions_decoded = np.argmax(test_predictions, axis=1)

        # Generate and print Confusion Matrix
        logging.info('\nCreating Confusion Matrix for Test Data...')
        conf_mat_cnn = confusion_matrix(test_labels_decoded, test_predictions_decoded)
        logging.info('Confusion Matrix:\n' + str(conf_mat_cnn))

        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat_cnn, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix for CNN Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cnn_conf_mat_save_path = os.path.join(cnn_results_dir, 'CNN_Confusion_Matrix.png')
        plt.savefig(cnn_conf_mat_save_path)
        logging.info(f"CNN Confusion matrix plot saved to {cnn_conf_mat_save_path}")
        plt.clf()
        plt.close()

        # Generate and print Classification Report for the test data using utility function
        print_classification_report(test_labels_decoded, test_predictions_decoded, title="CNN Test Data ")

        # Plot training history (Loss and Accuracy)
        if history: # Only plot if training actually occurred
            plt.figure(figsize=(12, 5))
            
            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            cnn_history_plot_save_path = os.path.join(cnn_results_dir, 'CNN_Loss_Accuracy_Curves.png')
            plt.savefig(cnn_history_plot_save_path)
            logging.info(f"CNN Loss and Accuracy curves plot saved to {cnn_history_plot_save_path}")
            plt.clf()
            plt.close()

        # Save the pre-trained model.
        if save_model > 0:
            logging.info('Saving weights to file...')
            if full_weights_path: # Ensure full_weights_path is not None
                # Ensure the directory for weights exists before saving
                os.makedirs(os.path.dirname(full_weights_path), exist_ok=True)
                clf.save_weights(full_weights_path, overwrite=True)
            else:
                logging.warning("save_weights path not provided, cannot save model.")

        # Display prediction examples using utility function
        # Pass img_shape with channel dimension for CNN images
        cnn_examples_save_path = os.path.join(cnn_results_dir, 'CNN_Prediction_Examples.png')
        display_prediction_examples(test_img, test_labels, test_predictions_decoded, img_shape=(28, 28, 1), save_path=cnn_examples_save_path)
    finally:
        # Ensure handlers are removed to prevent issues with subsequent runs or other modules
        for handler in root_logger.handlers[:]: # Iterate over a copy to allow modification
            root_logger.removeHandler(handler)
        # sys.stdout = old_stdout # Removed - replaced by logging
        # log_file.close() # Removed - replaced by logging

# Optional: if you want to run this file directly for testing, keep this block
if __name__ == "__main__":
    # When run directly, project_root_path is determined based on current file's location
    # For direct run, we need to manually parse args for CNN.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_model", type=int, default=-1)
    ap.add_argument("-l", "--load_model", type=int, default=-1)
    ap.add_argument("-w", "--save_weights", type=str, default="cnn_weights.hdf5")
    args = vars(ap.parse_args())
    
    run_cnn_model(_project_root_path, 
                  save_model=args["save_model"], 
                  load_model=args["load_model"], 
                  save_weights=args["save_weights"])

#---------------------- EOC ---------------------
