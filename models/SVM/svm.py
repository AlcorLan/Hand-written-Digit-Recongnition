import sys
import os
import numpy as np
import pickle
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
import logging
from matplotlib import style

# Calculate project root path dynamically based on the current script's location
# This path will be used for relative imports and file saving/loading
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_path = os.path.abspath(os.path.join(_current_file_dir, '..'))
if _project_root_path not in sys.path:
    sys.path.append(_project_root_path)

# Import utility functions
from utils.data_preprocessing import load_and_preprocess_mnist_data
from utils.evaluation_metrics import plot_confusion_matrices, display_prediction_examples, print_classification_report
from utils.visualization import plot_dimensionality_reduction

# Configure logging at the beginning of the script
# Log file will now be saved in the Results/SVM directory
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

style.use('ggplot')

def run_svm_model(project_root_path=_project_root_path, log_file_path=None):
    """
    Runs the Support Vector Machine model for handwritten digit recognition.

    Args:
        project_root_path (str): The absolute path to the project root directory.
        log_file_path (str): The path to the log file, relative to project_root_path. (Optional, defaults to Results/SVM/summary.log)
    """
    print("\n--- Running Support Vector Machine Model ---")

    # Define the output directory for SVM results
    svm_results_dir = os.path.join(project_root_path, 'Results', 'SVM')
    os.makedirs(svm_results_dir, exist_ok=True)

    # Set up logging to the new log file path
    if log_file_path is None:
        log_file_path = os.path.join(svm_results_dir, 'summary.log')
        
    # Remove existing handlers to avoid duplicate logs when run multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path, filemode='w')
    
    # Temporarily redirect stdout to log file
    old_stdout = sys.stdout
    sys.stdout = open(log_file_path, "a") # Use append mode to not overwrite logging.basicConfig

    try:
        # Load and preprocess data using the utility function
        X_train, X_test, y_train, y_test, test_img, test_labels = load_and_preprocess_mnist_data(project_root_path)

        logging.info('\nSVM Classifier Parameters: C=1, kernel=\'linear\'') # Log model parameters
        print('\nPickling the Classifier for Future Use...')
        clf = svm.SVC(C=1, kernel='linear', random_state=42)
        clf.fit(X_train, y_train)

        # Save and Load the model
        pickle_path = os.path.join(svm_results_dir, 'MNIST_SVM.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(pickle_path, 'rb') as f:
            clf = pickle.load(f)

        print('\nCalculating Accuracy of trained Classifier...')
        confidence = clf.score(X_test, y_test)

        print('\nMaking Predictions on Validation Data...')
        y_pred = clf.predict(X_test)

        print('\nCalculating Accuracy of Predictions...')
        accuracy = accuracy_score(y_test, y_pred)

        print('\nSVM Trained Classifier Confidence: ', confidence)
        print('\nPredicted Values: ', y_pred)
        print('\nAccuracy of Classifier on Validation Image Data: ', accuracy)

        test_labels_pred = clf.predict(test_img)

        # Define save paths for plots
        confusion_matrix_save_path = os.path.join(svm_results_dir, 'SVM_Confusion_Matrices.png')
        prediction_examples_save_path = os.path.join(svm_results_dir, 'SVM_Prediction_Examples.png')

        plot_confusion_matrices(y_test, y_pred, test_labels, test_labels_pred, title_prefix="SVM ", save_path=confusion_matrix_save_path)
        print_classification_report(y_test, y_pred, title="SVM Validation Data ")
        print_classification_report(test_labels, test_labels_pred, title="SVM Test Data ")

        display_prediction_examples(test_img, test_labels, test_labels_pred, save_path=prediction_examples_save_path)
    finally:
        sys.stdout = old_stdout

# Optional: if you want to run this file directly for testing, keep this block
if __name__ == "__main__":
    # When run directly, project_root_path is determined based on current file's location
    run_svm_model()

#---------------------- EOC ---------------------#

