import sys
import os
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# Calculate project root path (assuming voting_ensemble.py is in the project root)
project_root_path = os.path.abspath(os.path.dirname(__file__))
# Add project root to sys.path to allow imports from utils and model directories
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

# Import utility functions
from utils.data_preprocessing import load_and_preprocess_mnist_data
from utils.evaluation_metrics import plot_confusion_matrices, display_prediction_examples, print_classification_report, display_voting_correction_details

# Define log file path for voting_ensemble within Results/Ensemble
log_file_path_for_ensemble = os.path.join(project_root_path, 'Results', 'Ensemble', 'summary.log')
os.makedirs(os.path.dirname(log_file_path_for_ensemble), exist_ok=True) # Ensure directory exists

# Configure logging at the beginning of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename=log_file_path_for_ensemble)

def run_voting_ensemble(project_root_path=project_root_path, log_file_path=log_file_path_for_ensemble):
    print("\n--- Running Voting Ensemble Model ---")
    old_stdout = sys.stdout
    current_log_file = log_file_path
    log_file = open(current_log_file, "w")
    sys.stdout = log_file

    # Define paths for saving results (plots)
    results_dir = os.path.join(project_root_path, 'Results')
    ensemble_results_dir = os.path.join(results_dir, 'Ensemble')
    os.makedirs(ensemble_results_dir, exist_ok=True)

    try:
        # --- 1. Load Data ---
        # We need the raw test_img and test_labels for CNN, and flattened for others.
        # load_and_preprocess_mnist_data returns flattened X_train, X_test, y_train, y_test, test_img (flattened), test_labels (original)
        # Let's modify data_preprocessing to return original test_img too for CNN later.
        # For now, let's directly load from MNIST_Dataset_Loader for testing images to ensure consistency for all models.

        from MNIST_Dataset_Loader.mnist_loader import MNIST
        from keras.utils import to_categorical # For CNN labels
        from sklearn.model_selection import train_test_split

        print('\nLoading original MNIST Test Data for ensemble...')
        mnist_loader = MNIST(project_root_path)
        # Load full training and testing datasets
        img_train_raw, labels_train_raw, _ = mnist_loader.load_training() # We don't need training filenames for this ensemble
        img_test_raw, labels_test_raw, test_filenames = mnist_loader.load_testing()
        
        # Use the raw test_img and labels for consistent prediction across models.
        # KNN/SVM/RFC will use the flattened version, CNN will use the reshaped version.
        test_img_flattened = np.array(img_test_raw).reshape((len(img_test_raw), -1))
        test_labels_original = np.array(labels_test_raw)

        # --- 2. Load Models and Get Predictions ---
        all_predictions = []

        # Load KNN Model
        print('\nLoading KNN model...')
        knn_pickle_path = os.path.join(project_root_path, 'Results', 'KNN', 'MNIST_KNN.pickle')
        with open(knn_pickle_path, 'rb') as f:
            knn_clf = pickle.load(f)
        knn_predictions = knn_clf.predict(test_img_flattened)
        all_predictions.append(knn_predictions)
        knn_accuracy = accuracy_score(test_labels_original, knn_predictions)
        print(f'KNN predictions obtained. Accuracy: {knn_accuracy * 100:.2f}%')

        # Load SVM Model
        print('\nLoading SVM model...')
        svm_pickle_path = os.path.join(project_root_path, 'Results', 'SVM', 'MNIST_SVM.pickle')
        with open(svm_pickle_path, 'rb') as f:
            svm_clf = pickle.load(f)
        svm_predictions = svm_clf.predict(test_img_flattened)
        all_predictions.append(svm_predictions)
        svm_accuracy = accuracy_score(test_labels_original, svm_predictions)
        print(f'SVM predictions obtained. Accuracy: {svm_accuracy * 100:.2f}%')

        # Load RFC Model
        print('\nLoading RFC model...')
        rfc_pickle_path = os.path.join(project_root_path, 'Results', 'RFC', 'MNIST_RFC.pickle')
        with open(rfc_pickle_path, 'rb') as f:
            rfc_clf = pickle.load(f)
        rfc_predictions = rfc_clf.predict(test_img_flattened)
        all_predictions.append(rfc_predictions)
        rfc_accuracy = accuracy_score(test_labels_original, rfc_predictions)
        print(f'RFC predictions obtained. Accuracy: {rfc_accuracy * 100:.2f}%')

        # Load CNN Model
        print('\nLoading CNN model...')
        # Need to re-add CNN_Keras path to sys.path for CNN import if not already there
        cnn_keras_path = os.path.join(project_root_path, 'models', 'CNN_Keras')
        if cnn_keras_path not in sys.path:
            sys.path.append(cnn_keras_path)
        from models.CNN_Keras.cnn.neural_network import CNN

        # CNN requires data in NHWC format and one-hot labels for training/evaluation.
        # For prediction, it expects NHWC, but output is probabilities.
        test_img_cnn_format = np.array(img_test_raw)[:, :, :, np.newaxis] # Reshape to NHWC

        total_classes = 10
        # test_labels_one_hot = to_categorical(test_labels_original, total_classes) # Not needed for prediction itself, but good to remember

        cnn_weights_path = os.path.join(project_root_path, 'Results', 'CNN', 'cnn_weights.hdf5')
        
        # Check if weights file exists before trying to load it
        if not os.path.exists(cnn_weights_path):
            print(f"Warning: CNN weights not found at {cnn_weights_path}. Training a new CNN model for ensemble (this might take time).")
            # If weights not found, train a basic CNN (simplified from CNN_MNIST.py)
            # This requires also training data for CNN, which is not passed here directly.
            # For a proper ensemble, training all models if weights don't exist is better in a dedicated training script.
            # For simplicity, if weights don't exist, we'll skip CNN for voting or raise error.
            # Let's assume pre-trained weights are available as per previous steps.
            raise FileNotFoundError(f"CNN weights file not found at {cnn_weights_path}. Please train CNN model first.")

        cnn_clf = CNN.build(width=28, height=28, depth=1, total_classes=total_classes, Saved_Weights_Path=cnn_weights_path)
        cnn_clf.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"]) # Optimizer needed for compile, but not for prediction with loaded weights
        
        # Predict probabilities, then get class labels
        cnn_probs = cnn_clf.predict(test_img_cnn_format)
        cnn_predictions = np.argmax(cnn_probs, axis=1)
        all_predictions.append(cnn_predictions)
        cnn_accuracy = accuracy_score(test_labels_original, cnn_predictions)
        print(f'CNN predictions obtained. Accuracy: {cnn_accuracy * 100:.2f}%')

        # Collect all individual model predictions for detailed display later
        individual_model_predictions = {
            'KNN': knn_predictions,
            'SVM': svm_predictions,
            'RFC': rfc_predictions,
            'CNN': cnn_predictions
        }

        # --- 3. Implement Voting Logic ---
        # Convert list of prediction arrays into a single 2D array (num_samples, num_models)
        all_predictions_array = np.array(all_predictions).T # Transpose to (num_samples, num_models)

        print('\nPerforming weighted voting...')
        num_samples = test_labels_original.shape[0]
        num_models = len(all_predictions)
        total_classes = 10 # Assuming 0-9 digits

        # Get model weights from their accuracies
        # Exclude 'Ensemble' from model_accuracies if it's already present in model_accuracies
        model_accuracies = {
            'KNN': knn_accuracy,
            'SVM': svm_accuracy,
            'RFC': rfc_accuracy,
            'CNN': cnn_accuracy,
            'Ensemble': accuracy_score(test_labels_original, mode(all_predictions_array, axis=1)[0].flatten())
        }

        model_individual_accuracies = {model: accuracy for model, accuracy in model_accuracies.items() if model != 'Ensemble'}
        
        # Normalize weights: scale accuracies so the highest accuracy gets a weight of 1.0
        max_accuracy = max(model_individual_accuracies.values())
        normalized_weights = {model: accuracy / max_accuracy for model, accuracy in model_individual_accuracies.items()}

        # Optional: Add a boosting factor for CNN if its importance needs to be further emphasized
        cnn_boost_factor = 2.0 # You can adjust this value (e.g., 1.5, 2.0) based on desired CNN influence
        if 'CNN' in normalized_weights:
            normalized_weights['CNN'] *= cnn_boost_factor

        # Ensure the order of weights matches the order of predictions in all_predictions
        # The order in `all_predictions` is KNN, SVM, RFC, CNN.
        ordered_weights = [
            normalized_weights['KNN'],
            normalized_weights['SVM'],
            normalized_weights['RFC'],
            normalized_weights['CNN']
        ]

        ensemble_predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            # Initialize scores for each class for the current sample
            class_scores = np.zeros(total_classes)
            for j in range(num_models):
                model_predicted_class = all_predictions_array[i, j]
                weight = ordered_weights[j]
                class_scores[model_predicted_class] += weight
            
            # The ensemble prediction is the class with the highest weighted score
            ensemble_predictions[i] = np.argmax(class_scores)
        
        print('Weighted voting complete.')

        # --- 4. Evaluate Ensemble Model ---
        print('\nEvaluating Ensemble Model...')
        # Use original test_labels for evaluation against ensemble predictions
        # Note: test_labels_original is 1D (e.g., [7, 2, 1, ...])

        # Overall Accuracy
        ensemble_accuracy = accuracy_score(test_labels_original, ensemble_predictions)
        print(f'\nEnsemble Model Accuracy: {ensemble_accuracy * 100:.2f}%')

        # Store all accuracies for comparison
        model_accuracies = {
            'KNN': knn_accuracy,
            'SVM': svm_accuracy,
            'RFC': rfc_accuracy,
            'CNN': cnn_accuracy,
            'Ensemble': ensemble_accuracy
        }

        print('\n--- Model Accuracy Summary ---')
        for model, accuracy in model_accuracies.items():
            print(f'{model}: {accuracy * 100:.2f}%')

        # Confusion Matrix
        print('\nCreating Confusion Matrix for Ensemble...')
        # plot_confusion_matrices expects y_true_train, y_pred_train, y_true_test, y_pred_test
        # For ensemble, we only have test data. So we'll adapt or use a single plot.
        # Let's create a single plot for ensemble confusion matrix.
        from sklearn.metrics import confusion_matrix
        ensemble_conf_mat = confusion_matrix(test_labels_original, ensemble_predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(ensemble_conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix for Voting Ensemble')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Save the confusion matrix plot
        conf_mat_save_path = os.path.join(ensemble_results_dir, 'Ensemble_Confusion_Matrix.png')
        plt.savefig(conf_mat_save_path)
        print(f"Ensemble Confusion matrix plot saved to {conf_mat_save_path}")
        plt.clf()
        plt.close()

        # Classification Report
        print_classification_report(test_labels_original, ensemble_predictions, title="Ensemble Test Data ")

        # --- 5. Display Prediction Examples ---
        # Need to provide original test_img (raw or appropriate format) for display_prediction_examples
        prediction_examples_save_path = os.path.join(ensemble_results_dir, 'Ensemble_Prediction_Examples.png')
        display_prediction_examples(test_img_flattened, test_labels_original, ensemble_predictions, title_prefix="Ensemble ", save_path=prediction_examples_save_path)

        # --- 6. Visualize Correction Examples by Voting ---
        # Now using the new display_voting_correction_details function
        print('\nVisualizing examples corrected by voting with detailed model predictions...')
        corrected_examples_save_path = os.path.join(ensemble_results_dir, 'Voting_Corrected_Details_Examples.png')
        display_voting_correction_details(
            test_img_flattened,
            test_labels_original,
            ensemble_predictions,
            individual_model_predictions,
            img_filenames=test_filenames, # Pass filenames here
            save_path=corrected_examples_save_path
        )

        # --- 7. Plot Model Accuracies Comparison ---
        print('\nPlotting model accuracies comparison...')
        models = list(model_accuracies.keys())
        accuracies = [model_accuracies[model] * 100 for model in models]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=accuracies, hue=models, palette='viridis', legend=False)
        plt.title('Comparison of Model Accuracies')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.ylim(96.0, 100)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.15, f'{acc:.2f}%', ha='center', va='bottom')
        accuracy_plot_save_path = os.path.join(ensemble_results_dir, 'Model_Accuracies_Comparison.png')
        plt.savefig(accuracy_plot_save_path)
        print(f"Model accuracies comparison plot saved to {accuracy_plot_save_path}")
        plt.clf()
        plt.close()

    finally:
        sys.stdout = old_stdout
        log_file.close()

if __name__ == "__main__":
    run_voting_ensemble(project_root_path) 