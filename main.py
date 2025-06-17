import os # Import os module
import sys
import argparse
import importlib.util # Added for dynamic module loading
import logging # Import logging module
import matplotlib # Import matplotlib to set its logging level

# Set TensorFlow logging level to suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = all messages, 1 = INFO filtered, 2 = WARNING filtered, 3 = ERROR filtered

# Set matplotlib logging level to WARNING to suppress debug messages
matplotlib.use('Agg') # Use non-interactive backend
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from voting_ensemble import run_voting_ensemble # Import the ensemble function
from utils.data_preprocessing import load_and_preprocess_mnist_data # For dimensionality reduction plotting
from utils.visualization import plot_dimensionality_reduction # For dimensionality reduction plotting

# Calculate project root path (where main.py resides)
project_root_path = os.path.abspath(os.path.dirname(__file__))
# Add project root to sys.path to allow imports from utils and model directories
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

# Helper function to dynamically load model functions
def load_model_function(model_dir, model_file_name, function_name, project_root_path):
    script_path = os.path.join(project_root_path, model_dir, model_file_name)
    module_name = f"__{model_dir.replace('.', '_')}_{model_file_name.replace('.py', '')}_dynamic_module__"
    
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Cannot find module at {script_path}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Run different Handwritten Digit Recognition models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=['KNN', 'SVM', 'RFC', 'CNN', 'Ensemble'],
        required=True,
        help="Specify the model to run (KNN, SVM, RFC, CNN, Ensemble)"
    )
    parser.add_argument(
        "--save_model",
        type=int,
        default=-1,
        help="For CNN: Set to 1 to save the trained model weights. Default is -1 (do not save)."
    )
    parser.add_argument(
        "--load_model",
        type=int,
        default=-1,
        help="For CNN: Set to 1 to load pre-trained model weights. Default is -1 (do not load)."
    )
    parser.add_argument(
        "--save_weights",
        type=str,
        default="cnn_weights.hdf5",
        help="For CNN: Specify the path to save/load model weights (e.g., cnn_weights.hdf5). This path is relative to the main project directory."
    )
    parser.add_argument(
        "--plot_dim_reduction",
        action='store_true',
        help="Set this flag to plot dimensionality reduction (PCA and t-SNE) results."
    )

    args = parser.parse_args()

    if args.plot_dim_reduction:
        print('\nVisualizing Data in Reduced Dimensions (PCA & t-SNE)...')
        X_train, _, y_train, _, _, _ = load_and_preprocess_mnist_data(project_root_path)
        
        pca_save_path = os.path.join(project_root_path, 'Results', 'Dimensionality_Reduction', 'PCA_2D.png')
        tsne_save_path = os.path.join(project_root_path, 'Results', 'Dimensionality_Reduction', 'tSNE_2D.png')

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
        
        plot_dimensionality_reduction(X_train, y_train, method='pca', n_components=2, 
                                      title='PCA of Training Data (2D)', save_path=pca_save_path)
        plot_dimensionality_reduction(X_train, y_train, method='tsne', n_components=2, 
                                      title='t-SNE of Training Data (2D)', save_path=tsne_save_path, sample_size=2000)
        print('Dimensionality reduction plots saved to Results/Dimensionality_Reduction/')

    # Dynamically load model functions
    model_functions = {
        'KNN': load_model_function('models/KNN', 'knn.py', 'run_knn_model', project_root_path),
        'SVM': load_model_function('models/SVM', 'svm.py', 'run_svm_model', project_root_path),
        'RFC': load_model_function('models/RFC', 'RFC.py', 'run_rfc_model', project_root_path),
        'CNN': load_model_function('models/CNN_Keras', 'CNN_MNIST.py', 'run_cnn_model', project_root_path),
    }

    if args.model in model_functions:
        if args.model == 'CNN':
            model_functions[args.model](project_root_path,
                                         save_model=args.save_model,
                                         load_model=args.load_model,
                                         save_weights=args.save_weights)
        else:
            model_functions[args.model](project_root_path)
    elif args.model == 'Ensemble':
        run_voting_ensemble(project_root_path)
    else:
        print(f"Error: Model '{args.model}' not recognized.")

if __name__ == "__main__":
    main() 