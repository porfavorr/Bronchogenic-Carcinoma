"""
Utility functions for the Cancer Detection Project.
Includes functions for image loading, plotting, model summaries, and file operations (pickle).
"""
import os
import cv2 # OpenCV for image manipulation if needed, though PIL is primary for loading
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image # Pillow for robust image loading

from tensorflow.keras.preprocessing.image import img_to_array # Keras utility
# plot_model requires pydot and graphviz to be installed
# from tensorflow.keras.utils import plot_model 

from config import TARGET_IMAGE_SIZE, get_logger # Assuming TARGET_IMAGE_SIZE is (width, height)

# Initialize logger for this module
logger = get_logger(__name__)

def load_and_preprocess_image_pil(image_path_or_bytes, target_size=TARGET_IMAGE_SIZE):
    """
    Loads an image using PIL, resizes it, converts to array, and normalizes.
    Args:
        image_path_or_bytes (str or BytesIO): Path to the image file or image bytes.
        target_size (tuple): Desired output size (width, height).
    Returns:
        numpy.ndarray: Preprocessed image array (normalized to [0, 1]) or None if loading fails.
    """
    try:
        # Check if input is a path (string) or bytes-like object
        if isinstance(image_path_or_bytes, str):
            if not os.path.exists(image_path_or_bytes):
                logger.error(f"Image not found at path: {image_path_or_bytes}")
                return None
            img = Image.open(image_path_or_bytes)
        elif hasattr(image_path_or_bytes, 'read'): # Check if it's a file-like object (e.g., BytesIO)
            img = Image.open(image_path_or_bytes)
        else:
            logger.error(f"Invalid image input type: {type(image_path_or_bytes)}. Expected path string or BytesIO.")
            return None

        # Convert to RGB if not already (e.g., handles grayscale, RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.debug(f"Image converted to RGB mode from {img.mode}.")

        img = img.resize(target_size, Image.Resampling.LANCZOS) # High quality downsampling
        img_array = img_to_array(img)  # Converts PIL image to NumPy array (H, W, C)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        logger.debug(f"Image loaded and preprocessed successfully from: {image_path_or_bytes if isinstance(image_path_or_bytes, str) else 'BytesIO object'}")
        return img_array
    except FileNotFoundError: # Specifically for path strings
        logger.error(f"Image file not found: {image_path_or_bytes}")
        return None
    except UnidentifiedImageError: # Pillow specific error for corrupted/unsupported images
        logger.error(f"Cannot identify image file (possibly corrupted or unsupported format): {image_path_or_bytes if isinstance(image_path_or_bytes, str) else 'BytesIO object'}")
        return None
    except Exception as e:
        logger.error(f"Error loading/preprocessing image '{image_path_or_bytes if isinstance(image_path_or_bytes, str) else 'BytesIO object'}' with PIL: {e}", exc_info=True)
        return None


def save_training_plot(history_data, plot_type, save_path):
    """
    Saves training history plots (accuracy or loss).
    Args:
        history_data (dict): Training history dictionary (e.g., from Keras model.fit).
                               Expected keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        plot_type (str): 'accuracy' or 'loss'.
        save_path (str): Full path (including filename) to save the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(history_data.get(plot_type, [])) + 1)

        if plot_type == 'accuracy':
            if 'accuracy' not in history_data or 'val_accuracy' not in history_data:
                logger.warning("Accuracy/val_accuracy keys missing in history_data. Skipping accuracy plot.")
                plt.close()
                return
            plt.plot(epochs_range, history_data['accuracy'], label='Training Accuracy')
            plt.plot(epochs_range, history_data['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy Over Epochs')
            plt.ylabel('Accuracy')
        elif plot_type == 'loss':
            if 'loss' not in history_data or 'val_loss' not in history_data:
                logger.warning("Loss/val_loss keys missing in history_data. Skipping loss plot.")
                plt.close()
                return
            plt.plot(epochs_range, history_data['loss'], label='Training Loss')
            plt.plot(epochs_range, history_data['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.ylabel('Loss')
        else:
            logger.warning(f"Unknown plot type: {plot_type}. Choose 'accuracy' or 'loss'.")
            plt.close()
            return

        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close() # Close the figure to free memory
        logger.info(f"{plot_type.capitalize()} plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot '{plot_type}' to '{save_path}': {e}", exc_info=True)


def save_model_summary_and_plot(model, base_filepath):
    """
    Saves the model summary to a text file and model architecture plot to an image file.
    Args:
        model (keras.Model): Trained Keras model.
        base_filepath (str): Base path for saving (e.g., 'reports/colon/colon_model').
                             '_summary.txt' and '_architecture.png' will be appended.
    """
    summary_path = f"{base_filepath}_summary.txt"
    # plot_path = f"{base_filepath}_architecture.png" # For plot_model

    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True) # Ensure directory exists
        with open(summary_path, 'w') as f:
            # Redirect model.summary() output to the file
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Model summary saved to {summary_path}")

        # Keras plot_model can be problematic due to dependencies (Graphviz, pydot).
        # Making it optional or providing alternatives if it fails.
        # try:
        #     plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True, dpi=96)
        #     logger.info(f"Model architecture plot saved to {plot_path}")
        # except ImportError:
        #     logger.warning(f"Could not import pydot or graphviz. Skipping model architecture plot. "
        #                    f"Install pydot and graphviz (OS-level) to enable this feature.")
        # except Exception as e_plot:
        #     logger.error(f"Error generating model architecture plot: {e_plot}", exc_info=True)
            
    except Exception as e_summary:
        logger.error(f"Error saving model summary: {e_summary}", exc_info=True)


def save_pickle_object(data_object, file_path):
    """Saves a Python object to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        with open(file_path, 'wb') as f:
            pickle.dump(data_object, f)
        logger.info(f"Object successfully saved to pickle file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to pickle file '{file_path}': {e}", exc_info=True)

def load_pickle_object(file_path):
    """Loads a Python object from a pickle file."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Pickle file not found: {file_path}")
            return None
        with open(file_path, 'rb') as f:
            data_object = pickle.load(f)
        logger.info(f"Object successfully loaded from pickle file: {file_path}")
        return data_object
    except FileNotFoundError: # Redundant due to os.path.exists, but good practice
        logger.error(f"Pickle file not found during load attempt: {file_path}")
        return None
    except pickle.UnpicklingError:
        logger.error(f"Error unpickling data from {file_path}. File might be corrupted or not a pickle file.")
        return None
    except Exception as e:
        logger.error(f"Error loading object from pickle file '{file_path}': {e}", exc_info=True)
        return None

def plot_and_save_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """
    Plots a confusion matrix as a heatmap using Seaborn and saves it.
    Args:
        cm (numpy.ndarray): The confusion matrix (output of sklearn.metrics.confusion_matrix).
        class_names (list): List of class names for axis labels.
        save_path (str): Full path (including filename) to save the heatmap image.
        title (str): Title for the plot.
    """
    try:
        plt.figure(figsize=(len(class_names) * 2.5, len(class_names) * 2)) # Adjust size based on num classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12}, cbar=True) # Added annot_kws and cbar
        plt.title(title, fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha="right") # Improve x-tick label readability
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close() # Close the figure
        logger.info(f"Confusion matrix heatmap saved to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting/saving confusion matrix heatmap to '{save_path}': {e}", exc_info=True)


if __name__ == '__main__':
    logger.info("Utils module self-test initiated.")
    
    # --- Test load_and_preprocess_image_pil ---
    # Create a dummy image for testing
    dummy_image_filename = "dummy_test_image_utils.png"
    try:
        img = Image.new('RGB', (200, 150), color = 'red') # Create a non-square image
        img.save(dummy_image_filename)
        logger.info(f"Created dummy image: {dummy_image_filename}")

        processed_img_array = load_and_preprocess_image_pil(dummy_image_filename, target_size=TARGET_IMAGE_SIZE)
        if processed_img_array is not None:
            assert processed_img_array.shape == (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3), \
                f"Processed image shape mismatch: {processed_img_array.shape}"
            assert np.max(processed_img_array) <= 1.0 and np.min(processed_img_array) >= 0.0, \
                "Processed image not normalized correctly."
            logger.info(f"load_and_preprocess_image_pil test successful. Shape: {processed_img_array.shape}")
        else:
            logger.error("load_and_preprocess_image_pil test failed to process image.")
        
        # Test with BytesIO
        with open(dummy_image_filename, 'rb') as f_bytes:
            img_bytes = io.BytesIO(f_bytes.read())
        processed_img_bytes_array = load_and_preprocess_image_pil(img_bytes, target_size=TARGET_IMAGE_SIZE)
        if processed_img_bytes_array is not None:
             logger.info(f"load_and_preprocess_image_pil with BytesIO successful. Shape: {processed_img_bytes_array.shape}")
        else:
            logger.error("load_and_preprocess_image_pil with BytesIO failed.")

    except Exception as e:
        logger.error(f"Error during image loading test: {e}")
    finally:
        if os.path.exists(dummy_image_filename):
            os.remove(dummy_image_filename)
            logger.info(f"Removed dummy image: {dummy_image_filename}")

    # --- Test save_training_plot ---
    dummy_history = {
        'accuracy': [0.1, 0.25, 0.35], 'val_accuracy': [0.12, 0.22, 0.32],
        'loss': [1.0, 0.85, 0.65], 'val_loss': [0.95, 0.88, 0.72]
    }
    temp_report_dir = os.path.join(PROJECT_ROOT, 'reports', 'utils_test_plots') # Assuming PROJECT_ROOT is from config
    if 'PROJECT_ROOT' not in globals(): # Define if not imported from config (e.g. direct run)
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        temp_report_dir = os.path.join(PROJECT_ROOT, 'reports', 'utils_test_plots')
        
    os.makedirs(temp_report_dir, exist_ok=True)
    save_training_plot(dummy_history, 'accuracy', os.path.join(temp_report_dir, 'dummy_accuracy.png'))
    save_training_plot(dummy_history, 'loss', os.path.join(temp_report_dir, 'dummy_loss.png'))
    logger.info(f"Test plots (if generated) are in '{temp_report_dir}'. Review them manually.")

    # --- Test pickle functions ---
    test_data_pickle = {"message": "Hello Pickle!", "values": [10, 20, 30]}
    pickle_test_path = os.path.join(temp_report_dir, 'test_object.pkl')
    save_pickle_object(test_data_pickle, pickle_test_path)
    loaded_data_pickle = load_pickle_object(pickle_test_path)
    assert loaded_data_pickle == test_data_pickle, "Pickle load/save failed: data mismatch."
    logger.info(f"Pickle save/load test successful. Loaded data: {loaded_data_pickle}")

    # --- Test plot_and_save_confusion_matrix ---
    dummy_cm = np.array([[10, 2], [3, 15]])
    dummy_classes = ['Class Alpha', 'Class Beta']
    cm_save_path = os.path.join(temp_report_dir, 'dummy_confusion_matrix.png')
    plot_and_save_confusion_matrix(dummy_cm, dummy_classes, cm_save_path, title='Dummy Test CM')
    logger.info(f"Confusion matrix plot test. Check '{cm_save_path}'.")

    # --- Test save_model_summary_and_plot (requires a dummy Keras model) ---
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    dummy_keras_model = Sequential([Input(shape=(10,)), Dense(5, activation='relu'), Dense(2, activation='softmax')])
    dummy_keras_model.compile(optimizer='adam', loss='categorical_crossentropy')
    summary_base_path = os.path.join(temp_report_dir, 'dummy_model')
    save_model_summary_and_plot(dummy_keras_model, summary_base_path)
    logger.info(f"Model summary test. Check files starting with '{summary_base_path}'.")
    
    logger.info("Utils module self-test completed. Please review generated files in 'reports/utils_test_plots'.")
    # Consider cleaning up temp_report_dir after testing if desired.
    # import shutil
    # if os.path.exists(temp_report_dir):
    #     shutil.rmtree(temp_report_dir)
    #     logger.info(f"Cleaned up test directory: {temp_report_dir}")
