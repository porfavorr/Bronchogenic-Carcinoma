"""
Configuration file for the Cancer Detection Project.
Stores paths, hyperparameters, and other global settings.
All paths are constructed relative to the project root.
"""
import os
import logging

# --- Project Root Directory ---
# Assumes this config.py is in a 'src' subdirectory of the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data Paths ---
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'lung_colon_image_set')
COLON_DATA_DIR = os.path.join(BASE_DATA_DIR, 'colon_image_sets')
LUNG_DATA_DIR = os.path.join(BASE_DATA_DIR, 'lung_image_sets')

# --- Output Directories ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs') # For TensorBoard and general logs

# Specific report directories
COLON_REPORT_DIR = os.path.join(REPORTS_DIR, 'colon')
LUNG_REPORT_DIR = os.path.join(REPORTS_DIR, 'lung')

# TensorBoard log directories
TENSORBOARD_LOG_DIR_COLON = os.path.join(LOGS_DIR, 'fit', 'colon')
TENSORBOARD_LOG_DIR_LUNG = os.path.join(LOGS_DIR, 'fit', 'lung')

# General application log file
APP_LOG_FILE = os.path.join(LOGS_DIR, 'application.log')

# --- Ensure Directories Exist ---
# Using a function for cleaner setup, called at the end of the file.
def create_directories():
    """Creates all necessary directories if they don't exist."""
    dirs_to_create = [
        MODEL_DIR, REPORTS_DIR, LOGS_DIR,
        COLON_REPORT_DIR, LUNG_REPORT_DIR,
        TENSORBOARD_LOG_DIR_COLON, TENSORBOARD_LOG_DIR_LUNG,
        # Data directories are expected to be created by the user
        # os.path.join(COLON_DATA_DIR, 'colon_aca'), os.path.join(COLON_DATA_DIR, 'colon_n'), # Example
        # os.path.join(LUNG_DATA_DIR, 'lung_aca'), os.path.join(LUNG_DATA_DIR, 'lung_n'), os.path.join(LUNG_DATA_DIR, 'lung_scc') # Example
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

# --- Image Processing Parameters ---
TARGET_IMAGE_WIDTH = 128
TARGET_IMAGE_HEIGHT = 128
TARGET_IMAGE_SIZE = (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT) # Order: (width, height) for PIL/Keras load_img
IMAGE_CHANNELS = 3  # RGB

# --- Dataset Split Parameters ---
VALIDATION_SPLIT = 0.2  # 20% of (training+validation) data for validation
TEST_SPLIT = 0.1        # 10% of original data for testing
RANDOM_STATE = 42       # For reproducible splits

# --- Colon Model Specific Configurations ---
COLON_MODEL_NAME = 'colon_cancer_model.keras' # Using .keras for modern Keras format
COLON_MODEL_PATH = os.path.join(MODEL_DIR, COLON_MODEL_NAME)
COLON_HISTORY_PATH = os.path.join(MODEL_DIR, 'colon_model_history.pkl')
COLON_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'colon_label_encoder.pkl')
COLON_CLASSES = ['colon_aca', 'colon_n'] # Adenocarcinoma, Normal
COLON_NUM_CLASSES = len(COLON_CLASSES)
COLON_DEFAULT_EPOCHS = 50 # As per user's potential requirement
COLON_DEFAULT_BATCH_SIZE = 32

# --- Lung Model Specific Configurations ---
LUNG_MODEL_NAME = 'lung_cancer_model.keras'
LUNG_MODEL_PATH = os.path.join(MODEL_DIR, LUNG_MODEL_NAME)
LUNG_HISTORY_PATH = os.path.join(MODEL_DIR, 'lung_model_history.pkl')
LUNG_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'lung_label_encoder.pkl')
LUNG_CLASSES = ['lung_aca', 'lung_n', 'lung_scc'] # Adenocarcinoma, Normal, Squamous Cell Carcinoma
LUNG_NUM_CLASSES = len(LUNG_CLASSES)
LUNG_DEFAULT_EPOCHS = 75 # As per user's potential requirement
LUNG_DEFAULT_BATCH_SIZE = 32

# --- Training Callbacks Parameters ---
EARLY_STOPPING_PATIENCE = 15 # Increased patience
REDUCE_LR_PATIENCE = 7      # Patience for learning rate reduction
REDUCE_LR_FACTOR = 0.2      # Factor to reduce LR by (new_lr = lr * factor)
MONITOR_METRIC = 'val_accuracy' # Metric to monitor for callbacks (could be 'val_loss')
MIN_DELTA_EARLY_STOPPING = 0.001 # Minimum change to qualify as an improvement for EarlyStopping

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
# Note: Logger instances are typically created in each module using get_logger(module_name)

def get_logger(name, level=LOG_LEVEL, log_file=APP_LOG_FILE):
    """
    Initializes and returns a logger instance.
    Args:
        name (str): Name of the logger (usually __name__ of the module).
        level (int): Logging level (e.g., logging.INFO).
        log_file (str): Path to the log file.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevents adding multiple handlers if logger already has them (e.g., in Streamlit re-runs)
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(ch)
        
        # File Handler (ensure logs directory exists for APP_LOG_FILE)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)
        
    return logger

# --- Create directories on import ---
create_directories() # Call this once when the module is loaded.

if __name__ == '__main__':
    # This block runs if the script is executed directly (e.g., python src/config.py)
    # Useful for verifying paths and configurations.
    logger_instance = get_logger(__name__) # Use the factory
    logger_instance.info("Configuration loaded and directories ensured.")
    logger_instance.info(f"Project Root: {PROJECT_ROOT}")
    logger_instance.info(f"Colon Data Directory: {COLON_DATA_DIR}")
    logger_instance.info(f"Lung Data Directory: {LUNG_DATA_DIR}")
    logger_instance.info(f"Models will be saved in: {MODEL_DIR}")
    logger_instance.info(f"Reports will be saved in: {REPORTS_DIR}")
    logger_instance.info(f"Logs (including TensorBoard) in: {LOGS_DIR}")
    logger_instance.info(f"Application log file: {APP_LOG_FILE}")
    logger_instance.info(f"Colon classes: {COLON_CLASSES}, Num: {COLON_NUM_CLASSES}")
    logger_instance.info(f"Lung classes: {LUNG_CLASSES}, Num: {LUNG_NUM_CLASSES}")
    logger_instance.info(f"Target image size (W, H): {TARGET_IMAGE_SIZE}")
