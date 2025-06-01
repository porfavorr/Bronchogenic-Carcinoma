"""
Main command-line interface (CLI) script to initiate and manage the training 
pipeline for the cancer detection models.

This script allows users to:
- Specify the type of model to train (e.g., 'colon' or 'lung').
- Override default training hyperparameters like epochs and batch size.
- Potentially select different model architectures or enable/disable features like transfer learning.

Example Usage:
  python src/main_train.py --model_type colon --epochs 100 --batch_size 16
  python src/main_train.py --model_type lung --use_transfer_learning
"""
import argparse
import os # For path manipulations if needed, though config handles most

# Import configurations and the main training function
from config import (
    # Colon model configs
    COLON_DATA_DIR, COLON_NUM_CLASSES, COLON_CLASSES, 
    COLON_MODEL_PATH, COLON_HISTORY_PATH, COLON_LABEL_ENCODER_PATH, 
    COLON_DEFAULT_EPOCHS, COLON_DEFAULT_BATCH_SIZE, COLON_REPORT_DIR,
    # Lung model configs
    LUNG_DATA_DIR, LUNG_NUM_CLASSES, LUNG_CLASSES, 
    LUNG_MODEL_PATH, LUNG_HISTORY_PATH, LUNG_LABEL_ENCODER_PATH, 
    LUNG_DEFAULT_EPOCHS, LUNG_DEFAULT_BATCH_SIZE, LUNG_REPORT_DIR,
    get_logger # For logging within this script
)
from training_pipeline import execute_training_pipeline # The core training orchestrator

# Initialize logger for this CLI script
logger = get_logger(__name__)

def setup_arg_parser():
    """
    Sets up and returns the argument parser for the training script.
    """
    parser = argparse.ArgumentParser(
        description="CLI for Training Cancer Detection Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True, 
        choices=['colon', 'lung'],
        help="Specify the type of cancer model to train: 'colon' or 'lung'."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, # If None, will use the default from config.py for the chosen model_type
        help="Number of training epochs. Overrides the default setting in config.py."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None, # If None, will use default from config.py
        help="Training batch size. Overrides the default setting in config.py."
    )
    parser.add_argument(
        "--use_transfer_learning",
        action='store_true', # Makes this a boolean flag; if present, it's True, else False.
        help="Enable the use of a pre-trained transfer learning model (e.g., VGG16) if implemented. "
             "Note: The specific transfer learning architecture must be available in model_architecture.py."
    )
    # Future potential arguments:
    # parser.add_argument("--learning_rate", type=float, help="Override initial learning rate.")
    # parser.add_argument("--optimizer", type=str, choices=['adam', 'sgd'], help="Choose optimizer.")
    # parser.add_argument("--custom_data_dir", type=str, help="Path to a custom dataset directory.")

    return parser

def main():
    """
    Main execution function for the training CLI.
    Parses arguments, sets up configurations, and calls the training pipeline.
    """
    arg_parser = setup_arg_parser()
    cli_args = arg_parser.parse_args()

    logger.info("Starting Model Training CLI...")
    logger.info(f"Command Line Arguments Received: {cli_args}")

    # Determine configurations based on the chosen model_type
    if cli_args.model_type == 'colon':
        # Use colon-specific configurations from config.py
        dataset_dir = COLON_DATA_DIR
        num_model_classes = COLON_NUM_CLASSES
        model_class_names = COLON_CLASSES
        output_model_path = COLON_MODEL_PATH
        output_history_path = COLON_HISTORY_PATH
        output_encoder_path = COLON_LABEL_ENCODER_PATH
        reports_dir = COLON_REPORT_DIR
        # Override epochs and batch_size if provided via CLI, else use config defaults
        epochs_to_run = cli_args.epochs if cli_args.epochs is not None else COLON_DEFAULT_EPOCHS
        batch_size_to_use = cli_args.batch_size if cli_args.batch_size is not None else COLON_DEFAULT_BATCH_SIZE
        
    elif cli_args.model_type == 'lung':
        # Use lung-specific configurations
        dataset_dir = LUNG_DATA_DIR
        num_model_classes = LUNG_NUM_CLASSES
        model_class_names = LUNG_CLASSES
        output_model_path = LUNG_MODEL_PATH
        output_history_path = LUNG_HISTORY_PATH
        output_encoder_path = LUNG_LABEL_ENCODER_PATH
        reports_dir = LUNG_REPORT_DIR
        epochs_to_run = cli_args.epochs if cli_args.epochs is not None else LUNG_DEFAULT_EPOCHS
        batch_size_to_use = cli_args.batch_size if cli_args.batch_size is not None else LUNG_DEFAULT_BATCH_SIZE
        
    else:
        # This case should ideally not be reached due to 'choices' in argparse,
        # but it's good practice for robustness.
        logger.error(f"Invalid model_type '{cli_args.model_type}' specified. Critical error.")
        print(f"Error: Invalid model type '{cli_args.model_type}'. Choose 'colon' or 'lung'.")
        return # Exit if model type is somehow invalid

    # Log the final parameters being used for the training run
    logger.info(f"--- Training Configuration for {cli_args.model_type.upper()} Model ---")
    logger.info(f"  Dataset Directory: {dataset_dir}")
    logger.info(f"  Number of Classes: {num_model_classes}")
    logger.info(f"  Class Names: {model_class_names}")
    logger.info(f"  Model Save Path: {output_model_path}")
    logger.info(f"  Training History Path: {output_history_path}")
    logger.info(f"  Label Encoder Path: {output_encoder_path}")
    logger.info(f"  Reports Directory: {reports_dir}")
    logger.info(f"  Epochs: {epochs_to_run}")
    logger.info(f"  Batch Size: {batch_size_to_use}")
    logger.info(f"  Use Transfer Learning: {cli_args.use_transfer_learning}")
    logger.info("----------------------------------------------------")

    # Ensure the reports directory exists (config.py should also handle this, but good for redundancy)
    os.makedirs(reports_dir, exist_ok=True)

    # --- Execute the Training Pipeline ---
    logger.info(f"Calling the training pipeline for {cli_args.model_type.upper()} model...")
    trained_model, training_history = execute_training_pipeline(
        model_type_str=cli_args.model_type,
        dataset_base_dir=dataset_dir,
        defined_num_classes=num_model_classes,
        defined_class_names=model_class_names,
        target_model_path=output_model_path,
        training_history_path=output_history_path,
        label_enc_path=output_encoder_path,
        model_report_dir=reports_dir,
        num_epochs=epochs_to_run,
        train_batch_size=batch_size_to_use,
        use_transfer_learning_flag=cli_args.use_transfer_learning
    )

    # --- Post-Training Summary ---
    if trained_model and training_history:
        logger.info(f"Training pipeline for {cli_args.model_type.upper()} model completed SUCCESSFULLY.")
        logger.info(f"  Final model saved at: {output_model_path}")
        logger.info(f"  Review training reports and plots in: {reports_dir}")
        logger.info(f"  TensorBoard logs (if generated) are in the 'logs/fit/{cli_args.model_type}/' directory.")
    else:
        logger.error(f"Training pipeline for {cli_args.model_type.upper()} model FAILED or was interrupted.")
        logger.error("  Please check the logs for detailed error messages.")

    logger.info("Model Training CLI finished.")

if __name__ == '__main__':
    # This condition ensures that main() is called only when the script is executed directly,
    # not when it's imported as a module elsewhere.
    # To run from project root: `python -m src.main_train --model_type colon ...`
    # Or, if in src directory: `python main_train.py --model_type colon ...`
    main()
