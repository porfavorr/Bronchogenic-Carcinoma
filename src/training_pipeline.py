"""
Training pipeline for cancer detection models.
This module orchestrates the entire training process:
1. Loading dataset paths and labels.
2. Encoding labels.
3. Creating data generators (with augmentation for training).
4. Building or loading the model architecture.
5. Defining Keras callbacks.
6. Training the model.
7. Saving the trained model, history, and label encoder.
8. Evaluating the model on the test set.
"""
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model # For loading the best model for evaluation

from config import (TARGET_IMAGE_SIZE, RANDOM_STATE,
                     EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, 
                     MONITOR_METRIC, MIN_DELTA_EARLY_STOPPING,
                     TENSORBOARD_LOG_DIR_COLON, TENSORBOARD_LOG_DIR_LUNG, # For TensorBoard
                     get_logger)
from data_preprocessing import (load_image_paths_and_labels_df, 
                                 create_and_save_label_encoder, 
                                 prepare_data_generators)
from model_architecture import get_colon_cancer_model, get_lung_cancer_model , build_vgg16_transfer_model
from evaluation import evaluate_full_model_performance
from utils import save_pickle_object, save_model_summary_and_plot, load_pickle_object

logger = get_logger(__name__)

def execute_training_pipeline(
    model_type_str, 
    dataset_base_dir, 
    defined_num_classes, 
    defined_class_names, 
    target_model_path, 
    training_history_path, 
    label_enc_path, 
    model_report_dir,
    num_epochs, 
    train_batch_size, 
    use_transfer_learning_flag=False): # Added flag for transfer learning
    """
    Main function to run the training pipeline for a specified model type.
    Args:
        model_type_str (str): 'colon' or 'lung'.
        dataset_base_dir (str): Path to the specific cancer type's image data directory.
        defined_num_classes (int): Expected number of classes for this model (from config).
        defined_class_names (list): List of expected class name strings (from config).
        target_model_path (str): Path to save the trained Keras model.
        training_history_path (str): Path to save the training history (pickle).
        label_enc_path (str): Path to save the LabelEncoder object (pickle).
        model_report_dir (str): Directory to save evaluation reports and plots for this model.
        num_epochs (int): Number of training epochs.
        train_batch_size (int): Training batch size.
        use_transfer_learning_flag (bool): If True, attempts to use a transfer learning model.
    Returns:
        tuple: (trained_keras_model, training_history_dict) or (None, None) on failure.
    """
    logger.info(f"===== Starting Training Pipeline for {model_type_str.upper()} Cancer Model =====")
    pipeline_start_time = time.time()

    # --- 1. Load Dataset Paths and Labels ---
    logger.info(f"Step 1: Loading dataset from: {dataset_base_dir}")
    dataframe_images = load_image_paths_and_labels_df(dataset_base_dir, defined_class_names)
    if dataframe_images is None or dataframe_images.empty:
        logger.error(f"Failed to load dataset for {model_type_str}. Aborting training pipeline.")
        return None, None
    logger.info(f"Loaded {len(dataframe_images)} image records for {model_type_str}.")

    # --- 2. Encode Labels ---
    logger.info(f"Step 2: Encoding labels and saving LabelEncoder to {label_enc_path}...")
    label_encoder_instance, _ = create_and_save_label_encoder(dataframe_images['label'], label_enc_path)
    if label_encoder_instance is None:
        logger.error("Failed to create or save LabelEncoder. Aborting training pipeline.")
        return None, None
    
    # Verify consistency of classes from encoder vs. config
    actual_classes_from_encoder = list(label_encoder_instance.classes_)
    if len(actual_classes_from_encoder) != defined_num_classes or \
       set(actual_classes_from_encoder) != set(defined_class_names):
        logger.error(f"CRITICAL MISMATCH: Classes from LabelEncoder ({actual_classes_from_encoder}) "
                       f"do not match configured classes ({defined_class_names}).")
        logger.error("This could be due to unexpected subdirectories in your dataset or incorrect config.")
        logger.error("Please verify your dataset structure and `config.py` definitions.")
        return None, None
    logger.info(f"Labels encoded. Actual classes found and encoded: {actual_classes_from_encoder}")
    num_classes_for_model = len(actual_classes_from_encoder) # Use this for model creation

    # --- 3. Create Data Generators ---
    logger.info("Step 3: Creating data generators (train/validation/test)...")
    train_gen, val_gen, test_gen, X_test_data, y_test_labels_cat = prepare_data_generators(
        dataframe_images, 
        label_encoder_instance,
        target_img_size=TARGET_IMAGE_SIZE, # (width, height) from config
        batch_sz=train_batch_size
    )
    if train_gen is None: # val_gen can be None if VAL_SPLIT is 0 or dataset too small
        logger.error(f"Failed to create data generators for {model_type_str}. Aborting training.")
        return None, None
    if train_gen.n == 0:
        logger.error("Training generator is empty (0 samples). Aborting. Check dataset and splits.")
        return None, None
    if val_gen is None and train_gen.n > 0 : # If train_gen exists but val_gen is None
        logger.warning("Validation generator is None. Model will train without validation steps.")
        # This might be acceptable if VALIDATION_SPLIT was 0.

    # --- 4. Build or Load Model Architecture ---
    # Input shape for the model (H, W, C)
    model_input_shape = (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3) # Height, Width, Channels
    
    logger.info(f"Step 4: Building model architecture for {model_type_str} with {num_classes_for_model} classes and input shape {model_input_shape}.")
    
    keras_model = None
    if use_transfer_learning_flag:
        logger.info("Attempting to use Transfer Learning model (VGG16)...")
        # keras_model = build_vgg16_transfer_model(model_input_shape, num_classes_for_model) # Uncomment if implemented
        # if keras_model is None:
        #     logger.warning("Failed to build VGG16 transfer model, or it's commented out. Falling back to custom CNN.")
        # else:
        #     logger.info("VGG16 Transfer Learning model built.")
        logger.warning("Transfer learning (VGG16) is currently commented out in model_architecture.py. Using custom CNN.")
        # Fallback to custom if VGG16 fails or is not primary
        if model_type_str == 'colon':
            keras_model = get_colon_cancer_model(model_input_shape, num_classes_for_model)
        elif model_type_str == 'lung':
            keras_model = get_lung_cancer_model(model_input_shape, num_classes_for_model)

    if keras_model is None: # If not using transfer learning or if it failed
        if model_type_str == 'colon':
            keras_model = get_colon_cancer_model(model_input_shape, num_classes_for_model)
        elif model_type_str == 'lung':
            keras_model = get_lung_cancer_model(model_input_shape, num_classes_for_model)
        else: # Should not happen if model_type_str is validated earlier
            logger.error(f"Unknown model type '{model_type_str}' for architecture creation.")
            return None, None

    if keras_model is None:
        logger.error(f"Failed to build Keras model for {model_type_str}. Aborting training.")
        return None, None
    
    # Save initial model summary and architecture plot (if plot_model is available)
    initial_model_summary_path_base = os.path.join(model_report_dir, f"{model_type_str}_model_initial_architecture")
    save_model_summary_and_plot(keras_model, initial_model_summary_path_base)
    logger.info(f"Initial model summary for '{keras_model.name}' saved.")

    # --- 5. Define Keras Callbacks ---
    logger.info("Step 5: Defining Keras callbacks...")
    # ModelCheckpoint: Save the best model found during training
    # Using .keras format for saving the entire model (architecture, weights, optimizer state)
    checkpoint_cb = ModelCheckpoint(
        filepath=target_model_path, # Save directly to the final path
        monitor=MONITOR_METRIC,    # Metric to monitor (e.g., 'val_accuracy' or 'val_loss')
        save_best_only=True,       # Only save if improvement is seen
        save_weights_only=False,   # Save the full model
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min', # 'max' for accuracy, 'min' for loss
        verbose=1
    )
    # EarlyStopping: Stop training if no improvement is seen for a number of epochs
    early_stopping_cb = EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=1,
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min',
        restore_best_weights=True, # Restores model weights from the epoch with the best value
        min_delta=MIN_DELTA_EARLY_STOPPING # Minimum change to qualify as improvement
    )
    # ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving
    reduce_lr_cb = ReduceLROnPlateau(
        monitor=MONITOR_METRIC,
        factor=REDUCE_LR_FACTOR,    # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,               # Lower bound on the learning rate
        verbose=1,
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min'
    )
    # TensorBoard: For visualizing training progress
    tb_log_dir = TENSORBOARD_LOG_DIR_COLON if model_type_str == 'colon' else TENSORBOARD_LOG_DIR_LUNG
    # Add timestamp to TensorBoard log directory for unique runs
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    tb_log_dir_run = os.path.join(tb_log_dir, current_time_str)
    os.makedirs(tb_log_dir_run, exist_ok=True)
    tensorboard_cb = TensorBoard(log_dir=tb_log_dir_run, histogram_freq=1) # Log histograms every epoch

    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]
    logger.info(f"Callbacks configured: ModelCheckpoint (to {target_model_path}), EarlyStopping, ReduceLROnPlateau, TensorBoard (to {tb_log_dir_run}).")

    # --- 6. Train the Model ---
    logger.info(f"Step 6: Starting model training for {model_type_str}...")
    logger.info(f"Epochs: {num_epochs}, Batch Size: {train_batch_size}")
    logger.info(f"Training samples: {train_gen.n}, Validation samples: {val_gen.n if val_gen else 0}")
    logger.info(f"Steps per epoch: {len(train_gen)}, Validation steps: {len(val_gen) if val_gen else 0}")

    training_history = None
    try:
        history_object = keras_model.fit(
            train_gen,
            epochs=num_epochs,
            validation_data=val_gen, # Keras handles val_gen being None (no validation)
            callbacks=callbacks_list,
            steps_per_epoch=len(train_gen), # num_samples // batch_size
            validation_steps=len(val_gen) if val_gen else None 
        )
        training_history = history_object.history # Extract the history dictionary
        logger.info(f"Training completed for {model_type_str}.")
    except Exception as e:
        logger.error(f"An error occurred during model.fit for {model_type_str}: {e}", exc_info=True)
        return None, None # Critical error, stop pipeline
        
    # Note: If EarlyStopping with restore_best_weights=True was used, 'keras_model' in memory
    # already has the best weights. ModelCheckpoint also saves the best model to target_model_path.
    # For safety, one could reload the explicitly saved best model before evaluation,
    # but it should be consistent if restore_best_weights=True.

    # --- 7. Save Training History ---
    if training_history:
        logger.info(f"Step 7: Saving training history to {training_history_path}...")
        save_pickle_object(training_history, training_history_path)
    else:
        logger.warning("Training history object is empty or not available. Skipping history save.")

    # --- 8. Evaluate Model (on Test set if available) ---
    logger.info(f"Step 8: Evaluating the best model on the test set for {model_type_str}...")
    
    # Load the best model saved by ModelCheckpoint for evaluation to ensure it's the one from disk.
    # This is especially important if EarlyStopping's restore_best_weights was False or had issues.
    best_model_for_eval = None
    if os.path.exists(target_model_path):
        try:
            best_model_for_eval = load_model(target_model_path)
            logger.info(f"Successfully loaded best model from {target_model_path} for final evaluation.")
        except Exception as e_load:
            logger.error(f"Failed to load best model from {target_model_path} for evaluation: {e_load}. "
                           "Falling back to model in memory (which should have best weights if EarlyStopping worked).")
            best_model_for_eval = keras_model # Fallback
    else:
        logger.warning(f"Best model file {target_model_path} not found after training. "
                       "Using model currently in memory for evaluation. This might not be the best performing one.")
        best_model_for_eval = keras_model # Fallback

    if best_model_for_eval:
        evaluate_full_model_performance(
            model=best_model_for_eval, 
            test_data_x=X_test_data, # Can be None if test_gen is used
            test_data_y_cat=y_test_labels_cat, # Can be None if test_gen is used
            test_generator=test_gen, # Pass the test generator
            class_label_names=actual_classes_from_encoder, 
            train_history_data=training_history, 
            model_type_name=model_type_str,
            reports_output_dir=model_report_dir,
            # use_test_generator_for_eval= (test_gen is not None and (X_test_data is None or X_test_data.size == 0)) # Logic for which data to use
            use_test_generator_for_eval= (test_gen is not None and test_gen.n > 0)
        )
    else:
        logger.error("No valid model available (neither loaded nor in memory) for final evaluation.")


    pipeline_end_time = time.time()
    logger.info(f"===== {model_type_str.upper()} Cancer Model Training Pipeline Finished =====")
    logger.info(f"Total pipeline execution time: {(pipeline_end_time - pipeline_start_time) / 60:.2f} minutes.")
    
    return best_model_for_eval, training_history # Return the model (potentially best) and history


if __name__ == '__main__':
    # This block is for testing the training pipeline directly.
    # It requires dummy data setup similar to data_preprocessing.py's self-test or actual data.
    logger.info("Initiating a self-test run of the training_pipeline.py.")

    # --- Configuration for self-test ---
    # Assuming config.py is in the same parent directory (src)
    from .config import (COLON_DATA_DIR, COLON_NUM_CLASSES, COLON_CLASSES, COLON_MODEL_PATH, 
                         COLON_HISTORY_PATH, COLON_LABEL_ENCODER_PATH, COLON_DEFAULT_EPOCHS, 
                         COLON_DEFAULT_BATCH_SIZE, COLON_REPORT_DIR, PROJECT_ROOT)

    # Create minimal dummy data for the colon model to run the pipeline
    from PIL import Image # For creating dummy images
    
    def setup_minimal_pipeline_test_data(data_dir_path, class_list, num_img_per_cls=30): # Enough for splits
        logger.info(f"Setting up minimal dummy data in {data_dir_path} for pipeline test.")
        for cls_item in class_list:
            cls_dir_path = os.path.join(data_dir_path, cls_item)
            os.makedirs(cls_dir_path, exist_ok=True)
            for i_img in range(num_img_per_cls):
                try:
                    # Create a tiny valid PNG image
                    img_obj = Image.new('RGB', (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]), color='cyan')
                    img_obj.save(os.path.join(cls_dir_path, f'pipeline_dummy_{cls_item}_{i_img}.png'))
                except Exception as e_img:
                    logger.error(f"Failed to create dummy image {i_img} for {cls_item}: {e_img}")
        logger.info("Minimal dummy data for pipeline test setup complete.")

    # Use a temporary directory for test outputs to avoid overwriting production models/reports
    test_model_dir = os.path.join(PROJECT_ROOT, 'trained_models', 'pipeline_test')
    test_report_dir_colon = os.path.join(PROJECT_ROOT, 'reports', 'colon', 'pipeline_test')
    os.makedirs(test_model_dir, exist_ok=True)
    os.makedirs(test_report_dir_colon, exist_ok=True)

    # Modify paths for test
    test_colon_model_path = os.path.join(test_model_dir, 'colon_cancer_model_pipeline_test.keras')
    test_colon_history_path = os.path.join(test_model_dir, 'colon_model_history_pipeline_test.pkl')
    test_colon_encoder_path = os.path.join(test_model_dir, 'colon_label_encoder_pipeline_test.pkl')
    
    # Setup dummy data specifically for colon (as an example)
    # Ensure COLON_DATA_DIR points to a location where dummy data can be created, or use a dedicated test data dir.
    # For this test, let's use a subdirectory within the existing COLON_DATA_DIR if it's safe,
    # or better, a completely separate dummy data path.
    dummy_colon_test_data_dir = os.path.join(PROJECT_ROOT, 'data', 'dummy_colon_for_pipeline_test')
    setup_minimal_pipeline_test_data(dummy_colon_test_data_dir, COLON_CLASSES)

    logger.info("Attempting to train COLON model with minimal dummy data for pipeline self-test...")
    # Use minimal epochs and batch size for quick testing
    test_epochs = 2 
    test_batch_size = 4 

    trained_model_obj, history_data_dict = execute_training_pipeline(
        model_type_str='colon',
        dataset_base_dir=dummy_colon_test_data_dir, # Use the dedicated dummy data path
        defined_num_classes=COLON_NUM_CLASSES,
        defined_class_names=COLON_CLASSES,
        target_model_path=test_colon_model_path,
        training_history_path=test_colon_history_path,
        label_enc_path=test_colon_encoder_path,
        model_report_dir=test_report_dir_colon, # Specific test report dir
        num_epochs=test_epochs,
        train_batch_size=test_batch_size,
        use_transfer_learning_flag=False # Test custom CNN first
    )

    if trained_model_obj and history_data_dict:
        logger.info("Colon model pipeline self-test completed successfully.")
        logger.info(f"Test model saved to: {test_colon_model_path}")
        logger.info(f"Test reports/plots in: {test_report_dir_colon}")
    else:
        logger.error("Colon model pipeline self-test failed.")

    # --- Cleanup of dummy data and test outputs (optional) ---
    import shutil
    if os.path.exists(dummy_colon_test_data_dir):
        shutil.rmtree(dummy_colon_test_data_dir)
        logger.info(f"Cleaned up dummy data directory: {dummy_colon_test_data_dir}")
    # Test output models/reports in 'pipeline_test' subdirs can be manually reviewed/deleted.
    # if os.path.exists(test_model_dir): shutil.rmtree(test_model_dir)
    # if os.path.exists(test_report_dir_colon): shutil.rmtree(test_report_dir_colon)
    
    logger.info("Training pipeline self-test finished.")
