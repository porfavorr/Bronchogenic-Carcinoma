"""
Model evaluation module.
Includes functions for:
- Plotting training history (accuracy and loss curves).
- Generating and saving classification reports.
- Plotting and saving confusion matrices.
- A comprehensive function to perform all evaluations.
"""
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt # Keep for direct use if needed, though utils.save_training_plot is preferred

from config import get_logger
from utils import save_training_plot, plot_and_save_confusion_matrix # Using refined utils functions

logger = get_logger(__name__)

def plot_and_save_training_history(training_history_data, model_type_name, reports_output_dir):
    """
    Plots and saves training accuracy and loss curves using utility functions.
    Args:
        training_history_data (dict): Keras training history (model.history.history).
        model_type_name (str): 'colon' or 'lung', for naming output files.
        reports_output_dir (str): Directory to save the plots.
    """
    if not training_history_data:
        logger.warning(f"No training history data provided for {model_type_name}. Skipping history plots.")
        return

    # Ensure the specific report directory exists (e.g., reports/colon/ or reports/lung/)
    os.makedirs(reports_output_dir, exist_ok=True)
    
    # Define paths for individual plots
    accuracy_plot_path = os.path.join(reports_output_dir, f'{model_type_name}_training_accuracy_plot.png')
    loss_plot_path = os.path.join(reports_output_dir, f'{model_type_name}_training_loss_plot.png')

    # Use utility functions to save plots
    if 'accuracy' in training_history_data and 'val_accuracy' in training_history_data:
        save_training_plot(training_history_data, 'accuracy', accuracy_plot_path)
    else:
        logger.warning(f"Accuracy/val_accuracy not found in history for {model_type_name}. Skipping accuracy plot.")
        
    if 'loss' in training_history_data and 'val_loss' in training_history_data:
        save_training_plot(training_history_data, 'loss', loss_plot_path)
    else:
        logger.warning(f"Loss/val_loss not found in history for {model_type_name}. Skipping loss plot.")

    # Optional: Combined plot (if desired, implement similar to previous versions or keep separate)
    # For now, focusing on separate plots via utils for clarity.
    # combined_plot_path = os.path.join(reports_output_dir, f'{model_type_name}_training_history_combined.png')
    # (Code for combined plot would go here if needed)


def generate_cls_report_and_cm(
    keras_model, 
    x_data_test, # Can be None if test_gen is provided
    y_data_test_categorical, # Can be None if test_gen is provided
    test_data_generator, # Keras ImageDataGenerator for test set
    class_labels_list, 
    model_type_name, 
    reports_output_dir,
    use_generator_for_prediction=False):
    """
    Generates, logs, and saves a classification report and a confusion matrix heatmap.
    Args:
        keras_model (tensorflow.keras.Model): The trained Keras model.
        x_data_test (numpy.ndarray or None): Test data features. Used if use_generator_for_prediction is False.
        y_data_test_categorical (numpy.ndarray or None): True labels for test data (one-hot encoded).
                                                        Used if use_generator_for_prediction is False.
        test_data_generator (keras.preprocessing.image.DirectoryIterator or None): Test data generator.
                                                                                Used if use_generator_for_prediction is True.
        class_labels_list (list): List of class names (e.g., ['colon_aca', 'colon_n']).
        model_type_name (str): 'colon' or 'lung', for naming output files.
        reports_output_dir (str): Directory to save the report and plot.
        use_generator_for_prediction (bool): If True, uses test_data_generator for predictions.
                                            Otherwise, uses x_data_test and y_data_test_categorical.
    """
    os.makedirs(reports_output_dir, exist_ok=True) # Ensure directory exists
    
    report_file_path = os.path.join(reports_output_dir, f'{model_type_name}_classification_report.txt')
    cm_plot_file_path = os.path.join(reports_output_dir, f'{model_type_name}_confusion_matrix.png')

    y_pred_probabilities = None
    y_true_indices = None

    try:
        if use_generator_for_prediction and test_data_generator is not None:
            if test_data_generator.n == 0:
                logger.warning(f"Test generator for {model_type_name} has 0 samples. Skipping report and CM.")
                return
            logger.info(f"Predicting using test generator ({test_data_generator.n} samples) for {model_type_name} evaluation...")
            # Ensure generator is not shuffled for evaluation; it should be set to shuffle=False during creation.
            y_pred_probabilities = keras_model.predict(test_data_generator, steps=len(test_data_generator), verbose=1)
            y_true_indices = test_data_generator.classes # True class indices from the generator
            # Ensure class_indices from generator match the order of class_labels_list if there's any doubt.
            # test_data_generator.class_indices gives a map like {'class_a': 0, 'class_b': 1}
            # y_true_indices should align with this.
        elif x_data_test is not None and y_data_test_categorical is not None and x_data_test.size > 0:
            if x_data_test.shape[0] == 0:
                logger.warning(f"X_test for {model_type_name} has 0 samples. Skipping report and CM.")
                return
            logger.info(f"Predicting on X_test ({x_data_test.shape[0]} samples) for {model_type_name} evaluation...")
            y_pred_probabilities = keras_model.predict(x_data_test, verbose=1)
            y_true_indices = np.argmax(y_data_test_categorical, axis=1) # Convert from one-hot to integer labels
        else:
            logger.error("Insufficient data for evaluation: Neither test generator nor X_test/y_test_cat provided or they are empty.")
            return

        if y_pred_probabilities is None:
            logger.error("Prediction probabilities are None. Cannot proceed with report/CM generation.")
            return

        y_pred_indices = np.argmax(y_pred_probabilities, axis=1) # Predicted class indices

        # --- Classification Report ---
        logger.info(f"Generating classification report for {model_type_name}...")
        # Ensure target_names in classification_report matches the order of y_true_indices and y_pred_indices
        cls_report_str = classification_report(y_true_indices, y_pred_indices, 
                                               target_names=class_labels_list, zero_division=0,
                                               digits=4) # Increased precision
        
        logger.info(f"\nClassification Report ({model_type_name}):\n{cls_report_str}")
        with open(report_file_path, 'w') as f:
            f.write(f"Classification Report for {model_type_name.capitalize()} Cancer Model\n")
            f.write("=" * 60 + "\n")
            f.write(cls_report_str)
        logger.info(f"Classification report saved to: {report_file_path}")

        # --- Confusion Matrix ---
        logger.info(f"Generating confusion matrix for {model_type_name}...")
        cm_array = confusion_matrix(y_true_indices, y_pred_indices)
        # Use the utility function for plotting and saving
        plot_and_save_confusion_matrix(cm_array, class_labels_list, cm_plot_file_path, 
                                       title=f'{model_type_name.capitalize()} Model Confusion Matrix')

    except Exception as e:
        logger.error(f"Error generating evaluation report/CM for {model_type_name}: {e}", exc_info=True)


def evaluate_full_model_performance(
    model, 
    test_data_x, # Can be None
    test_data_y_cat, # Can be None
    test_generator, # Can be None
    class_label_names, 
    train_history_data, 
    model_type_name, 
    reports_output_dir,
    skip_report_and_cm=False, # Flag to only plot history
    use_test_generator_for_eval=False): # Explicit flag to use generator
    """
    Comprehensive evaluation function: plots training history, generates classification report, and plots confusion matrix.
    Args:
        model (tensorflow.keras.Model or None): Trained Keras model. Can be None if skip_report_and_cm is True.
        test_data_x (numpy.ndarray or None): Test data features.
        test_data_y_cat (numpy.ndarray or None): True labels for test data (one-hot encoded).
        test_generator (keras.preprocessing.image.DirectoryIterator or None): Test data generator.
        class_label_names (list): List of class names for reports.
        train_history_data (dict or None): Training history (output of model.fit().history).
        model_type_name (str): 'colon' or 'lung', for file naming and titles.
        reports_output_dir (str): Directory to save all evaluation outputs.
        skip_report_and_cm (bool): If True, only plots training history.
        use_test_generator_for_eval (bool): If True and test_generator is provided, it will be used for predictions.
    """
    logger.info(f"--- Initiating Full Model Performance Evaluation for {model_type_name.upper()} ---")
    
    # 1. Plot training history (if available)
    if train_history_data:
        logger.info("Plotting training history...")
        plot_and_save_training_history(train_history_data, model_type_name, reports_output_dir)
    else:
        logger.info(f"No training history data provided for {model_type_name}. Skipping history plots.")

    if skip_report_and_cm:
        logger.info("Skipping classification report and confusion matrix generation as requested.")
        logger.info(f"--- Evaluation for {model_type_name.upper()} Finished (History Plots Only) ---")
        return

    if model is None:
        logger.error("Keras model object is None. Cannot generate classification report or confusion matrix.")
        logger.info(f"--- Evaluation for {model_type_name.upper()} Finished (Error: No Model) ---")
        return
        
    # 2. Generate Classification Report and Confusion Matrix
    logger.info("Generating classification report and confusion matrix...")
    generate_cls_report_and_cm(
        keras_model=model, 
        x_data_test=test_data_x, 
        y_data_test_categorical=test_data_y_cat, 
        test_data_generator=test_generator,
        class_labels_list=class_label_names, 
        model_type_name=model_type_name, 
        reports_output_dir=reports_output_dir,
        use_generator_for_prediction=use_test_generator_for_eval # Pass the flag
    )

    logger.info(f"--- Full Model Performance Evaluation for {model_type_name.upper()} Finished ---")


if __name__ == '__main__':
    logger.info("Evaluation module self-test initiated.")
    
    # --- Dummy data for testing plotting functions ---
    dummy_train_hist = {
        'accuracy': [0.65, 0.75, 0.85, 0.92],
        'val_accuracy': [0.60, 0.72, 0.81, 0.89],
        'loss': [0.90, 0.60, 0.40, 0.25],
        'val_loss': [1.00, 0.70, 0.50, 0.35]
    }
    dummy_cls_names = ['Negative', 'Positive'] # Example for a binary case
    
    # Define PROJECT_ROOT if not available (e.g., when running file directly)
    if 'PROJECT_ROOT' not in globals():
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    test_eval_reports_dir = os.path.join(PROJECT_ROOT, 'reports', 'evaluation_module_self_test')
    os.makedirs(test_eval_reports_dir, exist_ok=True)
    logger.info(f"Self-test reports will be saved in: {test_eval_reports_dir}")

    # 1. Test plotting training history
    logger.info("\n--- Testing plot_and_save_training_history ---")
    plot_and_save_training_history(dummy_train_hist, 'self_test_model', test_eval_reports_dir)
    logger.info("Training history plots generated for 'self_test_model'. Please review them.")

    # 2. Test generate_cls_report_and_cm (requires a mock model and data)
    logger.info("\n--- Testing generate_cls_report_and_cm with mock model & data ---")
    
    # Mocking a Keras model's predict method
    class MockKerasModel:
        def predict(self, data_input, verbose=0, steps=None):
            num_samples = data_input.shape[0] if hasattr(data_input, 'shape') else (steps * 2 if steps else 10) #Approximate for generator
            # Return random predictions for 2 classes (matching dummy_cls_names)
            return np.random.rand(num_samples, len(dummy_cls_names)) 

    mock_model_instance = MockKerasModel()
    
    # Dummy test data (X_test, y_test_cat)
    num_test_samples = 50
    dummy_X_test_data = np.random.rand(num_test_samples, 128, 128, 3) # Shape like (samples, H, W, C)
    dummy_y_true_int = np.random.randint(0, len(dummy_cls_names), num_test_samples)
    from tensorflow.keras.utils import to_categorical
    dummy_y_test_cat_data = to_categorical(dummy_y_true_int, num_classes=len(dummy_cls_names))

    generate_cls_report_and_cm(
        keras_model=mock_model_instance, 
        x_data_test=dummy_X_test_data, 
        y_data_test_categorical=dummy_y_test_cat_data,
        test_data_generator=None, # Testing with X_test, y_test first
        class_labels_list=dummy_cls_names, 
        model_type_name='mock_report_cm_test', 
        reports_output_dir=test_eval_reports_dir,
        use_generator_for_prediction=False
    )
    logger.info("generate_cls_report_and_cm test with mock model and X_test/y_test completed.")
    # Add a test case for using test_data_generator if possible (more complex to mock ImageDataGenerator)

    # 3. Test comprehensive evaluation function
    logger.info("\n--- Testing evaluate_full_model_performance ---")
    evaluate_full_model_performance(
        model=mock_model_instance, 
        test_data_x=dummy_X_test_data, 
        test_data_y_cat=dummy_y_test_cat_data,
        test_generator=None, # Not using generator for this specific test call
        class_label_names=dummy_cls_names,
        train_history_data=dummy_train_hist, 
        model_type_name='comprehensive_self_test', 
        reports_output_dir=test_eval_reports_dir,
        use_test_generator_for_eval=False
    )
    logger.info("Comprehensive evaluation self-test completed.")
    
    # Cleanup (optional, for automated test environments)
    # import shutil
    # if os.path.exists(test_eval_reports_dir):
    #     shutil.rmtree(test_eval_reports_dir)
    #     logger.info(f"Cleaned up self-test report directory: {test_eval_reports_dir}")
        
    logger.info("Evaluation module self-test finished. Review generated files in reports/evaluation_module_self_test.")
