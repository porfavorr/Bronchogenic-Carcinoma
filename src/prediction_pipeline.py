"""
Prediction pipeline for making classifications on single images using trained models.
This module provides a `Predictor` class that handles loading the appropriate
model and label encoder, preprocessing an input image, and returning the prediction.
"""
import numpy as np
from tensorflow.keras.models import load_model
import io # For handling image bytes from uploads

from config import (TARGET_IMAGE_SIZE, 
                     COLON_MODEL_PATH, COLON_LABEL_ENCODER_PATH, COLON_CLASSES,
                     LUNG_MODEL_PATH, LUNG_LABEL_ENCODER_PATH, LUNG_CLASSES,
                     get_logger)
from utils import load_and_preprocess_image_pil, load_pickle_object

logger = get_logger(__name__)

class ImagePredictor:
    """
    Handles loading a trained model and its associated label encoder for making predictions on images.
    """
    def __init__(self, model_type_identifier):
        """
        Initializes the predictor for a specific model type ('colon' or 'lung').
        Args:
            model_type_identifier (str): The type of model to load, e.g., "colon" or "lung".
        """
        self.model_type = model_type_identifier.lower()
        self.model = None
        self.label_encoder = None
        self.class_names = None # Store class names for this predictor

        self._load_model_and_encoder()

    def _load_model_and_encoder(self):
        """
        Loads the Keras model and the corresponding LabelEncoder based on the model_type.
        """
        logger.info(f"Initializing ImagePredictor for '{self.model_type}' model...")
        
        model_file_path = None
        encoder_file_path = None

        if self.model_type == 'colon':
            model_file_path = COLON_MODEL_PATH
            encoder_file_path = COLON_LABEL_ENCODER_PATH
            self.class_names = COLON_CLASSES
        elif self.model_type == 'lung':
            model_file_path = LUNG_MODEL_PATH
            encoder_file_path = LUNG_LABEL_ENCODER_PATH
            self.class_names = LUNG_CLASSES
        else:
            logger.error(f"Invalid model type specified for Predictor: '{self.model_type}'. "
                           "Choose from 'colon' or 'lung'.")
            # Raise an error or set a flag to indicate failure
            raise ValueError(f"Unsupported model type for ImagePredictor: {self.model_type}")

        # Load the Keras model
        try:
            self.model = load_model(model_file_path)
            logger.info(f"Keras model for '{self.model_type}' loaded successfully from: {model_file_path}")
        except Exception as e:
            logger.error(f"Error loading Keras model for '{self.model_type}' from {model_file_path}: {e}", exc_info=True)
            self.model = None # Ensure model is None if loading fails

        # Load the LabelEncoder
        self.label_encoder = load_pickle_object(encoder_file_path)
        if self.label_encoder:
            logger.info(f"LabelEncoder for '{self.model_type}' loaded successfully from: {encoder_file_path}")
            # Verify consistency between loaded encoder classes and config classes
            if set(self.label_encoder.classes_) != set(self.class_names):
                logger.warning(f"Mismatch! LabelEncoder classes ({list(self.label_encoder.classes_)}) "
                               f"do not perfectly match configured classes ({self.class_names}) for '{self.model_type}'. "
                               "This could lead to incorrect class name mapping if config was updated after encoder was saved.")
                # Update self.class_names to reflect what's in the encoder for safety, as encoder is ground truth for predictions
                self.class_names = list(self.label_encoder.classes_)
                logger.info(f"Using class names from loaded LabelEncoder for '{self.model_type}': {self.class_names}")
        else:
            logger.error(f"Failed to load LabelEncoder for '{self.model_type}' from {encoder_file_path}.")
            self.label_encoder = None # Ensure encoder is None

    def check_readiness(self):
        """
        Checks if both the model and label encoder have been successfully loaded.
        Returns:
            bool: True if ready for prediction, False otherwise.
        """
        if self.model is None:
            logger.warning(f"Predictor for '{self.model_type}' is NOT ready: Model not loaded.")
            return False
        if self.label_encoder is None:
            logger.warning(f"Predictor for '{self.model_type}' is NOT ready: LabelEncoder not loaded.")
            return False
        return True

    def predict_single_image(self, image_source):
        """
        Preprocesses a single image and predicts its class using the loaded model.
        Args:
            image_source (str or BytesIO): Path to the image file or a BytesIO object containing image data.
        Returns:
            tuple: (predicted_class_name_str, confidence_score_float) or (None, None) if prediction fails.
        """
        if not self.check_readiness():
            logger.error(f"Cannot predict. Predictor for '{self.model_type}' is not ready.")
            return None, None

        # Preprocess the image using the utility function
        # TARGET_IMAGE_SIZE should be (width, height) as expected by load_and_preprocess_image_pil
        preprocessed_img_array = load_and_preprocess_image_pil(image_source, target_size=TARGET_IMAGE_SIZE)
        
        if preprocessed_img_array is None:
            logger.error("Image preprocessing failed. Cannot make a prediction.")
            return None, None

        # Add batch dimension (model expects batch_size, height, width, channels)
        # preprocessed_img_array is (height, width, channels) from load_and_preprocess_image_pil
        img_for_prediction = np.expand_dims(preprocessed_img_array, axis=0)
        
        try:
            prediction_probabilities_array = self.model.predict(img_for_prediction, verbose=0)[0] # Get probabilities for the single image
            
            predicted_class_index = np.argmax(prediction_probabilities_array)
            confidence_value = float(np.max(prediction_probabilities_array)) # Convert to standard float
            
            # Decode the predicted class index to its string name using the label encoder
            predicted_class_name_str = self.label_encoder.inverse_transform([predicted_class_index])[0]
            
            logger.info(f"Prediction for '{self.model_type}': Class='{predicted_class_name_str}', Confidence={confidence_value*100:.2f}%")
            return predicted_class_name_str, confidence_value
            
        except Exception as e:
            logger.error(f"Error during model prediction or result processing for '{self.model_type}': {e}", exc_info=True)
            return None, None

# --- Global Predictor Instances (Lazy Loaded) ---
# This dictionary will store initialized predictor instances to avoid reloading models.
_GLOBAL_PREDICTORS_CACHE = {
    "colon": None,
    "lung": None
}

def get_cached_predictor(model_type_to_get):
    """
    Factory function to get or create (and cache) a Predictor instance.
    Args:
        model_type_to_get (str): 'colon' or 'lung'.
    Returns:
        ImagePredictor instance or None if initialization fails.
    """
    model_type_key = model_type_to_get.lower()
    if model_type_key not in _GLOBAL_PREDICTORS_CACHE:
        logger.error(f"Invalid model type '{model_type_key}' requested for predictor cache.")
        return None
        
    if _GLOBAL_PREDICTORS_CACHE[model_type_key] is None:
        logger.info(f"Predictor for '{model_type_key}' not found in cache. Attempting to create and cache now...")
        try:
            _GLOBAL_PREDICTORS_CACHE[model_type_key] = ImagePredictor(model_type_key)
            # After creation, immediately check its readiness. If not ready, don't cache a bad instance.
            if not _GLOBAL_PREDICTORS_CACHE[model_type_key].check_readiness():
                 logger.error(f"Failed to properly initialize and make predictor ready for '{model_type_key}'. Removing from cache.")
                 _GLOBAL_PREDICTORS_CACHE[model_type_key] = None # Set back to None if not ready
                 return None
            logger.info(f"Successfully created and cached predictor for '{model_type_key}'.")
        except ValueError: # Raised by ImagePredictor for unsupported model_type
            # Error already logged by ImagePredictor constructor
            _GLOBAL_PREDICTORS_CACHE[model_type_key] = None # Ensure it's None
            return None
        except Exception as e_init: # Catch any other unexpected errors during init
            logger.error(f"Unexpected exception during predictor initialization for '{model_type_key}': {e_init}", exc_info=True)
            _GLOBAL_PREDICTORS_CACHE[model_type_key] = None
            return None
    
    # Return the cached predictor (it might be None if previous initialization failed)
    return _GLOBAL_PREDICTORS_CACHE[model_type_key]


if __name__ == '__main__':
    logger.info("Prediction Pipeline (ImagePredictor) module self-test initiated.")

    # This self-test assumes that dummy/actual models and encoders might exist from training pipeline tests.
    # For a robust standalone test, we should ensure placeholder files are created if missing.
    from .utils import save_pickle_object # For creating dummy encoder
    from tensorflow.keras.models import Sequential # For dummy model
    from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
    from sklearn.preprocessing import LabelEncoder
    from PIL import Image
    import os

    def create_test_resources_if_needed(model_p, encoder_p, classes_list, input_h, input_w, input_c):
        # Create dummy Keras model if it doesn't exist
        if not os.path.exists(model_p):
            logger.warning(f"Dummy model for test not found at {model_p}. Creating a placeholder.")
            dummy_m = Sequential([
                Input(shape=(input_h, input_w, input_c)), # H, W, C
                Conv2D(2, (3,3), activation='relu', padding='same'), MaxPooling2D(), Flatten(),
                Dense(len(classes_list), activation='softmax')
            ])
            dummy_m.compile(optimizer='adam', loss='categorical_crossentropy')
            dummy_m.save(model_p)
            logger.info(f"Placeholder Keras model saved to {model_p}")

        # Create dummy LabelEncoder if it doesn't exist
        if not os.path.exists(encoder_p):
            logger.warning(f"Dummy LabelEncoder for test not found at {encoder_p}. Creating a placeholder.")
            dummy_le = LabelEncoder()
            dummy_le.fit(classes_list)
            save_pickle_object(dummy_le, encoder_p)
            logger.info(f"Placeholder LabelEncoder saved to {encoder_p}")

    # --- Test Colon Predictor ---
    logger.info("\n--- Testing Colon Predictor ---")
    create_test_resources_if_needed(COLON_MODEL_PATH, COLON_LABEL_ENCODER_PATH, COLON_CLASSES, 
                                    TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, IMAGE_CHANNELS)
    
    colon_pred_instance = get_cached_predictor('colon')
    if colon_pred_instance and colon_pred_instance.check_readiness():
        logger.info("Colon predictor obtained and ready.")
        # Create a dummy image for prediction test
        dummy_colon_img_path = "dummy_colon_pred_test.png"
        try:
            img = Image.new('RGB', TARGET_IMAGE_SIZE, color='magenta') # TARGET_IMAGE_SIZE is (W,H)
            img.save(dummy_colon_img_path)
            
            pred_name, pred_conf = colon_pred_instance.predict_single_image(dummy_colon_img_path)
            if pred_name is not None and pred_conf is not None:
                logger.info(f"Colon predictor test prediction: Class='{pred_name}', Confidence={pred_conf*100:.2f}%")
                assert pred_name in colon_pred_instance.class_names, "Predicted class name not in known classes."
            else:
                logger.error("Colon predictor test prediction failed to return results.")
        except Exception as e_pred_test:
            logger.error(f"Error during colon predictor self-test image prediction: {e_pred_test}")
        finally:
            if os.path.exists(dummy_colon_img_path):
                os.remove(dummy_colon_img_path)
    else:
        logger.error("Failed to get a ready Colon predictor instance for testing.")

    # --- Test Lung Predictor ---
    logger.info("\n--- Testing Lung Predictor ---")
    create_test_resources_if_needed(LUNG_MODEL_PATH, LUNG_LABEL_ENCODER_PATH, LUNG_CLASSES,
                                    TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, IMAGE_CHANNELS)

    lung_pred_instance = get_cached_predictor('lung')
    if lung_pred_instance and lung_pred_instance.check_readiness():
        logger.info("Lung predictor obtained and ready.")
        dummy_lung_img_path = "dummy_lung_pred_test.png"
        try:
            img = Image.new('RGB', TARGET_IMAGE_SIZE, color='yellow')
            img.save(dummy_lung_img_path)
            
            pred_name, pred_conf = lung_pred_instance.predict_single_image(dummy_lung_img_path)
            if pred_name is not None and pred_conf is not None:
                logger.info(f"Lung predictor test prediction: Class='{pred_name}', Confidence={pred_conf*100:.2f}%")
                assert pred_name in lung_pred_instance.class_names
            else:
                logger.error("Lung predictor test prediction failed.")
        except Exception as e_pred_test:
            logger.error(f"Error during lung predictor self-test image prediction: {e_pred_test}")
        finally:
            if os.path.exists(dummy_lung_img_path):
                os.remove(dummy_lung_img_path)
    else:
        logger.error("Failed to get a ready Lung predictor instance for testing.")

    # Test with invalid model type
    logger.info("\n--- Testing Invalid Predictor Type ---")
    invalid_predictor = get_cached_predictor('brain') # An unsupported type
    assert invalid_predictor is None, "Predictor for invalid type 'brain' should be None."
    logger.info("Test for invalid predictor type successful (returned None as expected).")
    
    logger.info("\nPrediction Pipeline (ImagePredictor) module self-test completed.")
    # Note: Dummy model/encoder files created for this test (if they didn't exist)
    # might remain in `trained_models`. They can be manually deleted or ignored.
