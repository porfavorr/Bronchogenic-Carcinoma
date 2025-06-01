"""
Data preprocessing module for loading image datasets, performing augmentation,
and creating data generators for training, validation, and testing.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # For one-hot encoding labels manually if needed

from config import (TARGET_IMAGE_SIZE, RANDOM_STATE, VALIDATION_SPLIT, TEST_SPLIT,
                     get_logger)
from utils import save_pickle_object, load_and_preprocess_image_pil # Using PIL loader

logger = get_logger(__name__)

def load_image_paths_and_labels_df(data_directory, expected_class_names):
    """
    Scans a directory structure to load image filepaths and their corresponding labels.
    Assumes `data_directory` contains subdirectories, each named after a class.
    Args:
        data_directory (str): The root directory of the image dataset.
        expected_class_names (list): A list of subdirectory names that represent the classes.
    Returns:
        pandas.DataFrame: A DataFrame with 'filepath' and 'label' columns.
                          Returns None if the data_directory is invalid or no images are found.
    """
    filepaths = []
    labels = []
    
    if not os.path.isdir(data_directory):
        logger.error(f"Dataset directory not found: {data_directory}")
        return None

    logger.info(f"Scanning for images in {data_directory} using class names: {expected_class_names}")
    found_classes = []

    for class_name in expected_class_names:
        class_path = os.path.join(data_directory, class_name)
        if not os.path.isdir(class_path):
            logger.warning(f"Class directory '{class_path}' not found. Skipping class '{class_name}'.")
            continue
        
        num_images_in_class = 0
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                filepaths.append(os.path.join(class_path, img_file))
                labels.append(class_name)
                num_images_in_class += 1
        
        if num_images_in_class > 0:
            logger.info(f"Found {num_images_in_class} images for class '{class_name}'.")
            found_classes.append(class_name)
        else:
            logger.warning(f"No image files found in class directory: {class_path}")

    if not filepaths:
        logger.error(f"No images found in any of the specified class subdirectories within {data_directory}.")
        return None

    if set(found_classes) != set(expected_class_names):
        logger.warning(f"Mismatch between expected classes and found classes. Expected: {expected_class_names}, Found: {found_classes}")
        # Decide if this should be an error or just a warning. For now, proceed if some data is found.

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    logger.info(f"Successfully loaded {len(df)} image paths and labels from {len(found_classes)} directories.")
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle DataFrame

def create_and_save_label_encoder(labels_series, encoder_save_path):
    """
    Fits a LabelEncoder on the given labels, saves it, and returns the encoder and encoded labels.
    Args:
        labels_series (pd.Series): A Pandas Series containing string labels.
        encoder_save_path (str): Path to save the fitted LabelEncoder object.
    Returns:
        tuple: (LabelEncoder instance, encoded integer labels) or (None, None) on error.
    """
    encoder = LabelEncoder()
    try:
        encoded_labels_int = encoder.fit_transform(labels_series)
        save_pickle_object(encoder, encoder_save_path)
        logger.info(f"LabelEncoder fitted and saved to {encoder_save_path}. Classes: {list(encoder.classes_)}")
        return encoder, encoded_labels_int
    except Exception as e:
        logger.error(f"Error fitting or saving LabelEncoder: {e}", exc_info=True)
        return None, None

def prepare_data_generators(dataframe, label_encoder, target_img_size=TARGET_IMAGE_SIZE, batch_sz=32):
    """
    Splits data into train, validation, and test sets, then creates Keras ImageDataGenerators.
    Args:
        dataframe (pd.DataFrame): DataFrame with 'filepath' and 'label' (original string labels).
        label_encoder (LabelEncoder): A pre-fitted LabelEncoder instance.
        target_img_size (tuple): Target image dimensions (width, height).
        batch_sz (int): Batch size for the generators.
    Returns:
        tuple: (train_generator, validation_generator, test_generator, X_test_arr, y_test_cat_arr)
               Returns (None, None, None, None, None) on error or if splits are too small.
    """
    if dataframe is None or dataframe.empty:
        logger.error("Input DataFrame is empty. Cannot create data generators.")
        return None, None, None, None, None
    if label_encoder is None:
        logger.error("LabelEncoder is None. Cannot proceed with data generator creation.")
        return None, None, None, None, None

    # Use original string labels for stratifying splits, ImageDataGenerator will handle encoding via 'classes' param.
    # Splitting: First, separate out the test set.
    if len(dataframe) * TEST_SPLIT < 1:
        logger.warning(f"Dataset too small for a test split of {TEST_SPLIT*100}%. Test set will be empty or very small.")
        # Handle very small datasets: perhaps no test set, or merge test with validation.
        # For now, proceed, but test_df might be empty.
    
    # Ensure there are enough samples for stratification, at least one per class for each split.
    # This can be complex with small multi-class datasets.
    # A simpler check: if any class has fewer than 2 samples, stratification might fail for train/val/test.
    min_samples_per_class = dataframe['label'].value_counts().min()
    if min_samples_per_class < 2 and (VALIDATION_SPLIT > 0 or TEST_SPLIT > 0) : # Need at least 2 for a split if both are >0
        logger.warning(f"One or more classes have fewer than 2 samples ({min_samples_per_class}). "
                       "Stratified splitting might be problematic or lead to empty splits. "
                       "Consider a larger dataset or simpler splitting strategy.")

    try:
        train_val_df, test_df = train_test_split(
            dataframe,
            test_size=TEST_SPLIT,
            stratify=dataframe['label'], # Stratify by original string labels
            random_state=RANDOM_STATE
        )
    except ValueError as e_stratify: # Handles cases where stratification is not possible
        logger.error(f"Stratification error during train/test split: {e_stratify}. This often happens with very small classes.")
        logger.info("Attempting split without stratification for train/test due to error.")
        train_val_df, test_df = train_test_split(dataframe, test_size=TEST_SPLIT, random_state=RANDOM_STATE)


    # Then, split the remaining data into training and validation sets.
    # Adjust validation split size because it's a percentage of train_val_df, not the original df.
    adjusted_val_split = VALIDATION_SPLIT / (1 - TEST_SPLIT) if (1 - TEST_SPLIT) > 0 else VALIDATION_SPLIT
    if len(train_val_df) * adjusted_val_split < 1 and VALIDATION_SPLIT > 0:
         logger.warning(f"Train/validation set too small for validation split of {VALIDATION_SPLIT*100}%. Validation set may be empty.")
    
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_split,
            stratify=train_val_df['label'], # Stratify by original string labels
            random_state=RANDOM_STATE
        )
    except ValueError as e_stratify_tv:
        logger.error(f"Stratification error during train/validation split: {e_stratify_tv}.")
        logger.info("Attempting split without stratification for train/validation due to error.")
        train_df, val_df = train_test_split(train_val_df, test_size=adjusted_val_split, random_state=RANDOM_STATE)


    logger.info(f"Dataset split complete: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)}) samples.")

    if train_df.empty:
        logger.error("Training set is empty after splits. Cannot proceed. Check dataset size and split ratios.")
        return None, None, None, None, None
    # Validation can be empty if VALIDATION_SPLIT is 0 or dataset is tiny.
    if val_df.empty and VALIDATION_SPLIT > 0:
        logger.warning("Validation set is empty after splits. Model will train without validation data if VALIDATION_SPLIT > 0.")
        # If validation is critical, this could be an error. For now, a warning.

    # --- ImageDataGenerator Setup ---
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixel values to [0,1]
        rotation_range=30,           # Randomly rotate images by up to 30 degrees
        width_shift_range=0.2,       # Randomly shift images horizontally by up to 20% of width
        height_shift_range=0.2,      # Randomly shift images vertically by up to 20% of height
        shear_range=0.2,             # Shear intensity (shear angle in counter-clockwise direction in radians)
        zoom_range=0.2,              # Randomly zoom image by up to 20%
        horizontal_flip=True,        # Randomly flip images horizontally
        fill_mode='nearest'          # Strategy for filling newly created pixels (e.g., after rotation or shift)
    )

    # Validation and Test data generator (only rescaling, no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # --- Flow from DataFrame ---
    # `classes` parameter in flow_from_dataframe ensures consistent class indexing.
    # It expects a list of the class names in the desired order.
    le_classes = list(label_encoder.classes_)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label', # Use original string labels
        classes=le_classes,
        target_size=target_img_size, # (width, height)
        batch_size=batch_sz,
        class_mode='categorical', # For multi-class classification, yields one-hot encoded labels
        shuffle=True,
        seed=RANDOM_STATE
    )

    validation_generator = None
    if not val_df.empty:
        validation_generator = val_test_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='filepath',
            y_col='label',
            classes=le_classes,
            target_size=target_img_size,
            batch_size=batch_sz,
            class_mode='categorical',
            shuffle=False # No need to shuffle validation data
        )
    else:
        logger.info("Validation DataFrame is empty, so validation_generator will be None.")


    test_generator = None
    X_test_arr, y_test_cat_arr = np.array([]), np.array([]) # Initialize as empty

    if not test_df.empty:
        test_generator = val_test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filepath',
            y_col='label',
            classes=le_classes,
            target_size=target_img_size,
            batch_size=batch_sz, # Can be 1 or batch_size for evaluation
            class_mode='categorical',
            shuffle=False # Important for consistent evaluation
        )
        
        # Also prepare X_test, y_test arrays for scikit-learn metrics if needed,
        # though model.evaluate can use the generator directly.
        # This can be memory-intensive for large test sets.
        logger.info("Preparing X_test and y_test arrays from test_df (can be memory intensive)...")
        temp_X_test = []
        temp_y_test_labels = [] # Store original string labels first
        for _, row in test_df.iterrows():
            img_array = load_and_preprocess_image_pil(row['filepath'], target_size=target_img_size)
            if img_array is not None:
                temp_X_test.append(img_array)
                temp_y_test_labels.append(row['label'])
        
        if temp_X_test:
            X_test_arr = np.array(temp_X_test)
            # Encode these labels using the *fitted* label_encoder
            y_test_int_arr = label_encoder.transform(temp_y_test_labels)
            y_test_cat_arr = to_categorical(y_test_int_arr, num_classes=len(le_classes))
            logger.info(f"Created X_test array with shape {X_test_arr.shape} and y_test_cat array with shape {y_test_cat_arr.shape}")
        else:
            logger.warning("X_test array is empty, possibly due to image loading issues from test_df.")
    else:
        logger.info("Test DataFrame is empty, so test_generator, X_test_arr, y_test_cat_arr will be empty/None.")

    # Sanity check class indices
    if train_generator.class_indices != {cls_name: i for i, cls_name in enumerate(le_classes)}:
        logger.warning(f"Mismatch in class indices! Train Generator: {train_generator.class_indices}, "
                       f"Expected from LabelEncoder: {{cls_name: i for i, cls_name in enumerate(le_classes)}}. "
                       f"This might lead to incorrect label mapping during training/evaluation.")

    logger.info(f"Data generators created. Train batches: {len(train_generator)}. "
                f"Validation batches: {len(validation_generator) if validation_generator else 0}. "
                f"Test batches: {len(test_generator) if test_generator else 0}.")
    
    return train_generator, validation_generator, test_generator, X_test_arr, y_test_cat_arr


if __name__ == '__main__':
    logger.info("Data Preprocessing module self-test initiated.")
    
    # --- Setup for self-test ---
    # Requires PROJECT_ROOT to be defined, e.g. by importing from config or defining here
    if 'PROJECT_ROOT' not in globals():
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    dummy_data_root = os.path.join(PROJECT_ROOT, 'data', 'dummy_test_dataset')
    dummy_colon_dir = os.path.join(dummy_data_root, 'colon_images')
    dummy_colon_classes = ['colon_healthy', 'colon_cancer']
    dummy_encoder_save_path = os.path.join(PROJECT_ROOT, 'trained_models', 'dummy_colon_encoder_test.pkl')

    # Create dummy data structure and a few tiny image files for testing
    def setup_dummy_image_data(base_dir, classes, num_images_per_class=25): # Increased for better split testing
        logger.info(f"Setting up dummy image data in {base_dir}...")
        for cls_name in classes:
            cls_path = os.path.join(base_dir, cls_name)
            os.makedirs(cls_path, exist_ok=True)
            for i in range(num_images_per_class):
                try:
                    # Create a tiny valid PNG image
                    img = Image.new('RGB', (20, 20), color='blue' if i % 2 == 0 else 'green')
                    img.save(os.path.join(cls_path, f'dummy_{cls_name}_{i+1}.png'))
                except Exception as e:
                    logger.error(f"Failed to create dummy image {i+1} for {cls_name}: {e}")
        logger.info("Dummy image data setup complete.")

    setup_dummy_image_data(dummy_colon_dir, dummy_colon_classes)

    # 1. Test load_image_paths_and_labels_df
    logger.info("\n--- Testing load_image_paths_and_labels_df ---")
    df_colon = load_image_paths_and_labels_df(dummy_colon_dir, dummy_colon_classes)
    if df_colon is not None and not df_colon.empty:
        logger.info(f"Loaded DataFrame head:\n{df_colon.head()}")
        logger.info(f"Value counts:\n{df_colon['label'].value_counts()}")
        assert len(df_colon) == len(dummy_colon_classes) * 25 # 25 images per class

        # 2. Test create_and_save_label_encoder
        logger.info("\n--- Testing create_and_save_label_encoder ---")
        colon_le, encoded_labels = create_and_save_label_encoder(df_colon['label'], dummy_encoder_save_path)
        if colon_le and encoded_labels is not None:
            assert len(colon_le.classes_) == len(dummy_colon_classes)
            logger.info(f"LabelEncoder classes: {list(colon_le.classes_)}")
            logger.info(f"First 5 encoded labels: {encoded_labels[:5]}")

            # 3. Test prepare_data_generators
            logger.info("\n--- Testing prepare_data_generators ---")
            # Note: TARGET_IMAGE_SIZE is from config, ensure it's (width, height)
            train_gen, val_gen, test_gen, X_test, y_test = prepare_data_generators(
                df_colon, colon_le, target_img_size=TARGET_IMAGE_SIZE, batch_sz=8 # Small batch for test
            )
            
            if train_gen: # val_gen or test_gen might be None if splits are too small or VAL/TEST_SPLIT is 0
                logger.info(f"Train generator: {train_gen.n} samples, {train_gen.num_classes} classes, {len(train_gen)} batches.")
                logger.info(f"Train generator class indices: {train_gen.class_indices}")
                
                # Try to get a batch
                try:
                    sample_batch_x, sample_batch_y = next(train_gen)
                    logger.info(f"Sample batch X shape from train_gen: {sample_batch_x.shape}") # (batch, H, W, C)
                    logger.info(f"Sample batch Y shape from train_gen: {sample_batch_y.shape}") # (batch, num_classes)
                    assert sample_batch_x.shape[1:] == (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3) # H, W, C
                except Exception as e_batch:
                    logger.error(f"Could not get a batch from train_generator: {e_batch}")

            if val_gen:
                logger.info(f"Validation generator: {val_gen.n} samples, {val_gen.num_classes} classes, {len(val_gen)} batches.")
            if test_gen:
                logger.info(f"Test generator: {test_gen.n} samples, {test_gen.num_classes} classes, {len(test_gen)} batches.")
            if X_test.size > 0 and y_test.size > 0:
                 logger.info(f"X_test array shape: {X_test.shape}, y_test_cat array shape: {y_test.shape}")
            else:
                logger.info("X_test/y_test arrays are empty, as expected if test_df was empty.")

        else:
            logger.error("LabelEncoder creation failed. Skipping data generator tests.")
    else:
        logger.error("DataFrame loading failed. Aborting further tests in data_preprocessing.")

    # --- Cleanup ---
    import shutil
    if os.path.exists(dummy_data_root):
        shutil.rmtree(dummy_data_root)
        logger.info(f"Cleaned up dummy data directory: {dummy_data_root}")
    if os.path.exists(dummy_encoder_save_path):
        os.remove(dummy_encoder_save_path)
        logger.info(f"Cleaned up dummy encoder: {dummy_encoder_save_path}")
    
    logger.info("Data Preprocessing module self-test completed.")
