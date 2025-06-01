"""
Defines CNN model architectures for cancer classification.
Includes a base CNN builder and specific functions for colon and lung cancer models.
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     BatchNormalization, Activation, Input, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.applications import VGG16 # For optional transfer learning

from config import TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, IMAGE_CHANNELS, get_logger

logger = get_logger(__name__)

def build_custom_cnn_v1(input_shape, num_classes, model_name="CustomCNN_v1"):
    """
    Builds a custom Convolutional Neural Network (CNN) model.
    This architecture is a moderately deep CNN inspired by common practices.
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        model_name (str): Name for the Keras model.
    Returns:
        tensorflow.keras.Model: Compiled Keras model.
    """
    model = Sequential(name=model_name)

    # Input Layer - implicitly defined by the first Conv2D layer's input_shape
    # model.add(Input(shape=input_shape)) # Can be explicit if preferred

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.0005))) # Added another conv
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0005))) # Added another conv
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3)) # Increased dropout

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0005))) # Added another conv
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35)) # Increased dropout

    # Block 4 (Optional, makes model deeper)
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0005)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))


    # Flattening and Dense Layers
    model.add(Flatten())
    
    model.add(Dense(512, kernel_regularizer=l2(0.001))) # Larger Dense layer
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # Standard dropout for dense layers

    model.add(Dense(256, kernel_regularizer=l2(0.001))) # Another Dense layer
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    # Determine activation and loss based on num_classes
    if num_classes == 1: # Binary classification (outputting a single probability, e.g., for sigmoid)
        # This setup is less common with ImageDataGenerator's 'categorical' mode.
        # Typically, even for binary, num_classes=2 is used with softmax.
        activation_func = 'sigmoid'
        loss_func = 'binary_crossentropy'
        logger.warning("num_classes=1 detected. Using sigmoid activation and binary_crossentropy. "
                       "Ensure labels are 0 or 1, not one-hot encoded for this setup.")
    elif num_classes == 2: # Binary classification often treated as 2 classes with softmax
        activation_func = 'softmax'
        loss_func = 'categorical_crossentropy' # Expects one-hot encoded labels
    else: # Multi-class classification
        activation_func = 'softmax'
        loss_func = 'categorical_crossentropy' # Expects one-hot encoded labels
        
    model.add(Dense(num_classes, activation=activation_func))

    # Compile the model
    optimizer = Adam(learning_rate=0.0001) # Slightly lower initial learning rate
    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy']) # Common metrics: accuracy, Precision, Recall, AUC

    logger.info(f"Custom CNN model '{model_name}' created successfully.")
    logger.info(f"Input shape: {input_shape}, Output classes: {num_classes}")
    logger.info(f"Output layer activation: '{activation_func}', Loss function: '{loss_func}'")
    
    # Log model summary (can be lengthy, so use logger.debug or a flag if too verbose for INFO)
    # model.summary(print_fn=logger.info) 
    
    return model

def get_colon_cancer_model(input_shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, IMAGE_CHANNELS), 
                           num_classes_colon=2): # Default from config.COLON_NUM_CLASSES
    """
    Factory function to get the CNN model for colon cancer classification.
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes_colon (int): Number of colon cancer classes.
    Returns:
        tensorflow.keras.Model: Compiled Keras model for colon cancer.
    """
    logger.info(f"Building Colon Cancer CNN Model (version 1) with {num_classes_colon} classes.")
    # TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH order is H, W
    return build_custom_cnn_v1(input_shape, num_classes_colon, model_name="ColonCancerCNN_CustomV1")

def get_lung_cancer_model(input_shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, IMAGE_CHANNELS), 
                          num_classes_lung=3): # Default from config.LUNG_NUM_CLASSES
    """
    Factory function to get the CNN model for lung cancer classification.
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes_lung (int): Number of lung cancer classes.
    Returns:
        tensorflow.keras.Model: Compiled Keras model for lung cancer.
    """
    logger.info(f"Building Lung Cancer CNN Model (version 1) with {num_classes_lung} classes.")
    # This could use a different architecture if desired, e.g., build_custom_cnn_v2(...)
    # For now, using the same architecture but it's clearly separated.
    return build_custom_cnn_v1(input_shape, num_classes_lung, model_name="LungCancerCNN_CustomV1")


# --- Optional: Transfer Learning Example (VGG16) ---
def build_vgg16_transfer_model(input_shape, num_classes, fine_tune_at=15):
    """
    Builds a transfer learning model using VGG16 as a base.
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        fine_tune_at (int): Layer index from which to start fine-tuning VGG16.
                            VGG16 has 19 layers (fine_tune_at=0 means all trainable,
                            fine_tune_at=19 means only top custom layers trainable).
                            Commonly, last few convolutional blocks are fine-tuned.
    Returns:
        tensorflow.keras.Model: Compiled Keras transfer learning model.
    """
    base_model = VGG16(input_shape=input_shape,
                       include_top=False,  # Exclude VGG16's original classifier
                       weights='imagenet') # Load weights pre-trained on ImageNet

    # Freeze the base model layers initially
    base_model.trainable = True # Set all to trainable first
    for layer in base_model.layers[:fine_tune_at]: # Freeze layers before fine_tune_at index
        layer.trainable = False
    
    logger.info(f"VGG16 base model loaded. Layers up to index {fine_tune_at-1} frozen.")
    logger.info(f"Number of layers in VGG16 base: {len(base_model.layers)}")

    # Add custom classifier on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Alternative to Flatten, can help reduce parameters
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    if num_classes == 1:
        activation_func = 'sigmoid'
        loss_func = 'binary_crossentropy'
    elif num_classes == 2:
        activation_func = 'softmax'
        loss_func = 'categorical_crossentropy'
    else:
        activation_func = 'softmax'
        loss_func = 'categorical_crossentropy'
        
    output_layer = Dense(num_classes, activation=activation_func, name='custom_output')(x)

    model = Model(inputs=base_model.input, outputs=output_layer, name=f"VGG16_Transfer_{num_classes}cls")

    # Compile with a lower learning rate for fine-tuning
    optimizer = Adam(learning_rate=1e-5) # Very low LR for fine-tuning pre-trained weights
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
    
    logger.info(f"VGG16 transfer learning model created: '{model.name}'.")
    # model.summary(print_fn=logger.info)
    return model


if __name__ == '__main__':
    logger.info("Model Architecture module self-test initiated.")
    
    # Define standard input shape for testing (H, W, C)
    test_input_shape = (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, IMAGE_CHANNELS) # from config
    
    logger.info(f"\n--- Testing Colon Cancer Model ({build_custom_cnn_v1.__name__}) ---")
    num_colon_classes_test = 2
    colon_model = get_colon_cancer_model(input_shape=test_input_shape, num_classes_colon=num_colon_classes_test)
    assert colon_model is not None, "Colon model creation returned None."
    assert colon_model.output_shape == (None, num_colon_classes_test), \
        f"Colon model output shape mismatch: {colon_model.output_shape}"
    logger.info("Colon cancer model created successfully. Summary:")
    colon_model.summary(print_fn=logger.info) # Print summary for verification
    
    logger.info(f"\n--- Testing Lung Cancer Model ({build_custom_cnn_v1.__name__}) ---")
    num_lung_classes_test = 3
    lung_model = get_lung_cancer_model(input_shape=test_input_shape, num_classes_lung=num_lung_classes_test)
    assert lung_model is not None, "Lung model creation returned None."
    assert lung_model.output_shape == (None, num_lung_classes_test), \
        f"Lung model output shape mismatch: {lung_model.output_shape}"
    logger.info("Lung cancer model created successfully. Summary:")
    lung_model.summary(print_fn=logger.info)

    # --- Test VGG16 Transfer Learning Model (if uncommented) ---
    # logger.info(f"\n--- Testing VGG16 Transfer Learning Model ---")
    # num_transfer_classes_test = 3
    # # VGG16 typically expects input size of at least 32x32. Our 128x128 is fine.
    # vgg_transfer_model = build_vgg16_transfer_model(input_shape=test_input_shape, num_classes=num_transfer_classes_test, fine_tune_at=15) # Fine-tune from block5_conv1
    # if vgg_transfer_model:
    #     assert vgg_transfer_model.output_shape == (None, num_transfer_classes_test), \
    #         f"VGG16 transfer model output shape mismatch: {vgg_transfer_model.output_shape}"
    #     logger.info("VGG16 transfer model created successfully. Summary:")
    #     vgg_transfer_model.summary(print_fn=logger.info)
    # else:
    #     logger.warning("VGG16 transfer model test skipped (function might be commented out).")

    logger.info("\nModel Architecture module self-test completed.")
