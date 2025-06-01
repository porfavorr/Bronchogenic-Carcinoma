# B.Tech Major Project: Bronchogenic Carcinoma Detection

This project aims to develop a system for classifying different types of cancer, specifically lung and colon cancer, from medical image scans using Deep Learning techniques. It provides functionalities for training custom Convolutional Neural Network (CNN) models, evaluating their performance, and making predictions on new images through a user-friendly web interface.

## 🌟 Features

* **Dual Cancer Specialization**: Independent models for lung and colon cancer detection.
* **Modular Architecture**: Code organized into distinct modules for clarity and scalability (configuration, utilities, data preprocessing, model architecture, training, evaluation, prediction, and UI).
* **Data Augmentation**: Enhances model generalization by augmenting the training dataset.
* **CNN Models**: Custom CNN architectures tailored for image classification tasks.
* **Transfer Learning (Optional Extension)**: `model_architecture.py` can be extended to include pre-trained models like VGG16 or ResNet50 (example provided but commented out).
* **Comprehensive Evaluation**: Includes metrics like accuracy, precision, recall, F1-score, confusion matrices, and training history plots.
* **Interactive Frontend**: A Streamlit web application allows users to upload images and receive instant predictions with confidence scores.
* **Command-Line Training**: Scripts to easily retrain models with specified parameters.

<!-- ## 📁 Project Structure -->

<!-- BTech_Cancer_Detection_Project/
│
├── data/                                # Dataset placeholder
│   ├── lung_colon_image_set/
│   │   ├── colon_image_sets/            # colon_aca/, colon_n/
│   │   └── lung_image_sets/             # lung_aca/, lung_n/, lung_scc/
│
├── trained_models/                      # Stores trained .keras models, history, and encoders
│
├── src/                                 # Source code
│   ├── config.py                        # Project configurations
│   ├── utils.py                         # Helper functions
│   ├── data_preprocessing.py            # Data loading and preprocessing
│   ├── model_architecture.py            # CNN model definitions
│   ├── training_pipeline.py             # Model training logic
│   ├── evaluation.py                    # Model evaluation and reporting
│   ├── prediction_pipeline.py           # Prediction logic
│   ├── app.py                           # Streamlit frontend
│   └── main_train.py                    # CLI for training
│
├── reports/                             # Stores evaluation outputs (plots, reports)
│   ├── colon/                           # Colon model specific reports
│   └── lung/                            # Lung model specific reports
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── run_app.sh / run_app.bat             # Scripts to launch the Streamlit app -->
## ⚙️ Setup and Installation

1.  **Clone the repository (or create the project structure and files manually):**
    ```bash
    git clone <your-repository-link>
    cd BTech_Cancer_Detection_Project
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    * Create the directory structure `data/lung_colon_image_set/colon_image_sets/` and `data/lung_colon_image_set/lung_image_sets/`.
    * Inside `colon_image_sets`, create subdirectories `colon_aca` and `colon_n`. Populate them with respective colon images (e.g., JPEGs, PNGs).
    * Inside `lung_image_sets`, create subdirectories `lung_aca`, `lung_n`, and `lung_scc`. Populate them with respective lung images.

5.  **Create necessary directories (if not already present):**
    ```bash
    mkdir trained_models
    mkdir -p reports/colon
    mkdir -p reports/lung
    mkdir -p logs/fit/colon # For TensorBoard logs
    mkdir -p logs/fit/lung  # For TensorBoard logs
    ```

## 🚀 How to Run

### 1. Training the Models

You can train the models using the `main_train.py` script.

* **To train the colon cancer model:**
    ```bash
    python src/main_train.py --model_type colon --epochs 50 --batch_size 32
    ```
    *(Adjust epochs and batch_size as needed. The default values from `config.py` will be used if not specified.)*

* **To train the lung cancer model:**
    ```bash
    python src/main_train.py --model_type lung --epochs 75 --batch_size 32
    ```

Trained models (`.keras`), label encoders (`.pkl`), and training history (`.pkl`) will be saved in the `trained_models/` directory. Evaluation plots and reports will be saved in the respective subdirectories within `reports/`. TensorBoard logs will be in `logs/fit/`.

### 2. Running the Prediction Web Application

Once the models are trained (or if you have pre-trained models in the `trained_models/` directory), you can start the Streamlit web application:

* **On Linux/macOS:**
    ```bash
    sh run_app.sh
    ```
* **On Windows:**
    ```bat
    run_app.bat
    ```
    Alternatively, you can run directly:
    ```bash
    streamlit run src/app.py
    ```

Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 🛠️ Modules Description

* **`src/config.py`**: Stores all global configurations like file paths, image dimensions, training hyperparameters (default epochs, batch size), and class names for both cancer types.
* **`src/utils.py`**: Contains helper functions for tasks like image loading, saving plots, creating loggers, saving model summaries, and handling pickle files for label encoders and history.
* **`src/data_preprocessing.py`**: Handles loading image paths and labels, image resizing and normalization, data augmentation (using Keras `ImageDataGenerator`), and creating data generators for training, validation, and testing.
* **`src/model_architecture.py`**: Defines the CNN architectures for both colon and lung cancer classification. Includes functions that return compiled Keras models.
* **`src/training_pipeline.py`**: Orchestrates the model training process. It uses functions from other modules to load data, build the model, train it with callbacks (like `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `TensorBoard`), and save the trained model and its history.
* **`src/evaluation.py`**: Contains functions to evaluate the trained models. This includes generating and saving classification reports, confusion matrices, and plotting training accuracy/loss curves.
* **`src/prediction_pipeline.py`**: Provides functions to load a trained model and its associated label encoder, preprocess an input image, and make a prediction, returning the predicted class and confidence score.
* **`src/app.py`**: The Streamlit web application. It provides a user interface for uploading an image, selecting the cancer type for prediction (lung or colon), and displays the classification result and model confidence. It also shows model status and provides disclaimers.
* **`src/main_train.py`**: A command-line interface script to trigger the training pipeline for either the lung or colon cancer model, allowing for customization of training parameters like epochs and batch size.

## 🎯 Potential Future Enhancements

* Integration of more advanced CNN architectures (e.g., ResNet, InceptionV3, EfficientNet) via transfer learning.
* Advanced visualization techniques like Grad-CAM to highlight regions of interest in the images.
* Support for more types of medical images or cancers.
* User authentication and a database for storing prediction history.
* Deployment to a cloud platform (e.g., Heroku, AWS, GCP).
* More sophisticated data augmentation techniques.
* Hyperparameter optimization using tools like KerasTuner or Optuna.
* Detailed logging of predictions made through the app.

---
Project developed as part of B.Tech curriculum.
name/team
