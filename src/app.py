"""
Streamlit Web Application for Cancer Detection from Medical Images.
This application provides a user interface to:
- Select the type of cancer to detect (Colon or Lung).
- Upload an image.
- View the model's prediction and confidence score.
- Display information about the models and disclaimers.
"""
import streamlit as st
from PIL import Image # For displaying images
import io # For handling byte streams from uploaded files
import os # For checking model file existence

# Import necessary components from the project
from prediction_pipeline import get_cached_predictor # Using the cached predictor
from config import (get_logger, 
                     COLON_CLASSES, LUNG_CLASSES, 
                     COLON_MODEL_PATH, LUNG_MODEL_PATH,
                     PROJECT_ROOT) # For displaying paths if needed

# Initialize logger for the Streamlit app
logger = get_logger(__name__)

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="AI Cancer Detection Aid",
    page_icon="🔬", # Microscope emoji
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded", # Keep sidebar open by default
    menu_items={
        'Get Help': 'mailto:yourprojectemail@example.com', # Replace with actual contact
        'Report a bug': "mailto:yourprojectemail@example.com",
        'About': """
        ## Bronchogenic Carcinoma (B.Tech Project)
        This application uses Convolutional Neural Networks (CNNs) to assist in identifying potential
        indicators of Colon and Lung cancer from medical images.

        **Developed by:** [Your Name/Team Name Here]
        **Institution:** [Your Institution Here]

        **Disclaimer:** This tool is for educational and research purposes ONLY.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        """
    }
)

# --- Helper Function to Display Model Status ---
def display_model_status_card(model_type_name, model_file_path, expected_classes_list):
    """
    Displays a card with information about the specified model's status.
    """
    st.subheader(f"{model_type_name.capitalize()} Cancer Detection Model")
    path_exists = os.path.exists(model_file_path)
    
    # Try to get the predictor to check full readiness (model + encoder)
    predictor_instance = get_cached_predictor(model_type_name)
    is_ready = predictor_instance and predictor_instance.check_readiness()

    if is_ready:
        st.success(f"✅ **Status:** Model and resources loaded successfully and ready for predictions.")
        st.caption(f"Model file: `{os.path.basename(model_file_path)}`")
    elif path_exists and not is_ready: # Model file exists but predictor not fully ready (e.g. encoder missing)
        st.warning(f"⚠️ **Status:** Model file found, but supporting resources (like label encoder) might be missing or failed to load. Predictions may not work.")
        st.caption(f"Model file: `{os.path.basename(model_file_path)}`")
    else: # Model file does not exist
        st.error(f"❌ **Status:** Model file NOT FOUND at `{model_file_path}`.")
        st.markdown(f"Please train the `{model_type_name}` model first using the command: "
                    f"`python src/main_train.py --model_type {model_type_name}`")

    st.markdown(f"**Target Classes:** `{', '.join(expected_classes_list)}`")
    st.markdown(f"**Expected Input:** Medical images (e.g., histology slides, CT scans - *user should specify based on their dataset*).")
    st.markdown("---")

# --- Sidebar Navigation and Information ---
st.sidebar.title("🔬 Bronchogenic Carcinoma")
st.sidebar.markdown("Navigate through the application using the options below.")

app_mode = st.sidebar.radio(
    "Select Mode:",
    ("🏠 Home", "🔍 Colon Cancer Prediction", "💨 Lung Cancer Prediction", "ℹ️ About & Disclaimer")
)

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  Select a **Prediction Mode** (Colon or Lung).
2.  Upload a clear image file (JPG, JPEG, PNG).
3.  Click the **"Predict"** button.
4.  Review the prediction result and confidence score.
""")

st.sidebar.markdown("---")
st.sidebar.warning("""
**Important Disclaimer:**
This tool is intended for educational and research purposes ONLY. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Predictions from this AI model should **NEVER** be used for self-diagnosis or to make medical decisions. Always consult a qualified healthcare professional for any health concerns.
""")


# --- Main Application Content based on Mode ---

if app_mode == "🏠 Home":
    st.title("Welcome to the AI-Powered Cancer Detection Aid")
    st.markdown("""
    This application leverages deep learning models to analyze medical images for potential signs of Colon and Lung cancer.
    It is developed as a B.Tech major project by Pulkit Hyanki and group to showcase the application of AI in healthcare.

    **Features:**
    -   Separate, specialized models for Colon and Lung cancer detection.
    -   User-friendly interface for image upload and prediction.
    -   Displays prediction results with associated confidence levels.

    **How to Get Started:**
    -   Use the sidebar navigation to select either **Colon Cancer Prediction** or **Lung Cancer Prediction**.
    -   Follow the instructions to upload an image and receive an analysis.
    -   Visit the **About & Disclaimer** section for more information on the project and important limitations.

    ---
    *Please remember the critical disclaimer regarding the use of this tool provided in the sidebar and the 'About' section.*
    """)
    
    st.header("Model Status Overview")
    col1, col2 = st.columns(2)
    with col1:
        display_model_status_card("colon", COLON_MODEL_PATH, COLON_CLASSES)
    with col2:
        display_model_status_card("lung", LUNG_MODEL_PATH, LUNG_CLASSES)


elif app_mode == "🔍 Colon Cancer Prediction":
    st.header("Colon Cancer Prediction Module")
    display_model_status_card("colon", COLON_MODEL_PATH, COLON_CLASSES)

    colon_predictor = get_cached_predictor('colon') # Get (or initialize) the predictor
    
    if not colon_predictor or not colon_predictor.check_readiness():
        st.error("🔴 Colon cancer prediction model is not available or not fully ready. "
                 "Please ensure the model is trained and all necessary files (model, encoder) are correctly placed and configured.")
    else:
        st.info("🟢 Colon cancer prediction model is loaded and ready.")
        
        uploaded_colon_image = st.file_uploader(
            "Upload a Colon Image (JPG, JPEG, PNG format)", 
            type=["jpg", "jpeg", "png"], 
            key="colon_image_uploader",
            help="Upload an image suspected to be of colon tissue."
        )

        if uploaded_colon_image is not None:
            try:
                image_bytes_data = uploaded_colon_image.getvalue() # Read image as bytes
                pil_image = Image.open(io.BytesIO(image_bytes_data)) # Open with PIL for display
                
                st.image(pil_image, caption="Uploaded Colon Image Preview", use_column_width=True)
                st.markdown("---")

                if st.button("🔬 Predict Colon Cancer Type", key="predict_colon_button", type="primary"):
                    with st.spinner("🧠 Analyzing colon image... Please wait."):
                        # Pass image bytes directly to the predictor
                        predicted_class_name, confidence_score = colon_predictor.predict_single_image(io.BytesIO(image_bytes_data))
                    
                    st.subheader("Prediction Result:")
                    if predicted_class_name and confidence_score is not None:
                        # Custom styling for results
                        is_normal_prediction = "n" in predicted_class_name.lower() # Check if 'n' (normal) is in class name
                        result_color = "green" if is_normal_prediction else "red"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; border: 2px solid {result_color}; background-color: #f0f8ff;">
                            <h3 style="color: {result_color};">Predicted Class: {predicted_class_name.replace('_', ' ').title()}</h3>
                            <h4>Confidence Score: {confidence_score*100:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True) # Add some space

                        # Provide brief interpretation (example)
                        if "aca" in predicted_class_name.lower():
                            st.warning("Interpretation: The model predicts Adenocarcinoma, a type of cancerous growth. "
                                       "This requires urgent review by a medical specialist.")
                        elif is_normal_prediction:
                            st.success("Interpretation: The model predicts Normal colon tissue. "
                                       "Continue with regular health check-ups as advised by your doctor.")
                        else: # For any other class if defined
                            st.info(f"Interpretation: The model predicted '{predicted_class_name}'. "
                                    "Please consult a healthcare professional for detailed analysis.")
                    else:
                        st.error("⚠️ Prediction failed. The image might be unsuitable, or an internal model error occurred. Please check application logs or try a different image.")
            except Exception as e:
                logger.error(f"Error processing uploaded colon image or during prediction: {e}", exc_info=True)
                st.error(f"An unexpected error occurred while processing the image: {e}")

elif app_mode == "💨 Lung Cancer Prediction":
    st.header("Lung Cancer Prediction Module")
    display_model_status_card("lung", LUNG_MODEL_PATH, LUNG_CLASSES)

    lung_predictor = get_cached_predictor('lung')

    if not lung_predictor or not lung_predictor.check_readiness():
        st.error("🔴 Lung cancer prediction model is not available or not fully ready. "
                 "Ensure the model is trained and all files are correctly placed.")
    else:
        st.info("🟢 Lung cancer prediction model is loaded and ready.")

        uploaded_lung_image = st.file_uploader(
            "Upload a Lung Image (JPG, JPEG, PNG format)", 
            type=["jpg", "jpeg", "png"], 
            key="lung_image_uploader",
            help="Upload an image suspected to be of lung tissue."
        )

        if uploaded_lung_image is not None:
            try:
                image_bytes_data = uploaded_lung_image.getvalue()
                pil_image = Image.open(io.BytesIO(image_bytes_data))

                st.image(pil_image, caption="Uploaded Lung Image Preview", use_column_width=True)
                st.markdown("---")

                if st.button("🔬 Predict Lung Cancer Type", key="predict_lung_button", type="primary"):
                    with st.spinner("🧠 Analyzing lung image... Please wait."):
                        predicted_class_name, confidence_score = lung_predictor.predict_single_image(io.BytesIO(image_bytes_data))
                    
                    st.subheader("Prediction Result:")
                    if predicted_class_name and confidence_score is not None:
                        is_normal_prediction = "n" in predicted_class_name.lower()
                        result_color = "green" if is_normal_prediction else "red"

                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; border: 2px solid {result_color}; background-color: #f0fff0;">
                            <h3 style="color: {result_color};">Predicted Class: {predicted_class_name.replace('_', ' ').title()}</h3>
                            <h4>Confidence Score: {confidence_score*100:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                        if "aca" in predicted_class_name.lower():
                            st.warning("Interpretation: The model predicts Lung Adenocarcinoma, a common type of lung cancer. "
                                       "Immediate consultation with a specialist is advised.")
                        elif "scc" in predicted_class_name.lower():
                             st.warning("Interpretation: The model predicts Lung Squamous Cell Carcinoma, another form of lung cancer. "
                                        "Seek specialist medical advice promptly.")
                        elif is_normal_prediction:
                            st.success("Interpretation: The model predicts Normal lung tissue. "
                                       "Maintain regular health screenings as recommended by your physician.")
                        else:
                            st.info(f"Interpretation: The model predicted '{predicted_class_name}'. "
                                    "A healthcare professional should be consulted for an accurate diagnosis.")
                    else:
                        st.error("⚠️ Prediction failed. The image might be unsuitable, or an internal model error occurred. Check logs or try another image.")
            except Exception as e:
                logger.error(f"Error processing uploaded lung image or during prediction: {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")

elif app_mode == "ℹ️ About & Disclaimer":
    st.title("About   Bronchogenic Carcinoma")
    st.markdown(f"""
    This application is a **B.Tech Major Project** aimed at demonstrating the potential of Deep Learning,
    specifically Convolutional Neural Networks (CNNs), in the domain of medical image analysis for cancer detection.
    It provides a platform for classifying images related to Colon and Lung cancers.

    **Core Technologies Employed:**
    -   **Programming Language:** Python
    -   **Deep Learning Framework:** TensorFlow with Keras API
    -   **Data Handling & Scientific Computing:** NumPy, Pandas
    -   **Image Processing:** Pillow (PIL), OpenCV (for backend processing if extended)
    -   **Model Evaluation:** Scikit-learn
    -   **Web Application Framework:** Streamlit

    **Project Objectives:**
    -   To develop and implement CNN models capable of distinguishing between different tissue types
        (e.g., normal, adenocarcinoma, squamous cell carcinoma) for colon and lung images.
    -   To engineer a modular and maintainable codebase that clearly separates functionalities such as
        data preprocessing, model definition, training procedures, evaluation, and prediction services.
    -   To create an intuitive and interactive web interface that allows users (e.g., students, researchers)
        to easily interact with the trained models by uploading images and receiving predictions.
    -   To serve as a comprehensive learning tool and a demonstrable piece of work for academic evaluation.

    **Dataset Information (Assumed Structure & Type):**
    The models integrated into this system are designed to be trained on datasets of medical images.
    Typically, these would be histology slides or relevant scans, categorized as follows:
    -   **Colon Cancer Dataset:**
        -   `colon_aca`: Images representing Adenocarcinoma of the colon.
        -   `colon_n`: Images representing Normal/Healthy colon tissue.
    -   **Lung Cancer Dataset:**
        -   `lung_aca`: Images representing Adenocarcinoma of the lung.
        -   `lung_n`: Images representing Normal/Healthy lung tissue.
        -   `lung_scc`: Images representing Squamous Cell Carcinoma of the lung.
    *The specific dataset used for training the deployed models should be detailed by the project author(s),
    including sources, image resolutions, and any specific preprocessing steps undertaken beyond the general pipeline.*

    **Model Architectures:**
    The system utilizes custom-designed CNN architectures. There is also a provision in the codebase
    (commented out example) for extending this with transfer learning techniques, employing pre-trained models
    like VGG16, ResNet, or EfficientNet, which could be a future enhancement.

    ---
    """)
    st.error("""
    **CRITICAL ETHICAL CONSIDERATIONS & DISCLAIMER OF LIABILITY:**

    ⚠️ **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.** ⚠️

      Bronchogenic Carcinoma is strictly an academic project and a proof-of-concept.
    It is **NOT a certified medical device** and **MUST NOT** be used for:
    -   Actual medical diagnosis.
    -   Making any treatment decisions.
    -   Self-diagnosis or self-treatment.
    -   Replacing the consultation, advice, or expertise of qualified healthcare professionals.

    The predictions generated by the AI models within this application are based on patterns learned from
    a specific dataset and may not be universally accurate or applicable to all individual cases or image types.
    AI models can have limitations, biases, and may produce incorrect or misleading results.

    **Reliance on any information or prediction provided by this application is SOLELY AT YOUR OWN RISK.**
    The developers, authors, and associated institutions disclaim any and all liability for any decisions,
    actions, or consequences resulting from the use of this application.

    **ALWAYS consult with a qualified doctor or other healthcare provider for any questions you may have
    regarding a medical condition or treatment options.** Never disregard professional medical advice or delay
    in seeking it because of something you have seen or read from this application.
    """)
    st.markdown("---")
    st.markdown(f"*Project Root Path (for reference): `{PROJECT_ROOT}`*")
    st.markdown("*Developed as part of a B.Tech curriculum.*")
    st.markdown("*[Consider adding Your Name/Team, Guide Name, and Institution Here]*")

# --- Footer Section ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 10px; color: #777;">
        <p>Bronchogenic Carcinoma Detection | Pulkit Hyanki &copy; 2024-2025</p>
        <p><small>Remember: This is an educational tool, not for medical diagnosis.</small></p>
    </div>
    """, unsafe_allow_html=True
)

logger.debug("Streamlit application page rerendered/initialized.")
