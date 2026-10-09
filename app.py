import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "resnet50_plant_disease.keras"

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "plant": "Pepper",
        "type": "Bacterial disease",
        "symptoms": "Small dark spots or lesions on leaves and fruits.",
        "management": "Remove severely affected leaves and maintain good airflow."
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper Leaf",
        "plant": "Pepper",
        "type": "Healthy",
        "symptoms": "No major visible disease symptoms.",
        "management": "Continue good watering, nutrition and crop monitoring."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "plant": "Potato",
        "type": "Fungal disease",
        "symptoms": "Dark circular lesions, often with concentric rings.",
        "management": "Remove infected foliage and maintain proper field sanitation."
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "plant": "Potato",
        "type": "Fungal-like disease",
        "symptoms": "Dark irregular lesions that may expand rapidly.",
        "management": "Remove infected material and avoid prolonged leaf wetness."
    },

    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "plant": "Potato",
        "type": "Healthy",
        "symptoms": "No major visible disease symptoms.",
        "management": "Continue regular crop monitoring."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "type": "Bacterial disease",
        "symptoms": "Small dark spots on leaves, stems or fruits.",
        "management": "Remove affected plant material and reduce unnecessary leaf wetness."
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "type": "Fungal disease",
        "symptoms": "Brown lesions with concentric ring patterns.",
        "management": "Remove infected leaves and improve plant airflow."
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "type": "Fungal-like disease",
        "symptoms": "Large dark irregular lesions on leaves.",
        "management": "Remove infected material and avoid excessive leaf moisture."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal disease",
        "symptoms": "Yellowish areas on upper leaf surfaces with mold growth underneath.",
        "management": "Improve ventilation and reduce humidity around foliage."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "symptoms": "Small circular spots with darker margins.",
        "management": "Remove infected leaves and maintain field sanitation."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "plant": "Tomato",
        "type": "Pest",
        "symptoms": "Fine speckling, yellowing and possible webbing on leaves.",
        "management": "Monitor leaf undersides and manage affected plants appropriately."
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "plant": "Tomato",
        "type": "Fungal disease",
        "symptoms": "Circular lesions that can develop target-like patterns.",
        "management": "Remove affected foliage and improve airflow."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "symptoms": "Leaf curling, yellowing and reduced plant growth.",
        "management": "Control insect vectors and remove severely affected plants."
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "plant": "Tomato",
        "type": "Viral disease",
        "symptoms": "Mosaic-like light and dark green patterns on leaves.",
        "management": "Remove infected plants and maintain good hygiene."
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "plant": "Tomato",
        "type": "Healthy",
        "symptoms": "No major visible disease symptoms.",
        "management": "Continue regular plant monitoring and healthy cultivation practices."
    }
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    model_error = str(e)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")
    resized = image.resize((224, 224))

    img_array = np.array(resized).astype("float32")

    # ResNet50 preprocessing
    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:
        results.append({
            "class": CLASS_NAMES[index],
            "name": DISEASE_INFO[CLASS_NAMES[index]]["name"],
            "confidence": float(predictions[index] * 100)
        })

    return results


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌿 Plant Disease AI")

st.sidebar.markdown(
    """
### Navigation

🏠 Home

🔬 Disease Prediction

📊 Model Comparison

📚 Disease Information

🕘 Prediction History

ℹ️ About
"""
)

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Disease Prediction",
        "Model Comparison",
        "Disease Information",
        "Prediction History",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-based educational plant disease detection system."
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title("🌿 Plant Disease Detection Using Deep Learning")

    st.subheader(
        "Explainable Plant Disease Detection Using ResNet50"
    )

    st.markdown(
        """
    Welcome to the Plant Disease Detection system.

    This application uses a **ResNet50 transfer-learning model**
    trained on the **PlantVillage dataset**.

    ### ✨ Main Features

    - 🌱 Plant disease prediction
    - 🎯 Confidence score
    - 🥇 Top-3 predictions
    - 📊 Model comparison
    - 📚 Disease information
    - 🕘 Prediction history
    - 🖼️ Image upload
    - ⚡ Fast prediction
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Plant Classes", "15")

    with col2:
        st.metric("Dataset Images", "20,638")

    with col3:
        st.metric("Primary Model", "ResNet50")

    st.success("System ready for prediction.")

# ============================================================
# DISEASE PREDICTION
# ============================================================

elif page == "Disease Prediction":

    st.title("🔬 Plant Disease Prediction")

    st.write(
        "Upload a clear plant leaf image to predict its disease."
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )

        with col2:

            if model is None:

                st.error("Model could not be loaded.")

                st.code(model_error)

            else:

                with st.spinner("Analyzing leaf..."):

                    results = predict_image(image)

                best = results[0]

                st.subheader("🎯 Prediction")

                st.success(best["name"])

                confidence = best["confidence"]

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                if confidence < 60:
                    st.warning(
                        "⚠️ Low confidence. Try uploading a clearer "
                        "leaf image with good lighting."
                    )

                elif confidence < 80:
                    st.info(
                        "ℹ️ Moderate confidence."
                    )

                else:
                    st.success(
                        "✅ High confidence prediction."
                    )

                # ------------------------------------------------
                # TOP 3
                # ------------------------------------------------

                st.subheader("🏆 Top-3 Predictions")

                top_data = []

                for item in results:

                    top_data.append({
                        "Prediction": item["name"],
                        "Confidence": f"{item['confidence']:.2f}%"
                    })

                st.table(
                    pd.DataFrame(top_data)
                )

                # ------------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------------

                st.subheader("📊 Prediction Probabilities")

                chart_data = pd.DataFrame({
                    "Disease": [
                        item["name"]
                        for item in results
                    ],
                    "Confidence": [
                        item["confidence"]
                        for item in results
                    ]
                })

                chart_data = chart_data.set_index("Disease")

                st.bar_chart(chart_data)

                # ------------------------------------------------
                # DISEASE INFORMATION
                # ------------------------------------------------

                info = DISEASE_INFO[best["class"]]

                st.subheader("📚 Disease Information")

                st.write(
                    f"**Plant:** {info['plant']}"
                )

                st.write(
                    f"**Type:** {info['type']}"
                )

                st.write(
                    f"**Symptoms:** {info['symptoms']}"
                )

                st.write(
                    f"**General Management:** {info['management']}"
                )

                # ------------------------------------------------
                # SAVE HISTORY
                # ------------------------------------------------

                st.session_state.history.append({
                    "Prediction": best["name"],
                    "Confidence": f"{confidence:.2f}%"
                })

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    st.write(
        "Comparative performance of the three deep-learning models "
        "used in the project."
    )

    comparison = pd.DataFrame({
        "Model": [
            "Basic CNN",
            "ResNet50",
            "MobileNetV2"
        ],
        "Accuracy": [
            90.42,
            96.24,
            92.33
        ],
        "Precision": [
            90.69,
            96.31,
            92.42
        ],
        "Recall": [
            90.42,
            96.24,
            92.33
        ],
        "F1 Score": [
            90.23,
            96.17,
            92.27
        ],
        "Parameters": [
            11170895,
            23851919,
            2423887
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.subheader("🏆 Best Overall Model")

    st.success(
        "ResNet50 achieved the strongest overall validation performance "
        "among the three compared models."
    )

    st.subheader("📈 Accuracy Comparison")

    accuracy_chart = comparison[
        ["Model", "Accuracy"]
    ].set_index("Model")

    st.bar_chart(accuracy_chart)

# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "Disease Information":

    st.title("📚 Disease Information")

    selected = st.selectbox(
        "Select a disease",
        CLASS_NAMES
    )

    info = DISEASE_INFO[selected]

    st.header(info["name"])

    st.write(
        f"**Plant:** {info['plant']}"
    )

    st.write(
        f"**Type:** {info['type']}"
    )

    st.write(
        f"**Symptoms:** {info['symptoms']}"
    )

    st.write(
        f"**General Management:** {info['management']}"
    )

# ============================================================
# HISTORY
# ============================================================

elif page == "Prediction History":

    st.title("🕘 Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions have been made during this session."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.success("History cleared.")

            st.rerun()

# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
### Plant Disease Detection Using CNN and Transfer Learning

This project compares three deep-learning approaches:

- Basic CNN
- ResNet50
- MobileNetV2

The system uses the **PlantVillage dataset** containing
15 plant disease/healthy classes.

### Project Workflow

**Dataset**

↓

**Image Preprocessing**

↓

**CNN / ResNet50 / MobileNetV2**

↓

**Performance Evaluation**

↓

**Best Model Selection**

↓

**Plant Disease Prediction**

### Technologies Used

- Python
- TensorFlow
- Keras
- ResNet50
- CNN
- MobileNetV2
- NumPy
- Pandas
- Streamlit

### Important Note

This application is intended for educational and demonstration
purposes. Predictions should not be treated as professional
agricultural diagnosis.
"""
    )

    st.success(
        "🌿 Plant Disease Detection System"
    )
