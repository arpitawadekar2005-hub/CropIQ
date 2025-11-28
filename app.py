import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gdown
# ---------- CONFIG ----------
MODEL_PATH = "plant_disease_model.h5"
CSV_PATH = "pesticide_data.csv"

DRIVE_FILE_ID = "170lylylDDiePU_pj1bfXppef6li1VMqz"  # <-- replace

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

classes = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___healthy',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___healthy'
]

# Load pesticide csv
pesticide_df = pd.read_csv(CSV_PATH)

# ------------------------------------------
# FUNCTIONS
# ------------------------------------------

def confidence_to_infection(confidence):
    return round(confidence * 100, 2)

def get_base_dose(plant, disease):
    row = pesticide_df[
        (pesticide_df['plant'].str.lower() == plant.lower()) &
        (pesticide_df['disease'].str.lower() == disease.lower())
    ]

    if row.empty:
        return None, None

    pesticide = row.iloc[0]['pesticide']
    base_ml_per_L = float(row.iloc[0]['base_ml_per_L'])
    return pesticide, base_ml_per_L

def compute_final_dose(base_ml_per_L, infection_percent, water_volume_ml=100):
    base_for_container = base_ml_per_L * (water_volume_ml / 1000.0)
    final_dose = base_for_container * (infection_percent / 100.0)
    return round(final_dose, 3)

def extract_plant_and_disease(label):
    parts = label.split("___")
    raw_plant = parts[0]
    raw_disease = parts[1] if len(parts) > 1 else "healthy"

    # Clean plant
    plant = (
        raw_plant.replace("_(including_sour)", "")
                 .replace("(including_sour)", "")
                 .replace("(maize)", "")
                 .replace(",", "")
                 .replace("_", " ")
                 .strip()
                 .lower()
    )

    # Clean disease
    disease = (
        raw_disease.replace("_", " ")
                   .replace("(Black Measles)", "")
                   .replace("(Black_Measles)", "")
                   .strip()
                   .lower()
    )

    return plant, disease



@st.cache_resource
def load_cnn_model():
    download_model()
    return load_model(MODEL_PATH)


# ------------------------------------------
# STREAMLIT APP
# ------------------------------------------
def main():
    st.title("🌿 Plant Disease Detector + Pesticide Calculator")
    st.write("Upload an image to predict disease and calculate pesticide dosage.")

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load model
        model = load_cnn_model()

        # Get input shape
        input_shape = model.input_shape
        img_size = input_shape[1]

        # Preprocess image
        img = image.load_img(uploaded_file, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)[0]
        pred_idx = np.argmax(pred)
        confidence = float(np.max(pred))

        # Identify disease
        predicted_label = classes[pred_idx]
        plant, disease = extract_plant_and_disease(predicted_label)

        # Calculate infection %
        infection_percent = confidence_to_infection(confidence)

        # Get pesticide info
        pesticide, base_ml_per_L = get_base_dose(plant, disease)

        st.subheader("🟩 Analysis Result")
        st.write(f"**Plant:** {plant}")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Infection %:** {infection_percent}%")

        if pesticide is None:
            st.error("⚠️ No pesticide information found for this plant & disease.")
        else:
            final_dose = compute_final_dose(base_ml_per_L, infection_percent)

            st.subheader("💧 Pesticide Calculation")
            st.write(f"**Recommended Pesticide:** {pesticide}")
            st.write(f"**Base Dose:** {base_ml_per_L} ml per 1 L water")
            st.write(f"**Final Dose for 100 ml spray:** `{final_dose} ml`")

if __name__ == "__main__":
    main()
