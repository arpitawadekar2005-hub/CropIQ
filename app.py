import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------- CONFIG: EDIT THESE ----------
MODEL_PATH = "model.h5"   # your .h5 file

@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model


def main():
    st.title("CNN Image Classifier")

    st.write("Upload an image and get prediction from your .h5 model.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Show image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load model
        model = load_cnn_model()

        # Infer input size from model
        # Expected input shape: (None, H, W, C)
        input_shape = model.input_shape
        if len(input_shape) != 4:
            st.error(f"Unexpected model input shape: {input_shape}")
            return

        img_size = input_shape[1]  # assume square images (H == W)

        # ---- Preprocess (same as your code) ----
        img = image.load_img(uploaded_file, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        # ---------------------------------------

        # Predict
        pred = model.predict(img_array)
        pred_class_idx = np.argmax(pred)
        confidence = float(np.max(pred) * 100)

        # Defensive check in case classes list is wrong length
        if pred_class_idx >= len(classes):
            st.error(
                f"Prediction index {pred_class_idx} is out of range for classes list "
                f"(len={len(classes)}). Fix your 'classes' array."
            )
            return

        st.subheader("Prediction")
        st.write(f"**Class:** {classes[pred_class_idx]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Optional: show raw probabilities
        with st.expander("Show raw model output"):
            st.write(pred.tolist())


if __name__ == "__main__":
    main()
