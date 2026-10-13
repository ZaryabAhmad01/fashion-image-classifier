

from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👗",
    layout="centered"
)


# ==================================================
# 2. FASHION-MNIST CLASS NAMES
# ==================================================
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ==================================================
# 3. MODEL FILE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fashion_cnn.keras"


# ==================================================
# 4. LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    """
    Load the trained Fashion-MNIST CNN model once
    and cache it for subsequent Streamlit sessions.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please add fashion_cnn.keras to the models folder."
        )

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    # Validate the model input and output dimensions.
    if loaded_model.input_shape != (None, 28, 28, 1):
        raise ValueError(
            f"Unexpected model input shape: "
            f"{loaded_model.input_shape}. "
            "Expected (None, 28, 28, 1)."
        )

    if loaded_model.output_shape != (None, 10):
        raise ValueError(
            f"Unexpected model output shape: "
            f"{loaded_model.output_shape}. "
            "Expected (None, 10)."
        )

    return loaded_model


# ==================================================
# 5. APPLICATION HEADER
# ==================================================
st.title("👗 Fashion Image Classifier")

st.write(
    "Upload a clothing image, and our trained CNN model "
    "will predict its category."
)

st.info(
    "This model was trained on Fashion-MNIST images "
    "(28 × 28 grayscale pixels). It works best with "
    "simple, centered clothing images. Real-world photos "
    "may produce inaccurate predictions."
)


# ==================================================
# 6. LOAD THE MODEL WITH ERROR HANDLING
# ==================================================
try:
    model = load_model()

except Exception as exc:
    st.error("The trained model could not be loaded.")

    st.code(
        f"{type(exc).__name__}: {exc}",
        language="text"
    )

    st.warning(
        "Check that models/fashion_cnn.keras exists in your "
        "GitHub repository and that the deployed TensorFlow "
        "version is compatible with the saved model."
    )

    st.stop()


# Display model status.
st.caption("Model loaded successfully.")


# ==================================================
# 7. IMAGE UPLOAD
# ==================================================
uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# 8. IMAGE PREPROCESSING AND PREDICTION
# ==================================================
if uploaded_file is not None:

    try:
        # Open the uploaded image.
        with Image.open(uploaded_file) as original:
            image = ImageOps.exif_transpose(original)
            image = image.convert("RGB")

        # Display the original image.
        st.image(
            image,
            caption="Uploaded clothing image",
            use_container_width=True
        )

        # Convert RGB to grayscale.
        grayscale_image = image.convert("L")

        # Resize to the Fashion-MNIST dimensions.
        resized_image = grayscale_image.resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        # Convert to a NumPy array and normalize.
        img_array = np.asarray(
            resized_image,
            dtype=np.float32
        ) / 255.0

        # Shape: (1, 28, 28, 1)
        img_array = img_array.reshape(
            1, 28, 28, 1
        )

        # Run the prediction.
        with st.spinner("Analyzing your image..."):
            predictions = model.predict(
                img_array,
                verbose=0
            )

        predictions = np.asarray(predictions)

        # Validate prediction output.
        if predictions.shape != (1, 10):
            raise ValueError(
                f"Unexpected prediction shape: "
                f"{predictions.shape}. Expected (1, 10)."
            )

        scores = predictions[0]

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "The model returned invalid prediction values."
            )

        # Determine the highest-scoring class.
        predicted_index = int(np.argmax(scores))
        predicted_class = CLASS_NAMES[predicted_index]

        # Determine whether outputs are softmax probabilities.
        last_layer = model.layers[-1]
        activation = getattr(last_layer, "activation", None)
        activation_name = getattr(
            activation,
            "__name__",
            ""
        )

        if activation_name == "softmax":
            probabilities = scores

            # Softmax probabilities should be nonnegative
            # and sum to approximately 1.
            if (
                np.any(probabilities < 0)
                or not np.isclose(
                    np.sum(probabilities),
                    1.0,
                    atol=1e-3
                )
            ):
                raise ValueError(
                    "The model's softmax output is not a "
                    "valid probability distribution."
                )

        else:
            # Convert logits into normalized scores for display.
            # These are not calibrated confidence estimates.
            st.warning(
                "The model does not have a recognized softmax "
                "output. Displayed scores are normalized values, "
                "not calibrated confidence percentages."
            )

            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)

            probabilities = (
                exp_scores / np.sum(exp_scores)
            )

        # ==================================================
        # 9. DISPLAY PREDICTION RESULT
        # ==================================================
        st.success(
            f"Predicted category: {predicted_class}"
        )

        confidence = float(
            probabilities[predicted_index] * 100
        )

        st.metric(
            label="Top prediction score",
            value=f"{confidence:.2f}%"
        )

        # ==================================================
        # 10. DISPLAY ALL CLASS SCORES
        # ==================================================
        st.subheader("Scores for all categories")

        # Show classes from highest to lowest score.
        sorted_indices = np.argsort(probabilities)[::-1]

        for index in sorted_indices:
            class_name = CLASS_NAMES[index]
            probability = float(probabilities[index])

            st.write(
                f"**{class_name}:** {probability * 100:.2f}%"
            )

            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )

    except (UnidentifiedImageError, OSError):
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG, JPEG, or PNG file."
        )

    except Exception as exc:
        st.error("An error occurred during image prediction.")

        st.code(
            f"{type(exc).__name__}: {exc}",
            language="text"
        )


# ==================================================
# 11. FOOTER
# ==================================================
st.divider()

st.caption(
    "Fashion Image Classifier | Powered by TensorFlow "
    "and Streamlit"
)
