
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👗",
    layout="centered"
)


# --------------------------------------------------
# 2. Class names (Fashion-MNIST order)
# --------------------------------------------------
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
    "Ankle boot",
]


# --------------------------------------------------
# 3. Load the trained CNN model
# --------------------------------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "fashion_cnn.h5"
)


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it across sessions."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Check that models/fashion_cnn.h5 is committed "
            "to your GitHub repository."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------
st.title("👗 Fashion Image Classifier")

st.write(
    "Upload a clothing image, and the trained CNN model "
    "will predict its Fashion-MNIST category."
)

st.warning(
    "This model was trained on 28 × 28 grayscale "
    "Fashion-MNIST images. Real-world clothing photos "
    "may produce inaccurate predictions because their "
    "backgrounds, shapes, and lighting differ from the dataset."
)


# --------------------------------------------------
# 5. Load model with a helpful error message
# --------------------------------------------------
try:
    model = load_model()

except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)

    st.info(
        "Check your model file, TensorFlow/Keras versions, "
        "and any custom layers used during training."
    )
    st.stop()


# --------------------------------------------------
# 6. Upload an image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 7. Preprocess and predict
# --------------------------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        # Convert to grayscale, matching Fashion-MNIST.
        image = image.convert("L")

        # Resize to the model's expected image dimensions.
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert pixels to float32 and normalize to [0, 1].
        img_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Shape: (batch_size, height, width, channels).
        img_array = img_array.reshape(1, 28, 28, 1)

        # Verify the model's input shape when available.
        expected_shape = model.input_shape

        if (
            isinstance(expected_shape, tuple)
            and len(expected_shape) == 4
            and expected_shape[1:] != (28, 28, 1)
        ):
            st.error(
                f"Input shape mismatch. Model expects "
                f"{expected_shape}, but preprocessing produces "
                f"{img_array.shape}."
            )
            st.stop()

        with st.spinner("Analyzing image..."):
            predictions = model.predict(
                img_array,
                verbose=0
            )

        predictions = np.asarray(predictions)

        if (
            predictions.ndim != 2
            or predictions.shape[0] != 1
            or predictions.shape[1] != len(CLASS_NAMES)
        ):
            st.error(
                "Unexpected model output shape: "
                f"{predictions.shape}. Expected (1, 10)."
            )
            st.stop()

        scores = predictions[0]

        # This app assumes the model outputs 10 class
        # probabilities, usually from a softmax output layer.
        if not np.all(np.isfinite(scores)):
            st.error("The model returned invalid prediction values.")
            st.stop()

        predicted_index = int(np.argmax(scores))
        predicted_class = CLASS_NAMES[predicted_index]

        st.success(f"Predicted category: **{predicted_class}**")

        # Only interpret scores as probabilities if the model
        # has a softmax output (or otherwise outputs probabilities).
        last_layer = model.layers[-1]
        activation = getattr(last_layer, "activation", None)
        activation_name = getattr(activation, "__name__", "")

        if activation_name == "softmax":
            probabilities = scores
        else:
            st.warning(
                "The model's last layer is not recognized as "
                "softmax. The values below may be logits, not "
                "probabilities; confidence percentages may be "
                "misleading."
            )

            # Convert logits to normalized scores for display only.
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / np.sum(exp_scores)

        confidence = float(probabilities[predicted_index]) * 100

        st.metric(
            label="Top prediction score",
            value=f"{confidence:.2f}%"
        )

        # Show every class score.
        st.subheader("Scores for all categories")

        for name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):
            probability = float(probability)

            st.write(f"{name}: {probability * 100:.2f}%")
            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )

    except (UnidentifiedImageError, OSError):
        st.error(
            "Could not read this image. Please upload a valid "
            "JPG, JPEG, or PNG file."
        )

    except Exception as exc:
        st.error("An error occurred while processing the image.")
        st.exception(exc)
