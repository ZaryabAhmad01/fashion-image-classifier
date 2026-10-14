
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fashion AI | Image Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9ff 0%, #f5f7fc 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #111b2e;
    }

    [data-testid="stSidebar"] * {
        color: #edf2ff;
    }

    .brand {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-bottom: 0.2rem;
    }

    .brand-accent {
        color: #8b91ff;
    }

    .muted {
        color: #71809a;
        font-size: 0.94rem;
        line-height: 1.7;
    }

    .hero {
        background: linear-gradient(120deg, #ffffff, #eef2ff);
        border: 1px solid #e4e9f7;
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 2.35rem;
        line-height: 1.2;
        color: #17243d;
        letter-spacing: -1.2px;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #687995;
        font-size: 1.02rem;
        margin: 0;
        line-height: 1.65;
    }

    .eyebrow {
        color: #635bdb;
        text-transform: uppercase;
        font-weight: 800;
        font-size: 0.76rem;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }

    .section-heading {
        color: #1d2b45;
        font-size: 1.18rem;
        font-weight: 750;
        margin: 0 0 6px 0;
    }

    .section-caption {
        color: #8190a8;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e6ebf5;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .result-panel {
        background: linear-gradient(135deg, #ffffff, #f0fbf6);
        border: 1px solid #ccefe0;
        border-radius: 20px;
        padding: 24px;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .result-label {
        color: #15966b;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .result-name {
        color: #137b59;
        font-size: 2rem;
        font-weight: 850;
        letter-spacing: -0.8px;
        margin: 6px 0 12px 0;
        overflow-wrap: anywhere;
    }

    .pill {
        display: inline-block;
        background: #e8edff;
        color: #5147c8;
        border: 1px solid #d9e0ff;
        border-radius: 20px;
        padding: 7px 12px;
        margin: 3px 5px 3px 0;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .side-card {
        background: #ffffff;
        border: 1px solid #e6ebf5;
        border-radius: 16px;
        padding: 17px;
        margin-bottom: 15px;
    }

    .side-title {
        color: #24334c;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 9px 0;
        border-bottom: 1px solid #edf0f6;
        color: #73819a;
        font-size: 0.84rem;
    }

    .info-value {
        color: #263650;
        font-weight: 650;
        text-align: right;
        overflow-wrap: anywhere;
    }

    .success-text {
        color: #16845f;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 12px;
    }

    .footer {
        text-align: center;
        color: #8996ab;
        font-size: 0.82rem;
        padding-top: 20px;
        padding-bottom: 8px;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b8c5f2;
        border-radius: 15px;
        padding: 12px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6ebf5;
        border-radius: 14px;
        padding: 14px;
    }

    div[data-testid="stProgressBar"] > div {
        border-radius: 20px;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 20px;
        }

        .hero h1 {
            font-size: 1.8rem;
        }

        .result-name {
            font-size: 1.6rem;
        }

        .block-container {
            padding-top: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. CLASS LABELS AND MODEL PATH
# ============================================================
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

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fashion_cnn.keras"


# ============================================================
# 4. LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Add fashion_cnn.keras to your repository's models folder."
        )

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    if loaded_model.input_shape != (None, 28, 28, 1):
        raise ValueError(
            f"Unexpected input shape: {loaded_model.input_shape}. "
            "Expected (None, 28, 28, 1)."
        )

    if loaded_model.output_shape != (None, 10):
        raise ValueError(
            f"Unexpected output shape: {loaded_model.output_shape}. "
            "Expected (None, 10)."
        )

    return loaded_model


# ============================================================
# 5. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 22px 0;">
            <div style="font-size: 2.5rem;">👗</div>
            <div style="font-size: 1.3rem; font-weight: 800;">
                Fashion <span style="color:#9da3ff;">AI</span>
            </div>
            <div style="color:#aab6cc; font-size:0.85rem;">
                Image Classification Studio
            </div>
        </div>
        <hr style="border-color:#354158;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    st.markdown("🏠  **Classifier**")
    st.markdown("🧠  Model overview")
    st.markdown("📊  Prediction analytics")

    st.markdown("---")
    st.markdown("#### Built with")
    st.markdown("🟠 TensorFlow / Keras")
    st.markdown("🔴 Streamlit")
    st.markdown("🔵 NumPy and Pillow")

    st.markdown("---")
    st.caption("Fashion AI · Portfolio Project")
    st.caption("Image classification with a CNN")


# ============================================================
# 6. HERO SECTION
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Computer Vision · Deep Learning</div>
        <h1>Fashion Image Classifier</h1>
        <p>
            Turn a clothing image into an AI prediction.
            Upload an image to identify its fashion category
            and explore the model's scores across ten classes.
        </p>
        <div style="margin-top:16px;">
            <span class="pill">✦ CNN-powered</span>
            <span class="pill">10 Fashion Categories</span>
            <span class="pill">Image Recognition</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. LOAD MODEL
# ============================================================
try:
    model = load_model()
except Exception as exc:
    st.error("The classifier could not initialize.")
    st.code(f"{type(exc).__name__}: {exc}", language="text")
    st.info(
        "Verify that models/fashion_cnn.keras exists and that "
        "the deployed TensorFlow/Keras versions are compatible."
    )
    st.stop()


# ============================================================
# 8. MAIN LAYOUT
# ============================================================
main_col, side_col = st.columns(
    [2.15, 1],
    gap="large",
)

with main_col:
    st.markdown(
        '<div class="section-heading">Image workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">'
        'Upload a clothing image to run classification.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:34px 20px;">
                <div style="font-size:3.2rem;">🖼️</div>
                <div style="font-size:1.15rem; font-weight:750; color:#293953;">
                    Ready to classify
                </div>
                <p style="color:#8491a8; line-height:1.7;">
                    Select an image above to begin.<br>
                    Your prediction and category scores will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### How to use")
        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("**01 · Upload**")
            st.caption("Choose a clothing image.")

        with step2:
            st.markdown("**02 · Analyze**")
            st.caption("The CNN evaluates the image.")

        with step3:
            st.markdown("**03 · Explore**")
            st.caption("Review the predicted class and scores.")

    else:
        try:
            with Image.open(uploaded_file) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")

            # Preprocessing must match the training pipeline.
            grayscale_image = image.convert("L")
            resized_image = grayscale_image.resize(
                (28, 28),
                Image.Resampling.LANCZOS,
            )

            img_array = np.asarray(
                resized_image,
                dtype=np.float32,
            ) / 255.0

            img_array = img_array.reshape(1, 28, 28, 1)

            with st.spinner("Analyzing image..."):
                predictions = np.asarray(
                    model.predict(img_array, verbose=0)
                )

            if predictions.shape != (1, 10):
                raise ValueError(
                    f"Unexpected output shape: {predictions.shape}"
                )

            scores = predictions[0]

            if not np.all(np.isfinite(scores)):
                raise ValueError("The model returned invalid scores.")

            last_layer = model.layers[-1]
            activation = getattr(last_layer, "activation", None)
            activation_name = getattr(activation, "__name__", "")

            if activation_name == "softmax":
                probabilities = scores.astype(np.float64)

                if (
                    np.any(probabilities < 0)
                    or not np.isclose(
                        probabilities.sum(),
                        1.0,
                        atol=1e-3,
                    )
                ):
                    raise ValueError(
                        "The model's softmax output is not a valid "
                        "probability distribution."
                    )
                is_softmax = True
            else:
                # Normalize logits for ranking and visualization.
                # These are not calibrated confidence estimates.
                shifted = scores.astype(np.float64) - np.max(scores)
                exp_scores = np.exp(shifted)
                probabilities = exp_scores / exp_scores.sum()
                is_softmax = False

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            top_score = float(probabilities[predicted_index] * 100)

            st.markdown(
                '<div class="section-heading">Classification result</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-caption">'
                'Prediction generated by your trained CNN model.'
                '</div>',
                unsafe_allow_html=True,
            )

            preview_col, result_col = st.columns(
                [0.9, 1.3],
                gap="medium",
            )

            with preview_col:
                st.image(
                    image,
                    caption="Uploaded image",
                    use_container_width=True,
                )

            with result_col:
                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="result-label">✓ Prediction complete</div>
                        <div style="color:#64748b; margin-top:15px;">
                            Predicted category
                        </div>
                        <div class="result-name">{predicted_class}</div>
                        <div style="color:#64748b; font-size:0.9rem;">
                            {"Top-class probability" if is_softmax else "Normalized top-class score"}
                        </div>
                        <div style="font-size:2rem; font-weight:850; color:#203650;">
                            {top_score:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if not is_softmax:
                st.caption(
                    "The model's final layer is not softmax. "
                    "The displayed percentages are normalized scores, "
                    "not calibrated probabilities."
                )

            st.markdown("#### Class score distribution")
            st.caption(
                "Scores are sorted from highest to lowest."
            )

            sorted_indices = np.argsort(probabilities)[::-1]

            for index in sorted_indices:
                name = CLASS_NAMES[index]
                probability = float(probabilities[index])

                score_col, value_col = st.columns([4, 1])

                with score_col:
                    st.write(name)

                with value_col:
                    st.write(f"**{probability * 100:.2f}%**")

                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

            st.download_button(
                label="Download prediction report",
                data=(
                    "Fashion Image Classifier - Prediction Report\n"
                    f"Predicted category: {predicted_class}\n"
                    f"Top-class score: {top_score:.2f}%\n\n"
                    "Category scores:\n"
                    + "\n".join(
                        f"{CLASS_NAMES[i]}: "
                        f"{probabilities[i] * 100:.2f}%"
                        for i in sorted_indices
                    )
                    + "\n"
                ),
                file_name="fashion_prediction_report.txt",
                mime="text/plain",
            )

        except (UnidentifiedImageError, OSError):
            st.error(
                "This file could not be read as an image. "
                "Please upload a valid JPG, JPEG, or PNG."
            )

        except Exception as exc:
            st.error("Unable to complete the prediction.")
            st.code(
                f"{type(exc).__name__}: {exc}",
                language="text",
            )


# ============================================================
# 9. MODEL INFORMATION SIDEBAR
# ============================================================
with side_col:
    st.markdown(
        """
        <div class="side-card">
            <div class="side-title">🧠 Model information</div>
            <div class="info-row">
                <span>Model format</span>
                <span class="info-value">Keras</span>
            </div>
            <div class="info-row">
                <span>Architecture</span>
                <span class="info-value">CNN</span>
            </div>
            <div class="info-row">
                <span>Input image</span>
                <span class="info-value">28 × 28 grayscale</span>
            </div>
            <div class="info-row">
                <span>Output classes</span>
                <span class="info-value">10</span>
            </div>
            <div class="info-row">
                <span>Model file</span>
                <span class="info-value">fashion_cnn.keras</span>
            </div>
            <div class="success-text">● Model loaded successfully</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-card">'
        '<div class="side-title">👚 Supported categories</div>',
        unsafe_allow_html=True,
    )

    for index, class_name in enumerate(CLASS_NAMES):
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                padding:7px 0;
                border-bottom:1px solid #edf0f6;
                color:#42516a;
                font-size:0.88rem;
            ">
                <span style="
                    display:inline-flex;
                    width:25px;
                    height:25px;
                    border-radius:8px;
                    align-items:center;
                    justify-content:center;
                    background:#edf0ff;
                    color:#554bc8;
                    font-size:0.75rem;
                    font-weight:800;
                ">{index}</span>
                <span>{class_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Keep the limitation available without making it the main message.
    with st.expander("About the model and its limitations"):
        st.write(
            "This classifier was trained on the Fashion-MNIST dataset, "
            "which contains 28 × 28 grayscale clothing images."
        )

        st.write(
            "The model is designed for ten predefined categories. "
            "Real-world photographs can differ from the training data "
            "in background, lighting, scale, orientation, and clothing "
            "appearance. Predictions may therefore be inaccurate."
        )

        st.write(
            "The displayed score describes the model output; it does "
            "not guarantee that a prediction is correct."
        )

    with st.expander("Technical details"):
        st.write(f"TensorFlow version: `{tf.__version__}`")
        st.write(f"NumPy version: `{np.__version__}`")
        st.write(f"Input shape: `{model.input_shape}`")
        st.write(f"Output shape: `{model.output_shape}`")
        st.write("Preprocessing: grayscale, resize to 28 × 28, normalize to [0, 1].")


# ============================================================
# 10. FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        <strong>Fashion AI</strong> &nbsp;·&nbsp;
        CNN-based image classification
        <br>
        Built with TensorFlow, Keras and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
