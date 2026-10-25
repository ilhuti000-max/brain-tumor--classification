import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# -------------------- CONFIGURATION --------------------
MODEL_PATH = os.getenv("BRAIN_MODEL_PATH", "brain_tumor_model.keras")

# IMPORTANT: Match these to the exact preprocessing used during training.
IMAGE_SIZE = (224, 224)  # Change to the training size, e.g. (150, 150)
RESCALE_BY_255 = True    # Set False if your model was trained on another scale.
MODEL_COLOR_MODE = "RGB" # Use "L" only if the model was trained on grayscale images.

# For a 2-output softmax model, this must match training class-index order.
# Example: {"no_tumor": 0, "tumor": 1}
CLASS_TO_INDEX = {"no_tumor": 0, "tumor": 1}

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- STYLING --------------------
st.markdown(
    """
    <style>
    .stApp { background: #f4f8fd; color: #17365d; }
    [data-testid="stHeader"] { background: rgba(244,248,253,0.92); }
    .hero {
        background: linear-gradient(115deg, #17395e, #245b8f);
        color: white; padding: 1.35rem 1.6rem; border-radius: 18px;
        margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(24,58,95,.12);
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #d9eaff; margin: .35rem 0 0 0; }
    .panel {
        background: white; padding: 1.2rem; border-radius: 16px;
        border: 1px solid #dce8f7; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(30,70,110,.045);
    }
    .muted { color: #5c7391; font-size: .94rem; }
    .result-tumor {
        border: 1px solid #ffd5d9; background: #fff5f5; color: #b42332;
        padding: 1rem; border-radius: 13px;
    }
    .result-clear {
        border: 1px solid #bdebd7; background: #effcf6; color: #08764e;
        padding: 1rem; border-radius: 13px;
    }
    .warning {
        background: #fff6e5; border: 1px solid #ffdf9e; color: #704b00;
        padding: 1rem; border-radius: 12px; font-size: .92rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fbff; border: 1px solid #e2ecf8;
        padding: .8rem; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model(model_path: str):
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model at '{model_path}'. Check MODEL_PATH and "
            "make sure the model was saved in a compatible Keras format."
        ) from exc


def prepare_image(pil_image: Image.Image) -> np.ndarray:
    """Keep this preprocessing identical to the model's training pipeline."""
    image = ImageOps.exif_transpose(pil_image).convert(MODEL_COLOR_MODE)
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)

    if MODEL_COLOR_MODE == "L":
        array = np.expand_dims(array, axis=-1)

    if RESCALE_BY_255:
        array = array / 255.0

    return np.expand_dims(array, axis=0)


def get_probabilities(model, batch: np.ndarray):
    """Returns (tumor_probability, no_tumor_probability)."""
    prediction = np.asarray(model.predict(batch, verbose=0))

    # Binary sigmoid model: output must represent the tumor class (label 1).
    if prediction.size == 1:
        tumor_probability = float(prediction.reshape(-1)[0])
        tumor_probability = float(np.clip(tumor_probability, 0.0, 1.0))
        return tumor_probability, 1.0 - tumor_probability

    # Two-class softmax model: class index order must match training.
    flat = prediction.reshape(-1)
    if flat.size == 2:
        no_tumor_idx = CLASS_TO_INDEX["no_tumor"]
        tumor_idx = CLASS_TO_INDEX["tumor"]
        no_tumor_probability = float(flat[no_tumor_idx])
        tumor_probability = float(flat[tumor_idx])
        total = no_tumor_probability + tumor_probability
        if total > 0:
            no_tumor_probability /= total
            tumor_probability /= total
        return tumor_probability, no_tumor_probability

    raise ValueError(
        f"Unexpected model output shape {prediction.shape}. "
        "Adapt get_probabilities() to your model's output."
    )


# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="hero">
      <h1>Brain Tumor Detection</h1>
      <p>AI-powered MRI image classification · Research prototype</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## About this project")
    st.write(
        "A deep-learning demonstration that classifies brain MRI images "
        "as **tumor** or **no tumor**."
    )
    st.markdown("---")
    st.markdown("### Important disclaimer")
    st.warning(
        
        "must not be used to make clinical decisions or replace professional "
        "medical advice."
    )

left, center, right = st.columns([1.02, 1.55, 1.0], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(" Upload MRI")
    st.write("Upload a brain MRI image (JPG, JPEG, or PNG).")
    uploaded_file = st.file_uploader(
        "Choose MRI image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.caption("Use images you are authorized to process. Avoid identifiable patient information.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Before running")
    st.markdown(
        "- Confirm image size and color mode match training.\n"
        "- Confirm the tumor class index.\n"
        "- Validate with unseen data."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(" MRI preview")
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Try a JPG or PNG file.")
            image = None
    else:
        image = None
        st.info("Your uploaded MRI image will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(" Prediction result")

    run_prediction = st.button(
        "▶ Run prediction",
        type="primary",
        use_container_width=True,
        disabled=(image is None),
    )

    if run_prediction and image is not None:
        if not Path(MODEL_PATH).exists():
            st.error(
                f"Model file not found: `{MODEL_PATH}`. Put your trained model "
                "in the same folder as app.py or set BRAIN_MODEL_PATH."
            )
        else:
            try:
                with st.spinner("Analyzing image…"):
                    model = load_model(MODEL_PATH)
                    batch = prepare_image(image)
                    tumor_prob, no_tumor_prob = get_probabilities(model, batch)

                
                
                is_tumor = tumor_prob >= no_tumor_prob
                if is_tumor:
                    st.markdown(
                        f'<div class="result-tumor"><h3>⚠️ Tumor predicted</h3>'
                        f'<p>Model-estimated tumor probability: <b>{tumor_prob:.1%}</b></p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-clear"><h3>✓ No tumor predicted</h3>'
                        f'<p>Model-estimated no-tumor probability: <b>{no_tumor_prob:.1%}</b></p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                m1, m2 = st.columns(2)
                m1.metric("Tumor probability", f"{tumor_prob:.1%}")
                m2.metric("No-tumor probability", f"{no_tumor_prob:.1%}")
                st.write("Class probabilities")
                st.progress(tumor_prob, text=f"Tumor — {tumor_prob:.1%}")
                st.progress(no_tumor_prob, text=f"No tumor — {no_tumor_prob:.1%}")
                st.caption(
                    "These are model scores, not guaranteed clinical probabilities. "
                    "Performance on this image may differ from test-set performance."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check model format, input shape, preprocessing, and output "
                    "class order against your training notebook."
                )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(" About this result")
    st.write(
        "The displayed scores are model outputs and may not reflect performance "
        "on new images."
    )
    st.markdown("</div>", unsafe_allow_html=True)
