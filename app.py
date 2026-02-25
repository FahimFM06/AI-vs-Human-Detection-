# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# Page 1: Clean glass sections
# Page 2: Water-style UI + LIME explanation shown like the screenshot
#         (using Matplotlib with WHITE background, big size)
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import base64

from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt  # for LIME pyplot figure

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="AI vs Human Detection", page_icon="🧠", layout="wide")

# ------------------------------
# Paths (repo root)
# ------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"
BK1_PATH = APP_DIR / "Bk1.png"
BK2_PATH = APP_DIR / "Bk2.png"

MAX_LEN = 300  # must match training input length


# ------------------------------
# Background + CSS theme
# ------------------------------
def set_background(image_path: Path, overlay_alpha: float = 0.72):
    if not image_path.exists():
        st.warning(f"Background image not found: {image_path.name}")
        return

    encoded = base64.b64encode(image_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, {overlay_alpha});
            z-index: 0;
        }}

        section[data-testid="stMain"] > div {{
            position: relative;
            z-index: 1;
        }}

        h1,h2,h3,h4 {{
            color:#ffffff !important;
            text-shadow:0 2px 14px rgba(0,0,0,.75);
        }}

        p,li,span,div {{
            color:#F1F1F1 !important;
            font-size:16px;
        }}

        /* Page 1 section blocks */
        .section {{
            background: rgba(10, 12, 16, 0.68);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 18px;
            margin: 12px 0px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
            backdrop-filter: blur(10px);
        }}

        /* Page 2 water blocks */
        .water {{
            background: rgba(0, 30, 45, 0.62);
            border: 1px solid rgba(120, 220, 255, 0.25);
            border-radius: 20px;
            padding: 22px;
            margin: 12px 0px;
            box-shadow: 0 12px 38px rgba(0,0,0,0.45);
            backdrop-filter: blur(14px);
        }}

        textarea {{
            color:#0b0d10 !important;
            background: rgba(255,255,255,0.92) !important;
            border-radius: 14px !important;
        }}

        .stButton > button {{
            border-radius: 14px;
            padding: 0.65rem 1.2rem;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,0.22);
            background: linear-gradient(90deg, rgba(40,160,255,0.9), rgba(0,220,200,0.85));
            color: white !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        }}

        [data-testid="stTable"] {{
            background: rgba(0, 25, 40, 0.45) !important;
            border-radius: 16px;
            padding: 10px;
            border: 1px solid rgba(120, 220, 255, 0.18);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ------------------------------
# Load model + tokenizer
# ------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer


# ------------------------------
# Predict probabilities -> [Human, AI]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    ai_probs = model.predict(padded, verbose=0).reshape(-1)
    human_probs = 1.0 - ai_probs
    return np.vstack([human_probs, ai_probs]).T


# ------------------------------
# Navigation state
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1


# ============================================================
# PAGE 1
# ============================================================
if st.session_state.page == 1:
    set_background(BK1_PATH, overlay_alpha=0.78)

    st.title("🧠 AI vs Human Text Detection")
    st.write(
        "Type or paste any text — the app will predict Human or AI, "
        "and then show the exact words that influenced the decision."
    )

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🚀 What you can do here")
    st.markdown(
        """
        ✅ Detect AI-generated vs Human-written text  
        ✅ See confidence (how sure the model is)  
        ✅ Use Explainable AI (LIME) to understand why the model decided  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🧭 How to use (3 simple steps)")
    st.markdown(
        """
        1. Click **Continue**  
        2. Paste your text  
        3. Click **Predict & Explain**  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ How the model works (easy)")
    st.markdown(
        """
        - Tokenizer converts words → numbers (IDs).  
        - Text padded to **300 tokens**.  
        - BiLSTM outputs probability of **AI (1)**.  
        - LIME shows which words influenced the prediction.  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🏷️ Labels")
    st.markdown("- Human = 0  \n- AI = 1")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("➡️ Continue"):
        st.session_state.page = 2
        st.rerun()


# ============================================================
# PAGE 2
# ============================================================
else:
    set_background(BK2_PATH, overlay_alpha=0.74)

    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Ensure files exist in repo root.")
        st.exception(e)
        st.stop()

    st.markdown('<div class="water">', unsafe_allow_html=True)
    st.title("🧪 Detection Platform")
    st.write("Paste your text below. You’ll get a prediction + confidence + LIME explanation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="water">', unsafe_allow_html=True)
    user_text = st.text_area("Enter text here:", height=220, placeholder="Paste a paragraph here...")

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("✅ Predict & Explain")
    with colB:
        if st.button("⬅️ Back"):
            st.session_state.page = 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            probs = predict_proba([user_text], model, tokenizer)[0]
            p_human, p_ai = float(probs[0]), float(probs[1])

            label = "AI-generated" if p_ai >= 0.5 else "Human-written"
            confidence = p_ai if p_ai >= 0.5 else p_human

            st.markdown('<div class="water">', unsafe_allow_html=True)
            st.subheader("📌 Prediction")
            st.markdown(
                f"""
                **Label:** `{label}`  
                **Confidence:** `{confidence:.4f}`  
                **P(Human):** `{p_human:.4f}`  |  **P(AI):** `{p_ai:.4f}`
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ------------------------------
            # LIME Explanation
            # ------------------------------
            explainer = LimeTextExplainer(class_names=["Human", "AI"])

            with st.spinner("Generating LIME explanation..."):
                exp = explainer.explain_instance(
                    user_text,
                    lambda texts: predict_proba(texts, model, tokenizer),
                    num_features=15
                )

            # Top important words table
            st.markdown('<div class="water">', unsafe_allow_html=True)
            st.subheader("🧾 Top Important Words")
            st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])
            st.markdown("</div>", unsafe_allow_html=True)

            # ✅ LIME Visual Explanation like the screenshot (WHITE background)
            st.markdown('<div class="water">', unsafe_allow_html=True)
            st.subheader("🧠 LIME Visual Explanation (like screenshot)")

            fig = exp.as_pyplot_figure()      # create matplotlib figure
            fig.set_size_inches(12, 5)        # make bigger for readability

            # Force white background (important!)
            fig.patch.set_facecolor("white")
            for ax in fig.axes:
                ax.set_facecolor("white")

            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
