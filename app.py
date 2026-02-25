# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# - Page 1: Hero + Guide (Bk1 background)
# - Page 2: Platform (Bk2 background)
# Improved readability: overlay + strong glass cards + typography
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import base64

from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="AI vs Human Detection", page_icon="🧠", layout="wide")

# ------------------------------
# Paths (relative to repo)
# ------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"
BK1_PATH = APP_DIR / "Bk1.png"
BK2_PATH = APP_DIR / "Bk2.png"

MAX_LEN = 300  # must match training

# ------------------------------
# Background with dark overlay + theme CSS
# ------------------------------
def set_background(image_path: Path, overlay_alpha: float = 0.70):
    if not image_path.exists():
        st.warning(f"Background image not found: {image_path.name}")
        return

    img_bytes = image_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        /* Full app background image */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Dark overlay to improve readability */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, {overlay_alpha});
            z-index: 0;
        }}

        /* Ensure all content stays above overlay */
        section[data-testid="stMain"] > div {{
            position: relative;
            z-index: 1;
        }}

        /* Global typography */
        h1, h2, h3, h4 {{
            color: #ffffff !important;
            text-shadow: 0px 2px 14px rgba(0,0,0,0.75);
            letter-spacing: 0.2px;
        }}

        p, li, span, div {{
            color: #EDEDED !important;
            font-size: 16px;
        }}

        /* Strong glass card */
        .glass {{
            background: rgba(15, 15, 18, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 22px;
            padding: 28px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
        }}

        /* Hero header style */
        .hero-title {{
            font-size: 44px;
            font-weight: 800;
            margin-bottom: 6px;
        }}

        .hero-sub {{
            font-size: 18px;
            opacity: 0.95;
            margin-top: 0;
        }}

        /* Section label pills */
        .pill {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
            font-size: 13px;
            margin-bottom: 10px;
        }}

        /* Make text area readable */
        textarea {{
            color: #000 !important;
            background: rgba(255,255,255,0.95) !important;
            border-radius: 14px !important;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 14px;
            padding: 0.65rem 1.2rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.22);
            background: linear-gradient(90deg, rgba(80,120,255,0.85), rgba(140,80,255,0.85));
            color: white !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            transition: 0.15s ease;
        }}

        /* Tables look better on dark */
        [data-testid="stTable"] {{
            background: rgba(15, 15, 18, 0.65) !important;
            border-radius: 16px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.12);
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
# Prediction function -> returns [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    ai_probs = model.predict(padded, verbose=0).reshape(-1)
    human_probs = 1.0 - ai_probs
    return np.vstack([human_probs, ai_probs]).T

# ------------------------------
# Session state navigation
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1

# ============================================================
# PAGE 1
# ============================================================
if st.session_state.page == 1:
    set_background(BK1_PATH, overlay_alpha=0.72)

    # Centered hero card
    left, mid, right = st.columns([1, 2.2, 1])
    with mid:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.markdown('<div class="pill">BiLSTM • Word2Vec Tokenizer • LIME Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">🧠 AI vs Human Text Detection</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Paste any text and get a prediction with word-level explanations showing <b>why</b> the model decided.</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("✅ How to use this app")
        st.markdown(
            """
            1. Click **Continue** to open the Detection Platform.  
            2. Paste your text into the input box.  
            3. Click **Predict & Explain**.  
            4. View prediction + confidence + LIME explanation.
            """
        )

        st.subheader("⚙️ How it works")
        st.markdown(
            """
            - **Tokenizer** converts words → integer IDs (same mapping as training).  
            - Text is padded/truncated to **300 tokens**.  
            - **BiLSTM** outputs probability of **AI (1)**.  
            - **LIME** highlights the words that most influenced the decision.  
            """
        )

        st.subheader("📌 Labels")
        st.markdown("- **Human = 0**  \n- **AI = 1**")

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("➡️ Continue"):
                st.session_state.page = 2
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2
# ============================================================
else:
    set_background(BK2_PATH, overlay_alpha=0.70)

    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Ensure these files exist in the repo root:")
        st.code("advanced_bilstm_model.keras\ntokenizer_word2vec.pkl\nBk1.png\nBk2.png")
        st.exception(e)
        st.stop()

    # Wider platform card
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.title("🧪 Detection Platform")
    st.write("Paste your text below. The app will predict **Human vs AI** and explain the decision using **LIME**.")

    user_text = st.text_area("Enter text here:", height=220, placeholder="Paste a paragraph here...")

    colA, colB, colC = st.columns([1.2, 1, 3])
    with colA:
        run_btn = st.button("✅ Predict & Explain")
    with colB:
        if st.button("⬅️ Back"):
            st.session_state.page = 1
            st.rerun()

    if run_btn:
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            probs = predict_proba([user_text], model, tokenizer)[0]
            p_human, p_ai = float(probs[0]), float(probs[1])

            if p_ai >= 0.5:
                label = "AI-generated"
                confidence = p_ai
            else:
                label = "Human-written"
                confidence = p_human

            st.subheader("📌 Prediction")
            st.markdown(
                f"""
                **Label:** `{label}`  
                **Confidence:** `{confidence:.4f}`  
                **P(Human):** `{p_human:.4f}`  |  **P(AI):** `{p_ai:.4f}`
                """
            )

            st.subheader("🔍 Explainable AI (LIME)")
            explainer = LimeTextExplainer(class_names=["Human", "AI"])

            with st.spinner("Generating explanation..."):
                exp = explainer.explain_instance(
                    user_text,
                    lambda texts: predict_proba(texts, model, tokenizer),
                    num_features=15
                )

            st.subheader("🧾 Top Important Words")
            st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])

            st.subheader("🧠 LIME Visual Explanation")
            components.html(exp.as_html(), height=500, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)
