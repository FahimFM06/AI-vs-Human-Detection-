# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# - Page 1: Guide + How it works (Bk1 background)
# - Page 2: Prediction platform + LIME explanation (Bk2 background)
# Navigation: Continue / Back buttons (no sidebar needed)
# ============================================================

# ------------------------------
# Import libraries
# ------------------------------
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
st.set_page_config(
    page_title="AI vs Human Detection",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Paths (relative to GitHub repo)
# ------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"
BK1_PATH = APP_DIR / "Bk1.png"
BK2_PATH = APP_DIR / "Bk2.png"

# ------------------------------
# Must match your training sequence length
# ------------------------------
MAX_LEN = 300

# ------------------------------
# Helper: set background image with CSS
# ------------------------------
def set_background(image_path: Path):
    if not image_path.exists():
        st.warning(f"Background image not found: {image_path.name}")
        return

    img_bytes = image_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Make default text color bright for readability */
        html, body, [class*="css"]  {{
            color: #ffffff !important;
        }}

        /* Make text areas and inputs readable */
        textarea, input {{
            color: #000000 !important;
        }}

        /* Nice translucent container panel */
        .glass {{
            background: rgba(0, 0, 0, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}

        /* Button styling */
        .stButton>button {{
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# Load model + tokenizer once
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
# Predict probabilities
# Returns: array (n,2) => [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    ai_probs = model.predict(padded, verbose=0).reshape(-1)
    human_probs = 1.0 - ai_probs
    return np.vstack([human_probs, ai_probs]).T

# ------------------------------
# Session state for simple page navigation
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1  # start at page 1

# ============================================================
# PAGE 1: GUIDE + SUMMARY (Bk1)
# ============================================================
if st.session_state.page == 1:
    set_background(BK1_PATH)

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.title("🧠 AI vs Human Text Detection")
    st.write(
        """
        Welcome! This app predicts whether a text is **Human-written** or **AI-generated** using a trained **BiLSTM** model.
        It also provides **Explainable AI** using **LIME**, so you can see which words influenced the prediction.
        """
    )

    st.subheader("✅ How to use this app")
    st.markdown(
        """
        1. Click **Continue** to open the Detection Platform.  
        2. Paste your text into the input box.  
        3. Click **Predict & Explain**.  
        4. You will see:
           - Prediction label (Human / AI)
           - Confidence score
           - LIME explanation (important words + visual explanation)
        """
    )

    st.subheader("⚙️ How this system works (short)")
    st.markdown(
        """
        - **Tokenizer (Word2Vec-based)** converts words into integer IDs.  
        - Text is padded/truncated to **300 tokens**.  
        - **BiLSTM** outputs the probability of **AI (1)**.  
        - **LIME** highlights the words that most influenced the output.
        """
    )

    st.subheader("📌 Classes")
    st.markdown(
        """
        - **Human = 0**  
        - **AI = 1**
        """
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➡️ Continue"):
            st.session_state.page = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2: PLATFORM (Bk2)
# ============================================================
elif st.session_state.page == 2:
    set_background(BK2_PATH)

    # Load model/tokenizer safely
    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Ensure these files exist in your GitHub repo root:")
        st.code("advanced_bilstm_model.keras\ntokenizer_word2vec.pkl\nBk1.png\nBk2.png")
        st.exception(e)
        st.stop()

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.title("🧪 Detection Platform")
    st.write("Paste text below to classify it and view LIME explanations.")

    user_text = st.text_area(
        "Enter text here:",
        height=200,
        placeholder="Paste a paragraph here..."
    )

    colA, colB = st.columns([1, 1])

    with colA:
        run_btn = st.button("✅ Predict & Explain")

    with colB:
        if st.button("⬅️ Back"):
            st.session_state.page = 1
            st.rerun()

    if run_btn:
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
            st.stop()

        probs = predict_proba([user_text], model, tokenizer)[0]
        p_human, p_ai = float(probs[0]), float(probs[1])

        if p_ai >= 0.5:
            label = "AI-generated"
            confidence = p_ai
        else:
            label = "Human-written"
            confidence = p_human

        st.subheader("📌 Prediction")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")
        st.write(f"**P(Human):** {p_human:.4f}   |   **P(AI):** {p_ai:.4f}")

        st.subheader("🔍 Explainable AI (LIME)")
        explainer = LimeTextExplainer(class_names=["Human", "AI"])

        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_instance(
                user_text,
                lambda texts: predict_proba(texts, model, tokenizer),
                num_features=15
            )

        st.subheader("🧾 Top Important Words")
        st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])

        st.subheader("🧠 LIME Visual Explanation")
        components.html(exp.as_html(), height=450, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)
