# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# Page 1: Clean guide (keep like your screenshot)
# Page 2: "Water / frosted glass" style so all text is visible (white text)
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
# Paths
# ------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"
BK1_PATH = APP_DIR / "Bk1.png"
BK2_PATH = APP_DIR / "Bk2.png"

MAX_LEN = 300  # must match training

# ------------------------------
# Background + CSS
# mode="page1" or "page2"
# ------------------------------
def set_background(image_path: Path, mode: str = "page1"):
    if not image_path.exists():
        st.warning(f"Background image not found: {image_path.name}")
        return

    encoded = base64.b64encode(image_path.read_bytes()).decode()

    # Page 1 overlay is simple dark
    if mode == "page1":
        overlay = "rgba(0,0,0,0.72)"

    # Page 2 overlay has "water" effect (gradient + blur look)
    else:
        overlay = """
        linear-gradient(120deg,
            rgba(10, 25, 45, 0.78),
            rgba(18, 40, 65, 0.74),
            rgba(5, 20, 30, 0.76)
        )
        """

    st.markdown(
        f"""
        <style>
        /* Background image */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Overlay layer */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: {overlay};
            z-index: 0;
        }}

        /* Ensure content stays above overlay */
        section[data-testid="stMain"] > div {{
            position: relative;
            z-index: 1;
        }}

        /* Headings */
        h1,h2,h3,h4 {{
            color: #ffffff !important;
            text-shadow: 0 2px 14px rgba(0,0,0,0.75);
        }}

        /* Default text */
        p, li, span, div {{
            color: #F3F6FF !important;
            font-size: 16px;
        }}

        /* Page 1: clean transparent separators */
        .line {{
            height: 1px;
            background: rgba(255,255,255,0.18);
            margin: 14px 0;
            border-radius: 999px;
        }}

        /* Page 2: water / frosted glass card */
        .water-card {{
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 18px 60px rgba(0,0,0,0.55);
            backdrop-filter: blur(14px);
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 14px;
            padding: 0.65rem 1.2rem;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,0.20);
            background: linear-gradient(90deg, rgba(50,160,255,0.85), rgba(120,90,255,0.85));
            color: #fff !important;
            box-shadow: 0 10px 22px rgba(0,0,0,0.35);
        }}

        /* Text area readable */
        textarea {{
            color: #0A0A0A !important;
            background: rgba(255,255,255,0.96) !important;
            border-radius: 14px !important;
        }}

        /* Make iframe full width */
        iframe {{
            width: 100% !important;
        }}

        /* On page 2: make Streamlit tables LIGHT with dark text */
        [data-testid="stTable"] {{
            background: rgba(255,255,255,0.92) !important;
            border-radius: 14px;
            padding: 10px;
            border: 1px solid rgba(0,0,0,0.10);
        }}
        [data-testid="stTable"] * {{
            color: #111111 !important;
        }}

        /* LIME viewer white box (always readable) */
        .lime-viewer {{
            background: rgba(255,255,255,0.98);
            border-radius: 16px;
            padding: 14px;
            border: 1px solid rgba(0,0,0,0.10);
            box-shadow: 0 10px 28px rgba(0,0,0,0.25);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# Load artifacts
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
# Predict probabilities -> [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    ai_probs = model.predict(padded, verbose=0).reshape(-1)
    human_probs = 1.0 - ai_probs
    return np.vstack([human_probs, ai_probs]).T

# ------------------------------
# Fix LIME html readability inside iframe
# ------------------------------
def make_lime_html_readable(html: str) -> str:
    css = """
    <style>
      body { background: #ffffff !important; color: #111 !important; font-size: 16px !important; }
      * { color: #111 !important; font-family: Arial, sans-serif !important; }
      table { font-size: 14px !important; }
      text { fill: #111 !important; }
    </style>
    """
    if "<head>" in html:
        return html.replace("<head>", "<head>" + css)
    return css + html

# ------------------------------
# Navigation state
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1

# ============================================================
# PAGE 1 (keep like your screenshot)
# ============================================================
if st.session_state.page == 1:
    set_background(BK1_PATH, mode="page1")

    st.title("🧠 AI vs Human Text Detection")
    st.write("Type or paste any text — the app will predict **Human** or **AI**, and then show the exact words that influenced the decision.")
    st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    st.subheader("🚀 What you can do here")
    st.markdown(
        """
        ✅ Detect AI-generated vs Human-written text  
        ✅ See confidence (how sure the model is)  
        ✅ Use Explainable AI (LIME) to understand why the model decided  
        """
    )
    st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    st.subheader("🧭 How to use (3 simple steps)")
    st.markdown(
        """
        1. Click **Continue**  
        2. Paste your text  
        3. Click **Predict & Explain**  
        """
    )
    st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    st.subheader("⚙️ How the model works (easy)")
    st.markdown(
        """
        - The **Tokenizer** converts words → numbers (IDs).  
        - We pad the text to **300 tokens** so the model always gets the same input size.  
        - The **BiLSTM** learns writing patterns and outputs probability of **AI (1)**.  
        - **LIME** highlights the words that pushed the prediction toward Human or AI.  
        """
    )
    st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    st.subheader("🏷️ Labels")
    st.markdown("- Human = 0  \n- AI = 1")

    if st.button("➡️ Continue"):
        st.session_state.page = 2
        st.rerun()

# ============================================================
# PAGE 2 (water / frosted glass style + white text)
# ============================================================
else:
    set_background(BK2_PATH, mode="page2")

    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Ensure files exist in repo root.")
        st.exception(e)
        st.stop()

    # Water-style card container
    st.markdown('<div class="water-card">', unsafe_allow_html=True)

    st.title("🧪 Detection Platform")
    st.write("Paste your text below — you’ll get prediction, confidence, and LIME explanation.")

    user_text = st.text_area("Enter text here:", height=220, placeholder="Paste a paragraph here...")

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
        else:
            probs = predict_proba([user_text], model, tokenizer)[0]
            p_human, p_ai = float(probs[0]), float(probs[1])

            label = "AI-generated" if p_ai >= 0.5 else "Human-written"
            confidence = p_ai if p_ai >= 0.5 else p_human

            st.subheader("📌 Prediction")
            st.markdown(
                f"""
                **Label:** `{label}`  
                **Confidence:** `{confidence:.4f}`  
                **P(Human):** `{p_human:.4f}`  |  **P(AI):** `{p_ai:.4f}`
                """
            )

            st.subheader("🧾 Top Important Words (LIME)")
            explainer = LimeTextExplainer(class_names=["Human", "AI"])

            with st.spinner("Generating explanation..."):
                exp = explainer.explain_instance(
                    user_text,
                    lambda texts: predict_proba(texts, model, tokenizer),
                    num_features=15
                )

            # This table is now forced to light background + dark text via CSS
            st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])

            st.subheader("🧠 LIME Visual Explanation")
            lime_html = make_lime_html_readable(exp.as_html())

            # White panel for LIME HTML
            st.markdown('<div class="lime-viewer">', unsafe_allow_html=True)
            components.html(lime_html, height=650, scrolling=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
