# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# Fix: Make LIME Visual Explanation readable (force white bg + black text)
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

st.set_page_config(page_title="AI vs Human Detection", page_icon="🧠", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"
BK1_PATH = APP_DIR / "Bk1.png"
BK2_PATH = APP_DIR / "Bk2.png"

MAX_LEN = 300

# ------------------------------
# Background + theme CSS
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

        h1,h2,h3,h4 {{ color:#fff !important; text-shadow:0 2px 14px rgba(0,0,0,.75); }}
        p,li,span,div {{ color:#F1F1F1 !important; font-size:16px; }}

        .glass {{
            background: rgba(15, 15, 18, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 22px;
            padding: 26px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
        }}

        textarea {{
            color:#000 !important;
            background: rgba(255,255,255,0.96) !important;
            border-radius: 14px !important;
        }}

        .stButton > button {{
            border-radius: 14px;
            padding: 0.65rem 1.2rem;
            font-weight: 750;
            border: 1px solid rgba(255,255,255,0.22);
            background: linear-gradient(90deg, rgba(80,120,255,0.88), rgba(140,80,255,0.88));
            color: white !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        }}

        /* Make iframe full width */
        iframe {{ width: 100% !important; }}
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
# Predict probabilities -> [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    ai_probs = model.predict(padded, verbose=0).reshape(-1)
    human_probs = 1.0 - ai_probs
    return np.vstack([human_probs, ai_probs]).T

# ------------------------------
# IMPORTANT: Force LIME HTML to be readable
# ------------------------------
def make_lime_html_readable(html: str) -> str:
    # This CSS is injected INSIDE the LIME HTML document
    # It forces white background + black text + larger font
    readable_css = """
    <style>
      body { background: #ffffff !important; color: #111111 !important; font-size: 16px !important; }
      * { color: #111111 !important; font-family: Arial, sans-serif !important; }
      h1,h2,h3,h4 { color: #111111 !important; }
      .lime { max-width: 100% !important; }
      table { font-size: 14px !important; }
      /* Make highlighted text easier to see */
      span { font-weight: 700 !important; }
      /* Improve contrast of charts/labels */
      text { fill: #111111 !important; }
    </style>
    """

    # Insert CSS right after <head> if present
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + readable_css)
    else:
        # fallback: prepend
        html = readable_css + html

    return html

# ------------------------------
# Navigation state
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1

# ============================================================
# PAGE 1
# ============================================================
if st.session_state.page == 1:
    set_background(BK1_PATH, overlay_alpha=0.75)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title("🧠 AI vs Human Text Detection")
    st.write("Paste any text and get a prediction with word-level explanations showing **why** the model decided.")
    st.subheader("✅ How to use")
    st.markdown("1) Click **Continue**  \n2) Paste text  \n3) Click **Predict & Explain**")
    st.subheader("⚙️ What happens inside")
    st.markdown("- Tokenizer converts words → IDs  \n- Text padded to **300 tokens**  \n- BiLSTM predicts **AI probability**  \n- LIME highlights important words")
    st.subheader("🏷️ Labels")
    st.markdown("- Human = 0  \n- AI = 1")

    if st.button("➡️ Continue"):
        st.session_state.page = 2
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2
# ============================================================
else:
    set_background(BK2_PATH, overlay_alpha=0.73)

    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Ensure files exist in repo root.")
        st.exception(e)
        st.stop()

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title("🧪 Detection Platform")

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
                f"**Label:** `{label}`  \n"
                f"**Confidence:** `{confidence:.4f}`  \n"
                f"**P(Human):** `{p_human:.4f}`  |  **P(AI):** `{p_ai:.4f}`"
            )

            st.subheader("🧾 Top Important Words (LIME)")
            explainer = LimeTextExplainer(class_names=["Human", "AI"])

            with st.spinner("Generating explanation..."):
                exp = explainer.explain_instance(
                    user_text,
                    lambda texts: predict_proba(texts, model, tokenizer),
                    num_features=15
                )

            st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])

            # ✅ Make LIME HTML readable
            st.subheader("🧠 LIME Visual Explanation (Readable)")
            lime_html = make_lime_html_readable(exp.as_html())

            # Put it in a big iframe so user can read without scrolling too much
            components.html(lime_html, height=650, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)
