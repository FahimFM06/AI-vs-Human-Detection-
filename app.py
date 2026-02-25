# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# - Page 1: Summary (How it works)
# - Page 2: Platform (Predict + Explain)
# ============================================================

# ------------------------------
# Import libraries
# ------------------------------
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Resolve paths RELATIVE to this app.py file (works on Streamlit Cloud)
# ------------------------------
APP_DIR = Path(__file__).resolve().parent  # folder where app.py is located
MODEL_PATH = APP_DIR / "advanced_bilstm_model.keras"
TOKENIZER_PATH = APP_DIR / "tokenizer_word2vec.pkl"

# ------------------------------
# Must match your training sequence length
# ------------------------------
MAX_LEN = 300

# ------------------------------
# Load model + tokenizer once (cached)
# ------------------------------
@st.cache_resource
def load_artifacts():
    # Check model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Check tokenizer file exists
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

    # Load BiLSTM model
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Load tokenizer
    tokenizer = joblib.load(TOKENIZER_PATH)

    return model, tokenizer

# ------------------------------
# Predict probabilities for LIME + UI
# Returns: array (n,2) -> [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    # Convert raw text -> integer sequences
    seqs = tokenizer.texts_to_sequences(text_list)

    # Pad/truncate sequences
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    # Model outputs probability of AI (class 1)
    ai_probs = model.predict(padded, verbose=0).reshape(-1)

    # Convert to 2-class probabilities
    human_probs = 1.0 - ai_probs

    # Return [Human, AI]
    return np.vstack([human_probs, ai_probs]).T

# ------------------------------
# Sidebar navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1) Project Summary", "2) Detection Platform"])

# ============================================================
# PAGE 1: SUMMARY
# ============================================================
if page == "1) Project Summary":
    st.title("🧠 AI vs Human Text Detection (BiLSTM + LIME Explainability)")

    st.write(
        """
        This project detects whether a text is **Human-written** or **AI-generated** using a trained **BiLSTM** model.
        It also uses **LIME (Explainable AI)** to highlight which words influenced the decision.
        """
    )

    st.subheader("Workflow")
    st.markdown(
        """
        1. **Input Text** → user pastes a paragraph  
        2. **Tokenizer** → converts words into integer IDs (same mapping used during training)  
        3. **Padding** → sequences padded/truncated to fixed length (**300 tokens**)  
        4. **BiLSTM Prediction** → outputs probability of AI text  
        5. **LIME Explanation** → shows important words contributing to the prediction  
        """
    )

    st.subheader("Classes")
    st.markdown(
        """
        - **Human = 0**
        - **AI = 1**
        """
    )

    st.subheader("What the platform shows")
    st.markdown(
        """
        - Predicted label (Human / AI)  
        - Confidence score  
        - Word importance list (LIME)  
        - Visual explanation (LIME HTML)  
        """
    )

# ============================================================
# PAGE 2: PLATFORM
# ============================================================
elif page == "2) Detection Platform":
    st.title("🧪 Detection Platform")
    st.write("Paste text below to classify it and see the LIME explanation.")

    # Load artifacts safely
    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Make sure these files exist in your GitHub repo root:")
        st.code("advanced_bilstm_model.keras\ntokenizer_word2vec.pkl")
        st.exception(e)
        st.stop()

    # Text input
    user_text = st.text_area(
        "Enter text here:",
        height=200,
        placeholder="Paste a paragraph here..."
    )

    # Run prediction button
    if st.button("Predict & Explain"):
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
            st.stop()

        # Predict
        probs = predict_proba([user_text], model, tokenizer)[0]
        p_human, p_ai = float(probs[0]), float(probs[1])

        # Decide label
        if p_ai >= 0.5:
            label = "AI-generated"
            confidence = p_ai
        else:
            label = "Human-written"
            confidence = p_human

        # Display prediction
        st.subheader("Prediction")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")
        st.write(f"**P(Human):** {p_human:.4f}   |   **P(AI):** {p_ai:.4f}")

        # LIME explanation
        st.subheader("Explainable AI (LIME)")
        explainer = LimeTextExplainer(class_names=["Human", "AI"])

        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_instance(
                user_text,
                lambda texts: predict_proba(texts, model, tokenizer),
                num_features=15
            )

        # Show top words
        st.subheader("Top Important Words")
        st.table([{"word": w, "weight": float(s)} for w, s in exp.as_list()])

        # Show LIME visualization
        st.subheader("LIME Visual Explanation")
        components.html(exp.as_html(), height=450, scrolling=True)
