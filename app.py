# ============================================================
# Streamlit App: AI vs Human Text Detection (BiLSTM + LIME)
# Page 1: Summary (How the project works)
# Page 2: Platform (Predict + Explain)
# ============================================================

# ------------------------------
# Import libraries
# ------------------------------
import streamlit as st  # streamlit UI
import numpy as np  # arrays
import tensorflow as tf  # load keras model
import joblib  # load tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # padding sequences
from lime.lime_text import LimeTextExplainer  # LIME for explanations
import streamlit.components.v1 as components  # render HTML in streamlit

# ------------------------------
# App config
# ------------------------------
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Paths (edit if needed)
# ------------------------------
BASE_PATH = "/content/drive/My Drive/TUD/Projects/AI Vs Human Text/"
MODEL_PATH = BASE_PATH + "advanced_bilstm_model.keras"
TOKENIZER_PATH = BASE_PATH + "tokenizer_word2vec.pkl"

# ------------------------------
# Model input length (must match training)
# ------------------------------
MAX_LEN = 300

# ------------------------------
# Load model + tokenizer once (cached)
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)  # load trained BiLSTM
    tokenizer = joblib.load(TOKENIZER_PATH)  # load tokenizer
    return model, tokenizer

# ------------------------------
# Convert a list of texts to model probabilities
# Returns: array of shape (n, 2) => [P(Human), P(AI)]
# ------------------------------
def predict_proba(text_list, model, tokenizer):
    seqs = tokenizer.texts_to_sequences(text_list)  # convert text to integer tokens
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")  # pad
    ai_probs = model.predict(padded, verbose=0).reshape(-1)  # model outputs P(AI)
    human_probs = 1.0 - ai_probs  # convert to P(Human)
    return np.vstack([human_probs, ai_probs]).T  # stack into [Human, AI]

# ------------------------------
# Sidebar navigation (2 pages)
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1) Project Summary", "2) Detection Platform"])

# ============================================================
# PAGE 1: SUMMARY
# ============================================================
if page == "1) Project Summary":

    st.title("🧠 AI vs Human Text Detection (BiLSTM + LIME)")
    st.write(
        """
        This project detects whether a given text is **Human-written** or **AI-generated** using a trained **BiLSTM** model.
        It also uses **LIME (Explainable AI)** to highlight which words influenced the model’s decision.
        """
    )

    st.subheader("How the system works (simple flow)")
    st.markdown(
        """
        **Step A — Input text**  
        You paste a text into the app.

        **Step B — Tokenizer**  
        The tokenizer converts words into integer IDs (the same mapping used during training).

        **Step C — Padding**  
        The sequence is padded/truncated to a fixed length (MAX_LEN = 300).

        **Step D — BiLSTM Prediction**  
        The BiLSTM outputs a probability for class **AI (1)**.  
        We also compute **Human (0) = 1 - AI**.

        **Step E — LIME Explanation**  
        LIME perturbs the input text (hides/changes some words) and observes how predictions change.  
        Then it reports which words push the prediction toward **Human** or **AI**.
        """
    )

    st.subheader("Classes")
    st.markdown(
        """
        - **Human = 0**
        - **AI = 1**
        """
    )

    st.subheader("What you will see on the Platform page")
    st.markdown(
        """
        - Predicted label (Human / AI)
        - Confidence score
        - LIME explanation (important words + weights)
        - A visual HTML explanation from LIME
        """
    )

# ============================================================
# PAGE 2: PLATFORM
# ============================================================
elif page == "2) Detection Platform":

    st.title("🧪 Detection Platform")
    st.write("Paste text below to classify it and see the LIME explanation.")

    # ------------------------------
    # Load artifacts (model + tokenizer)
    # ------------------------------
    try:
        model, tokenizer = load_artifacts()
    except Exception as e:
        st.error("Could not load model/tokenizer. Check paths and files.")
        st.exception(e)
        st.stop()

    # ------------------------------
    # Text input box
    # ------------------------------
    user_text = st.text_area(
        "Enter text here:",
        height=200,
        placeholder="Paste a paragraph here..."
    )

    # ------------------------------
    # Button to run prediction
    # ------------------------------
    if st.button("Predict & Explain"):

        # Basic validation
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
            st.stop()

        # ------------------------------
        # Predict probabilities
        # ------------------------------
        probs = predict_proba([user_text], model, tokenizer)[0]  # [P(Human), P(AI)]
        p_human, p_ai = float(probs[0]), float(probs[1])  # unpack probabilities

        # ------------------------------
        # Decide final label
        # ------------------------------
        if p_ai >= 0.5:
            label = "AI-generated"
            confidence = p_ai
        else:
            label = "Human-written"
            confidence = p_human

        # ------------------------------
        # Show prediction
        # ------------------------------
        st.subheader("Prediction")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")
        st.write(f"**P(Human):** {p_human:.4f}   |   **P(AI):** {p_ai:.4f}")

        # ------------------------------
        # Create LIME explainer
        # ------------------------------
        explainer = LimeTextExplainer(class_names=["Human", "AI"])

        # ------------------------------
        # Explain prediction with LIME
        # ------------------------------
        with st.spinner("Generating LIME explanation..."):
            exp = explainer.explain_instance(
                user_text,  # input text
                lambda texts: predict_proba(texts, model, tokenizer),  # prediction function
                num_features=15  # number of important words
            )

        # ------------------------------
        # Show word importance list
        # ------------------------------
        st.subheader("Top Important Words (LIME)")
        explanation_list = exp.as_list()  # list of (word, weight)

        # Display in a simple table
        st.table(
            [{"word": w, "weight": float(s)} for w, s in explanation_list]
        )

        # ------------------------------
        # Render LIME HTML visualization
        # ------------------------------
        st.subheader("LIME Visual Explanation")
        lime_html = exp.as_html()  # html string
        components.html(lime_html, height=400, scrolling=True)
