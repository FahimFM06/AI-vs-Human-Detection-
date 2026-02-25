# 🧠 AI vs Human Text Detection  
### BiLSTM + Explainable AI (LIME) + Streamlit Deployment

---

## 🔗 Live Demo (Streamlit App)

👉 https://kgtalyuz6zshlndvzgzzqz.streamlit.app/

---

## 📌 Project Overview

This project builds an **AI vs Human text detection system** using deep learning and explainable AI.  
The goal is to classify whether a given text is:

- **Human-written (0)**
- **AI-generated (1)**

The final deployed system uses:

- ✅ **BiLSTM (Deep Learning Model)**
- ✅ **Word2Vec-based Tokenizer**
- ✅ **LIME (Explainable AI)**
- ✅ **Streamlit Web Application**

---

## 📊 Dataset

- Original dataset: ~487K samples
- Balanced dataset created: **150,000 samples**
  - 75,000 Human
  - 75,000 AI
- Split into:
  - 70% Train
  - 15% Validation
  - 15% Test

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Lowercasing text
- Removing URLs
- Removing HTML tags
- Removing punctuation
- Removing numbers
- Removing extra spaces
- Removing very short texts
- Reset indexing
- Balanced sampling

---

## 🧠 Feature Representation Strategies Tested

Multiple vectorization techniques were explored:

### 🔹 Classical Approaches
- TF-IDF (Word-level)
- TF-IDF (Character-level)

### 🔹 Deep Learning Approaches
- Word2Vec embeddings (trained on dataset)
- Keras Tokenizer (sequence encoding)
- Transformer Tokenization:
  - DistilBERT
  - BERT
  - RoBERTa

---

## 🤖 Models Tested

### ✅ Baseline Models
- Logistic Regression
- Naive Bayes
- Support Vector Machine (LinearSVC)

### ✅ Deep Learning Models
- LSTM
- BiLSTM
- CNN (1D Convolution)

### ✅ Transformer Models
- DistilBERT
- BERT
- RoBERTa

---

## 🏆 Final Selected Model

After comparison, the **BiLSTM model** was selected because:

- Achieved ~**99.5% test accuracy**
- Strong generalization
- Stable validation performance
- Efficient inference time
- Easier integration with LIME for explainability

---

## 🧩 Final Architecture

**Pipeline:**

1. Input text  
2. Tokenizer (Word2Vec-based)  
3. Sequence padding (length = 300)  
4. BiLSTM model  
5. Output probability (AI vs Human)  
6. LIME explanation for word importance  

---

## 🔍 Explainable AI (LIME)

LIME is used to:

- Highlight important words influencing prediction
- Show positive/negative contributions
- Provide local interpretability
- Improve transparency and trust

Output includes:
- Word importance list
- Visual explanation

---

## 🌐 Streamlit Web Application

The project includes a fully deployed Streamlit app with:

### Page 1 – Project Summary
- Overview
- Workflow explanation
- Model details
- Architecture summary

### Page 2 – Detection Platform
- Text input
- Prediction result
- Confidence score
- LIME explanation
- Word importance visualization

---

## 📁 Project Structure

AI_vs_Human_Text_Detection/
│
├── app.py
├── requirements.txt
├── advanced_bilstm_model.keras
├── tokenizer_word2vec.pkl
├── README.md


---

## 📈 Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~97% |
| SVM | ~98% |
| CNN | ~99% |
| **BiLSTM (Final)** | **~99.5%** |

---

## 🎯 Key Contributions

✔ Balanced dataset engineering  
✔ Multi-vectorization comparison  
✔ Classical + Deep + Transformer benchmarking  
✔ BiLSTM optimization  
✔ Explainable AI integration  
✔ Full-stack ML deployment  

---

## 🏁 Final Outcome

This project delivers:

- A production-ready deep learning classifier  
- Explainable AI integration  
- Web deployment  
- Research-grade documentation  
- Portfolio-ready AI system  


