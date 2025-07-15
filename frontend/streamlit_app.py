# frontend/streamlit_app.py

import streamlit as st

# ─── Page config must be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1) Download NLTK data (no punkt) ----------------------------------------
@st.cache_resource
def download_nltk_data():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet',   'wordnet'),
        ('corpora/omw-1.4',   'omw-1.4'),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)
    st.sidebar.success("✅ NLTK data loaded.")

download_nltk_data()

# --- 2) Preprocessing (regex tokenizer instead of nltk.word_tokenize) --------
def preprocess_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Lowercase, strip URLs, HTML tags, punctuation, numbers, extra spaces
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # **Regex tokenizer** (no external data needed)
    tokens = re.findall(r"\b\w+\b", text)

    # Remove stopwords + lemmatize
    sw = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in sw]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(tok) for tok in tokens]

    return " ".join(tokens)

# --- 3) Load pickled pipeline -----------------------------------------------
@st.cache_resource
def load_model():
    # __file__ → frontend/streamlit_app.py, so go up one to project root
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    model_path = os.path.join(base, "models", "sentiment_pipeline.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at:\n  {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- 4) Streamlit UI --------------------------------------------------------
st.title("💬 Product Review Sentiment Analysis")
st.write("---")
st.markdown(
    "This app predicts whether a review is **Positive**, **Negative**, or **Neutral** "
    "using a TF‑IDF → SMOTE → MultinomialNB pipeline."
)

if model is None:
    st.warning(
        "Model pipeline isn’t loaded. Make sure you’ve run your training script and "
        "`sentiment_pipeline.pkl` sits in the `models/` folder."
    )
else:
    review = st.text_area(
        "Enter your product review:",
        height=180,
        placeholder="E.g., “Absolutely loved it—would buy again!”",
    )
    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please type a review before clicking Analyze.")
        else:
            with st.spinner("🔎 Analyzing..."):
                clean = preprocess_text(review)
                if not clean:
                    st.warning("Text cleaned to empty—try a different review.")
                else:
                    pred = model.predict([clean])[0]
                    st.markdown("---")
                    if pred == "positive":
                        st.success("## 😊 Positive")
                    elif pred == "negative":
                        st.error("## 😠 Negative")
                    else:
                        st.info("## 😐 Neutral")

                    st.markdown("---")
                    st.subheader("📄 Review Details")
                    st.write("**Original:**")
                    st.code(review)
                    st.write("**Cleaned for model:**")
                    st.code(clean)

st.write("---")
st.caption("Model pipeline: TF‑IDF → SMOTE → MultinomialNB")
