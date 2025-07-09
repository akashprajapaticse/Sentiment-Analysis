# sentiment_project/app.py

import streamlit as st
import pickle
import re
import nltk
import os
import pandas as pd # Used for pd.isna check in preprocess_text

# --- NLTK Downloads (Streamlit friendly) ---
# These are placed here so Streamlit can ensure the data is present when the app runs.
# In a production environment, you might handle these downloads during deployment setup.
@st.cache_resource # Cache the NLTK downloads to run only once
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    st.sidebar.success("NLTK data loaded.")

download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Preprocessing Function (MUST be identical to the one used during training) ---
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    try:
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = nltk.word_tokenize(text)
        stopwords_set = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopwords_set]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}") # Display error in Streamlit
        return ""

# --- Load the Trained Model ---
# Use st.cache_resource to load the model only once when the app starts
@st.cache_resource
def load_model():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models') # Correct path for relative access
    MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'sentiment_pipeline.pkl')
    
    try:
        with open(MODEL_PIPELINE_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PIPELINE_PATH}. Please ensure you have run train_model.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Product Review Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Product Review Sentiment Analysis")
st.write("---")
st.markdown("""
    This application predicts the sentiment of a product review (Positive, Negative, or Neutral) 
    using a machine learning model.
    """)

if model is None:
    st.warning("The sentiment analysis model could not be loaded. Please ensure 'train_model.py' has been run successfully to create 'sentiment_pipeline.pkl' and that it's located in the 'models/' directory (one level up from this app.py script).")
else:
    st.subheader("Enter Your Product Review:")
    user_input = st.text_area(
        "Type or paste your review here:", 
        height=180, 
        placeholder="E.g., This product is amazing, highly recommend it!",
        help="The model will analyze the sentiment of the text you enter."
    )

    if st.button("Analyze Sentiment", help="Click to get the sentiment prediction"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                processed_input = preprocess_text(user_input)
                
                if processed_input: # Ensure preprocessing didn't result in an empty string
                    prediction = model.predict([processed_input]) # model.predict expects a list-like input
                    sentiment = prediction[0]

                    st.markdown("---")
                    st.subheader("Prediction Result:")

                    if sentiment == 'positive':
                        st.success(f"## 😊 Positive")
                    elif sentiment == 'negative':
                        st.error(f"## 😠 Negative")
                    else: # neutral
                        st.info(f"## 😐 Neutral")

                    st.markdown("---")
                    st.subheader("Details:")
                    st.write(f"**Original Review:**")
                    st.code(user_input, language='text')
                    st.write(f"**Cleaned Review (as processed by model):**")
                    st.code(processed_input, language='text')

                else:
                    st.warning("The input review became empty after preprocessing. Please try a different review.")
        else:
            st.warning("Please enter some text in the review box to analyze its sentiment.")

st.write("---")
st.caption("Model developed with TF-IDF, SMOTE, and Multinomial Naive Bayes.")