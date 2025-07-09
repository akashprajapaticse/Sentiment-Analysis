# sentiment_project/frontend/streamlit_app.py

import streamlit as st
import requests
import json
import os
import sys

# Add the project root to the sys path to enable importing from scripts/
# This is crucial for the Streamlit app to find the preprocessor, if needed for local prediction
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Option 1: Direct model loading (if you want Streamlit to run model locally)
# import pickle
# from scripts.train_model import preprocess_text
# try:
#     MODEL_DIR = os.path.join(project_root, 'models')
#     MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
#     VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
#
#     with open(MODEL_PATH, 'rb') as model_file:
#         local_model = pickle.load(model_file)
#     with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
#         local_vectorizer = pickle.load(vectorizer_file)
#     st.success("Local model loaded successfully!")
#     LOCAL_MODEL_LOADED = True
# except Exception as e:
#     st.warning(f"Could not load local model (running via API instead): {e}")
#     LOCAL_MODEL_LOADED = False


# Option 2: Use the FastAPI backend (Recommended for separation of concerns)
# You need to run the FastAPI app separately (e.g., uvicorn app.api:app --reload)
FASTAPI_URL = "http://127.0.0.1:8000/predict" # Adjust if your FastAPI runs on a different port or host

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("Product Review Sentiment Analyzer")
st.markdown("Enter a product review below to get its sentiment prediction (Positive, Negative, or Neutral).")

user_input = st.text_area("Enter your review here:", height=150, placeholder="e.g., This product is amazing, I love it!")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                # --- Using FastAPI Backend ---
                headers = {'Content-Type': 'application/json'}
                data = json.dumps({"text": user_input})
                response = requests.post(FASTAPI_URL, headers=headers, data=data)

                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment")
                    st.write("---")
                    st.subheader("Predicted Sentiment:")
                    if sentiment.lower() == "positive":
                        st.success(f"**Positive!** 🎉")
                    elif sentiment.lower() == "negative":
                        st.error(f"**Negative!** 😠")
                    elif sentiment.lower() == "neutral":
                        st.info(f"**Neutral.** 😐")
                    else:
                        st.warning(f"**Sentiment: {sentiment.capitalize()}** (Unknown category)")
                else:
                    st.error(f"Error from API: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
                    st.info("Please ensure the FastAPI backend is running.")

                # --- Alternative: Local Model Prediction (if you enabled Option 1 above) ---
                # if LOCAL_MODEL_LOADED:
                #     processed_text = preprocess_text(user_input)
                #     if not processed_text.strip():
                #         st.warning("Cannot analyze sentiment for empty or highly preprocessed text.")
                #     else:
                #         text_vectorized = local_vectorizer.transform([processed_text])
                #         prediction = local_model.predict(text_vectorized)[0]
                #         st.write("---")
                #         st.subheader("Predicted Sentiment (Local Model):")
                #         if prediction.lower() == "positive":
                #             st.success(f"**Positive!** 🎉")
                #         elif prediction.lower() == "negative":
                #             st.error(f"**Negative!** 😠")
                #         elif prediction.lower() == "neutral":
                #             st.info(f"**Neutral.** 😐")
                #         else:
                #             st.warning(f"**Sentiment: {prediction.capitalize()}** (Unknown category)")
                # else:
                #     st.warning("Local model not loaded. Please ensure the model training script has run successfully.")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure it is running (e.g., `uvicorn app.api:app --reload` from the project root).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("""
<style>
.stTextArea [data-baseweb="textarea"] {
    min-height: 150px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by a custom sentiment analysis model.")