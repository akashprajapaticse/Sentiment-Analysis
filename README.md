# 📝 Product Review Sentiment Analysis

This project uses **machine learning** and **NLP techniques** to classify product reviews into three sentiment classes: **Positive**, **Negative**, and **Neutral**. It includes a well-structured Python backend and an interactive frontend built with **Streamlit**.

---

## 🔍 Overview

- **Text Preprocessing**: Lowercasing, punctuation removal, stopword removal, lemmatization
- **Feature Extraction**: TF-IDF Vectorization
- **Class Imbalance Handling**: SMOTE
- **Model**: Multinomial Naive Bayes
- **Optimization**: GridSearchCV
- **Deployment**: Streamlit-based frontend for real-time prediction

---

## ✨ Features

✅ Sentiment Prediction (`Positive`, `Negative`, `Neutral`)  
✅ Balanced training using SMOTE  
✅ Optimized hyperparameters  
✅ End-to-end pipeline  
✅ User-friendly web interface

---

## 📁 Project Structure

```

Sentiment-Analysis/
├── data/
│   ├── Equal.csv
│   ├── RATIO.csv
│   └── sentiment.csv
├── frontend/
│   └── streamlit_app.py        # Streamlit web app
├── models/
│   └── sentiment_pipeline.pkl  # Trained ML model
├── scripts/
│   └── train_model.py          # Model training script
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/akashprajapaticse/Sentiment-Analysis.git
cd Sentiment-Analysis
````

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ The first run will download necessary NLTK resources like stopwords and punkt automatically.

---

## 🏃 How to Use

### 🔧 Train the Model

Run the model training script:

```bash
python scripts/train_model.py
```

> This saves the trained pipeline as `models/sentiment_pipeline.pkl`.

### 🌐 Run the Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

Visit [http://localhost:8501](http://localhost:8501) to interact with the app.

---

## 📊 Model Performance

* **Accuracy**: \~63%
* Performs well for positive/negative, with room to improve "Neutral" prediction

---

## 🌱 Future Improvements

* 💡 Use Word2Vec, GloVe, or BERT embeddings
* 🧠 Integrate RNN/LSTM for deeper semantic understanding
* 🗂️ Add more diverse training data
* 🎯 Enhance precision for neutral class

---

## 📜 License

MIT License

---

## 🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 👤 Author

**Akash Prajapati**
🔗 [GitHub](https://github.com/akashprajapaticse)

```