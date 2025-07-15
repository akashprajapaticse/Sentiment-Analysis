# scripts/train_model.py

import os
import pickle
import re
import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ─── 1) Fix your paths so MODEL_DIR is always the same folder the app loads from ─────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))  # project root
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, "sentiment_pipeline.pkl")


# ─── 2) Data loading & basic cleaning ─────────────────────────────────────────────
def load_and_clean():
    df = pd.read_csv(os.path.join(DATA_DIR, "sentiment.csv"), encoding="latin1")
    # rename your columns appropriately:
    df = df.rename(columns={"Review": "text", "Rate": "rating"})
    # map ratings → labels
    mapping = {1:"negative", 2:"negative", 3:"neutral", 4:"positive", 5:"positive"}
    df["label"] = df["rating"].map(mapping)
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(["positive","negative","neutral"])]
    return df[["text","label"]]


# ─── 3) Preprocess exactly like your app (or better: use regex tokenization) ───────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # simple regex tokenizer instead of word_tokenize
    tokens = re.findall(r"\b\w+\b", text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw]
    lemma = WordNetLemmatizer()
    return " ".join(lemma.lemmatize(t) for t in tokens)


# ─── 4) Main training routine ─────────────────────────────────────────────────────
def train_and_save():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    df = load_and_clean()
    df["cleaned"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # build your pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    MultinomialNB()),
    ])

    # grid on TRAIN only
    param_grid = {
        "tfidf__max_features": [10000, 20000, None],
        "tfidf__ngram_range":  [(1,1),(1,2)],
        "clf__alpha":          [0.1, 1.0],
    }
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, refit=True, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Train‑CV f1_macro:", gs.best_score_)

    best = gs.best_estimator_

    # final evaluation on hold‑out
    y_pred = best.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # *** Crucial: this is a FULLY fitted pipeline ***
    with open(MODEL_PIPELINE_PATH, "wb") as f:
        pickle.dump(best, f)
    print(f"✅ Saved fitted pipeline to {MODEL_PIPELINE_PATH}")


if __name__ == "__main__":
    train_and_save()
