# sentiment_project/scripts/train_model.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
DATA_DIR = os.path.join('..', 'data')
MODEL_DIR = os.path.join('..', 'models')

DATA_FILE_PATHS = [
    os.path.join(DATA_DIR, 'sentiment.csv'),
    os.path.join(DATA_DIR, 'Equal.csv'),
    os.path.join(DATA_DIR, 'RATIO.csv')
]

MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'sentiment_pipeline.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- NLTK Downloads ---
print("Downloading NLTK data (if not already present)...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' not found, downloading...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK 'wordnet' not found, downloading...")
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found, downloading...")
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' not found, downloading...")
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("NLTK 'omw-1.4' not found, downloading...")
    nltk.download('omw-1.4')
print("NLTK downloads complete.")

# --- Preprocessing Function ---
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    try:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = nltk.word_tokenize(text)
        stopwords_set = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopwords_set]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
    except Exception as e:
        # Added file_path to error message for better debugging if issues arise
        print(f"ERROR during preprocessing of text: '{str(text)[:100]}...' Error: {e}")
        return ""

def train_and_save_model():
    all_dfs = []
    CSV_ENCODING = 'latin1'

    for file_path in DATA_FILE_PATHS:
        print(f"\nAttempting to load data from: {os.path.abspath(file_path)}")
        try:
            df_temp = pd.read_csv(file_path, encoding=CSV_ENCODING, low_memory=False)

            TEXT_COLUMN_FOR_THIS_FILE = 'Review'
            RATE_COLUMN_FOR_THIS_FILE = 'Rate'

            if TEXT_COLUMN_FOR_THIS_FILE not in df_temp.columns:
                print(f"WARNING: '{file_path}' does not contain expected text column '{TEXT_COLUMN_FOR_THIS_FILE}'. Skipping this file.")
                continue
            if RATE_COLUMN_FOR_THIS_FILE not in df_temp.columns:
                print(f"WARNING: '{file_path}' does not contain expected rate column '{RATE_COLUMN_FOR_THIS_FILE}'. Skipping this file.")
                continue

            df_processed = df_temp[[TEXT_COLUMN_FOR_THIS_FILE, RATE_COLUMN_FOR_THIS_FILE]].copy()
            df_processed = df_processed.rename(columns={
                TEXT_COLUMN_FOR_THIS_FILE: 'review_text',
                RATE_COLUMN_FOR_THIS_FILE: 'sentiment_label_raw'
            })

            rate_to_sentiment_mapping = {
                1: 'negative',
                2: 'negative',
                3: 'neutral',
                4: 'positive',
                5: 'positive'
            }

            df_processed['sentiment_label_raw'] = pd.to_numeric(df_processed['sentiment_label_raw'], errors='coerce')
            df_processed['sentiment_label'] = df_processed['sentiment_label_raw'].map(rate_to_sentiment_mapping)

            initial_label_count = len(df_processed)
            df_processed = df_processed[df_processed['sentiment_label'].isin(['positive', 'negative', 'neutral'])]
            print(f"Removed {initial_label_count - len(df_processed)} rows with unhandled/NaN sentiment labels (from Rate column) from {file_path}.")

            all_dfs.append(df_processed)

        except FileNotFoundError as e:
            print(f"ERROR: {file_path} not found. Skipping. {e}")
        except UnicodeDecodeError as e:
            print(f"ERROR: Encoding issue with {file_path}. Try changing 'CSV_ENCODING'. Skipping. {e}")
        except KeyError as e:
            print(f"ERROR: Column name issue with {file_path}. Missing: '{e}'. Please verify TEXT_COLUMN_FOR_THIS_FILE/RATE_COLUMN_FOR_THIS_FILE. Skipping. {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred with {file_path}. Skipping. {e}")

    if not all_dfs:
        print("ERROR: No valid data files loaded. Cannot train model.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n--- Combined dataset size before final cleaning: {len(df)} ---")

    initial_rows = len(df)
    df.dropna(subset=['review_text'], inplace=True)
    df = df[df['review_text'].astype(str).str.strip() != '']
    rows_after_nan_empty_check = len(df)
    print(f"Removed {initial_rows - rows_after_nan_empty_check} rows with empty/NaN 'review_text' after initial load.")

    initial_rows_before_dedupe = len(df)
    df.drop_duplicates(subset=['review_text', 'sentiment_label'], inplace=True)
    rows_after_dedupe = len(df)
    print(f"Removed {initial_rows_before_dedupe - rows_after_dedupe} duplicate (review_text, sentiment_label) pairs.")

    print(f"\nFinal dataset size after initial cleaning and deduplication: {len(df)}.")
    print(f"Number of unique 'review_text' after deduplication: {df['review_text'].nunique()}")
    print(f"Number of unique 'sentiment_label' after deduplication: {df['sentiment_label'].nunique()}")

    print("\nData loaded and cleaned. Starting preprocessing of review text...")
    df['cleaned_review'] = df['review_text'].apply(preprocess_text)

    initial_rows_post_clean_apply = len(df)
    df.dropna(subset=['cleaned_review', 'sentiment_label'], inplace=True)
    df = df[df['cleaned_review'].str.strip() != '']
    rows_after_final_cleaning = len(df)
    print(f"Removed {initial_rows_post_clean_apply - rows_after_final_cleaning} rows due to empty/NaN reviews or missing labels after deep cleaning.")


    if df.empty:
        print("ERROR: No data left after cleaning. Cannot train model.")
        sys.exit(1)

    print("Preprocessing complete. Starting feature extraction (TF-IDF)...")
    X = df['cleaned_review']
    y = df['sentiment_label']

    print(f"Sentiment label distribution used for training:\n{y.value_counts()}")

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Starting Hyperparameter Tuning (GridSearchCV) ---")
    pipeline_steps = [
        ('tfidf', TfidfVectorizer()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', MultinomialNB())
    ]
    model_pipeline = Pipeline(pipeline_steps)

    # REVISED param_grid for more extensive search
    param_grid = {
        'tfidf__max_features': [10000, 20000, 30000, 40000, None], # Expanded max_features, None means no limit
        'tfidf__min_df': [1, 2, 3, 5], # Slightly more options for min_df
        'tfidf__ngram_range': [(1,1), (1,2), (1,3)], # Added (1,3) n-grams
        'tfidf__use_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'smote__sampling_strategy': ['auto', 0.6, 0.7, 0.8, 0.9, 1.0], # More granular oversampling ratios
        'classifier__alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0] # More granular alpha values, including smaller ones
    }

    # Increased CV folds for more robust evaluation - This will make it significantly slower!
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)

    print("Fitting GridSearchCV (this may take a while, especially with more parameters and CV folds)...")
    grid_search.fit(X, y)

    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation Macro F1-score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    print("\nFeature extraction complete. Splitting data into training and test sets for final evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples.")
    print(f"Train set sentiment distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set sentiment distribution:\n{y_test.value_counts(normalize=True)}")

    print("Training the BEST model pipeline from GridSearchCV on the full training set (with SMOTE applied inside the pipeline)...")
    best_model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- Model Evaluation (Best Model) ---")
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"\nSaving best model pipeline to {MODEL_PIPELINE_PATH}...")
    with open(MODEL_PIPELINE_PATH, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print("Model pipeline saved successfully!")
    print("This single file ('sentiment_pipeline.pkl') now contains the TF-IDF vectorizer, SMOTE, and the Multinomial Naive Bayes model with optimized hyperparameters.")

if __name__ == "__main__":
    train_and_save_model()