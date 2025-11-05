# severity_model.py
"""
A small utility to train/predict severity labels.
This is optional: you can use LLM's severity or combine both.
We provide a simple TF-IDF + RandomForest pipeline and a small synthetic trainer.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import List

MODEL_PATH = "severity_model.joblib"


def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    return pipe


def train_from_csv(csv_path: str, text_col: str = "description", label_col: str = "severity"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()
    pipe = build_pipeline()
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict_texts(texts: List[str]):
    model = load_model()
    if model is None:
        return ["Medium"] * len(texts)
    return model.predict(texts).tolist()
