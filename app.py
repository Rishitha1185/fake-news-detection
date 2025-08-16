import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

MODEL_PATH = "model.pkl"
DATA_PATH = "data/sample_news.csv"

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    # Clean minimal
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])]
    return df

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)),
        ("clf", LogisticRegression(max_iter=500)),
    ])

def train_and_save_model():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(pipe, MODEL_PATH)
    return acc

def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), None
    else:
        acc = train_and_save_model()
        return joblib.load(MODEL_PATH), acc

st.title("📰 Fake News Detector (Starter)")
st.write("Paste a news headline or short article below. The app uses TF‑IDF + Logistic Regression.")

model, first_train_acc = get_model()
if first_train_acc is not None:
    st.success(f"Model trained on sample data. Test accuracy: {first_train_acc:.2%}")

user_text = st.text_area("Enter text", height=180, placeholder="e.g., Government announces new policy to reduce emissions...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            proba = model.predict_proba([user_text])[0]
            # Assuming classes are in alphabetical order ['FAKE','REAL']
            classes = list(model.classes_)
            p_real = float(proba[classes.index('REAL')])
            label = "REAL" if p_real >= 0.5 else "FAKE"
            st.metric("Prediction", label)
            st.write(f"Confidence (REAL): {p_real:.2%}")

with col2:
    if st.button("Retrain on sample data"):
        acc = train_and_save_model()
        st.success(f"Retrained! New test accuracy on sample split: {acc:.2%}")
