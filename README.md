
# Fake News Detection (Starter Project)

A beginner-friendly, **ready-to-run** fake news detection app built with **Streamlit**, **scikit-learn**, and **TF‑IDF + Logistic Regression**.  
Works out of the box on a tiny sample dataset. You can later swap in a Kaggle dataset to improve accuracy.

## ✨ Features
- End-to-end text classifier (TF‑IDF ➜ Logistic Regression)
- Streamlit web app for quick demos
- Trains automatically on first run (uses `data/sample_news.csv`)
- Saves a `model.pkl` for faster subsequent runs
- Simple structure so you can learn and customize

## 🗂️ Project Structure
```
fake-news-detection-starter/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ sample_news.csv
```

## 🚀 Quickstart (Windows / macOS / Linux)
1) **Create a virtual environment & install packages**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

2) **Run the app**
```bash
streamlit run app.py
```
The app will auto-train on the included sample data the first time it runs, then save `model.pkl`.

3) **Try it**
Paste a news headline or short article and click **Predict**.

## 🔄 Train on your own dataset (optional)
- Replace `data/sample_news.csv` with your dataset having two columns: `text` and `label` (`REAL` or `FAKE`).
- Delete `model.pkl` and re-run the app. It will train again on the new data.

## 📦 How to upload to GitHub
1. Create a **new public repository** on GitHub (e.g. `fake-news-detection`).
2. Initialize and push:
```bash
git init
git add .
git commit -m "Initial commit: Fake News Detection (Streamlit + TF-IDF + Logistic Regression)"
git branch -M main
git remote add origin https://github.com/<your-username>/fake-news-detection.git
git push -u origin main
```
(If prompted, log in to GitHub in your terminal or set up a personal access token.)

## 📝 Notes
- This is a **starter** to showcase skills. For stronger results, use a larger dataset and consider models like Linear SVM or fine-tuned transformers.
- Keep your README updated with **results, screenshots, and what you learned**.

# fake-news-detection
fake news detection using Machine Learning and Streamlit

