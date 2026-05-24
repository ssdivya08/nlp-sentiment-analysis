# NLP Sentiment Analysis using Machine Learning

## 📌 Project Overview
This project implements an NLP based machine learning pipeline to classify customer reviews into Positive or Negative sentiments using TF IDF vectorization and Logistic Regression.

The system processes textual customer reviews and predicts sentiment automatically using Natural Language Processing techniques.

---

## 🎯 Objective
- Automate sentiment classification of customer reviews
- Build an end to end NLP pipeline
- Perform text preprocessing and feature extraction
- Achieve accurate sentiment prediction

---

## 🧠 Technologies Used
- Python
- Scikit learn
- Pandas
- NumPy
- TF IDF Vectorization
- Logistic Regression
- NLP

---

## ⚙️ Project Structure

```text
data/
    reviews.csv

notebook/
    nlp_review_classification.ipynb

output/
    results.txt

src/
    nlp_model.py
```
🚀 How It Works

Load customer review dataset

Perform text preprocessing

Convert text into numerical vectors using TF IDF

Train Logistic Regression model

Predict Positive or Negative sentiment

Evaluate model accuracy

📊 Model Performance

Algorithm: Logistic Regression

Feature Extraction: TF IDF

Accuracy: Approximately 92 percent

▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Run notebook:

jupyter notebook notebook/nlp_review_classification.ipynb

Or run Python script:

python src/nlp_model.py

🔮 Future Improvements

Add deep learning based NLP models

Build Streamlit web interface

Add real time review prediction

Support multilingual sentiment analysis

📜 License

This project is developed for educational and learning purposes.
