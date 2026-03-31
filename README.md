# Amazon Product Review — Sentiment Analysis

A machine learning web app that classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** using NLP preprocessing and ensemble models, served via a Streamlit interface.

## Overview

This project explores the Amazon reviews dataset, builds and compares multiple classification models (Random Forest, XGBoost, Decision Tree), and deploys the best-performing model as an interactive web application where users can type any review and get an instant sentiment prediction.

## Features

- End-to-end NLP pipeline: text cleaning, stopword removal, stemming, and vectorization
- Exploratory Data Analysis with visualizations (word clouds, distribution plots, confusion matrices)
- Model training and evaluation using Random Forest, Decision Tree, and XGBoost
- Hyperparameter tuning with GridSearchCV and StratifiedKFold cross-validation
- Trained models serialized with `pickle` for fast inference
- Interactive Streamlit app for real-time sentiment prediction

## Project Structure

```
├── app.py                              # Streamlit web application
├── sa.ipynb                            # Core sentiment analysis notebook
├── Data Exploration & Modelling.ipynb  # EDA and model training notebook
├── modified_reviews.csv                # Preprocessed dataset
├── vectorizer.pkl                      # Saved CountVectorizer
├── scaler.pkl                          # Saved MinMaxScaler
├── model_rf.pkl                        # Saved Random Forest model
├── model_xgb.pkl                       # Saved XGBoost model
└── requirements.txt                    # Python dependencies
```

## Tech Stack

| Category | Libraries |
|---|---|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud` |
| NLP | `nltk` (PorterStemmer, stopwords), `scikit-learn` (CountVectorizer) |
| ML Models | `scikit-learn` (Random Forest, Decision Tree), `xgboost` |
| Deployment | `streamlit` |
| Deep Learning (optional) | `keras`, `tensorflow` |

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/akavision/AMAZON-PRODUCT-Review-Sentiment-Analysis.git
cd AMAZON-PRODUCT-Review-Sentiment-Analysis
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

> **Note:** Update the file paths in `app.py` to use relative paths before running locally:
> ```python
> with open("vectorizer.pkl", "rb") as f:
>     vectorizer = pickle.load(f)
> ```

## How It Works

1. The user enters a product review in the text box.
2. The app preprocesses the text — removes special characters, lowercases, removes stopwords, and applies Porter Stemming.
3. The cleaned text is vectorized using the saved `CountVectorizer` and scaled with `MinMaxScaler`.
4. The Random Forest model predicts the sentiment class.
5. The result is displayed as **Positive 😊**, **Neutral 😐**, or **Negative 😞**.

## Model Performance

Models were evaluated using accuracy score, confusion matrix, and cross-validation on the Amazon Alexa reviews dataset. Random Forest was selected as the final deployed model based on overall performance.
