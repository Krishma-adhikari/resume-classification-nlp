# Resume Classification using NLP and Machine Learning

## Overview

This project classifies resumes into different job categories using Natural Language Processing (NLP) and Machine Learning.

The project demonstrates an end-to-end NLP workflow including text preprocessing, TF-IDF feature extraction, model comparison, and final model selection.

---

## Dataset

The dataset contains resume text and their corresponding job categories.

Target categories include fields such as:

- INFORMATION-TECHNOLOGY
- ENGINEERING
- FINANCE
- HEALTHCARE
- AUTOMOBILE
- AVIATION
- and more

---

## Workflow

```text
+-------------------+
|  Resume Dataset   |
+-------------------+
          |
          v
+-------------------+
| Text Preprocessing|
+-------------------+
          |
          v
+-------------------+
| TF-IDF Vectorizer |
+-------------------+
          |
          v
+-------------------+
| ML Model Training |
+-------------------+
          |
          v
+-------------------+
| Model Evaluation  |
+-------------------+
          |
          v
+-------------------+
| Final Prediction  |
+-------------------+
```

---

## NLP Preprocessing

Applied techniques:

- Lowercasing
- Removing URLs, emails, numbers, and special characters
- Tokenization
- Removing stopwords
- Lemmatization
- Text normalization

---

## Machine Learning Models

The following models were trained and compared:

- K-Nearest Neighbors
- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| KNN | 0.4447 | 0.4516 |
| Multinomial Naive Bayes | 0.5493 | 0.5098 |
| Logistic Regression | 0.6398 | 0.6219 |
| LinearSVC | **0.7243** | **0.7181** |

LinearSVC achieved the best performance and was selected as the final model.

---


## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib

---



## Conclusion

This project demonstrates how traditional NLP techniques and machine learning algorithms can be used for resume classification.

LinearSVC with TF-IDF features provided the best performance among the tested models and was selected as the final classifier.