# 🛡️ SpamShield: NLP-Powered Email & SMS Spam Classification Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-3776AB.svg)](https://www.nltk.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.21%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

> A high-throughput Natural Language Processing (NLP) text classification engine engineered to detect, filter, and neutralize spam and phishing messages with **98.21% accuracy** and near-zero false alarms.

---

## 📌 Project Overview

Unsolicited bulk email and malicious SMS messages pose major cybersecurity, productivity, and privacy threats. In modern production environments, effective spam filtering requires an asymmetric optimization:
1. **High Spam Recall**: Catching phishing links, scams, and deceptive advertisements.
2. **Minimal False Positives**: Preventing legitimate communications (**ham**) from ever being misclassified or discarded.

**SpamShield** demonstrates a full natural language processing lifecycle—from raw textual message ingestion, regex-driven normalization, and Porter stemming to Bag-of-Words vectorization and multi-model machine learning benchmarking.

---

## 🏗️ NLP Processing Pipeline & System Architecture

```
┌─────────────────────────────────┐
│ Raw Email / SMS Corpus Ingestion│ (5,572 records: Ham & Spam labels)
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ Text Sanitation & Normalization │ • Regex filtering: stripping non-alphabetic tokens
└───────────────┬─────────────────┘ • Lowercase normalization
                │
                ▼
┌─────────────────────────────────┐
│ Linguistic Preprocessing (NLTK) │ • Stopword suppression (English stopword lexicon)
└───────────────┬─────────────────┘ • Porter Stemming algorithm (inflectional reduction)
                │
                ▼
┌─────────────────────────────────┐
│ Vector Space Embedding          │ • CountVectorizer (Bag-of-Words)
└───────────────┬─────────────────┘ • High-frequency vocabulary cap: 4,000 features
                │
                ▼
┌─────────────────────────────────┐
│ Supervised Learning Benchmark   │ • Multinomial Naïve Bayes (Optimal for text)
└───────────────┬─────────────────┘ • Random Forest Ensemble
                │                   • Decision Tree Classifier
                ▼
┌─────────────────────────────────┐
│ Evaluation & Model Serialization│ • Confusion Matrix, Precision, Recall, F1
└─────────────────────────────────┘ • Exported artifacts: MNB.pkl, RFC.pkl, DTC.pkl
```

---

## 📁 Repository Structure

```plaintext
Email-Spam-Detection/
│
├── DTC.pkl                      # Serialized Decision Tree Classifier artifact
├── MNB.pkl                      # Serialized Multinomial Naïve Bayes artifact (Champion)
├── RFC.pkl                      # Serialized Random Forest Classifier artifact
├── EMAIL_SPAM_DETECTION.ipynb   # Interactive analysis and experimentation notebook
├── email_spam_detection.py      # Standalone training and pipeline execution script
├── LICENSE                      # GNU General Public License v3
└── README.md                    # Project documentation and quickstart guide
```

---

## 📊 Empirical Model Benchmark & Results

Three distinct classification paradigms were evaluated on an unseen test partition of **1,115 messages** (965 legitimate ham, 150 spam).

| Classifier Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | False Positives (Ham as Spam) | False Negatives (Spam missed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naïve Bayes (MNB)** 🏆 | **98.21%** | **0.93** | **0.93** | **0.93** | 10 | 10 |
| **Decision Tree Classifier (DTC)** | 97.49% | 0.96 | 0.85 | 0.90 | 6 | 22 |
| **Random Forest Classifier (RFC)** | 97.31% | **0.99** | 0.81 | 0.89 | **1** | 29 |

### 🔍 Architectural Insights:
* **Multinomial Naïve Bayes (Top Champion)**: Delivers the highest overall accuracy (**98.21%**) and an exceptional harmonic balance between precision and recall (**0.93 / 0.93**). Because text feature spaces with word frequencies fit multinomial event models naturally, MNB provides instant scoring with minimal memory overhead.
* **Random Forest Classifier (Precision Champion)**: Exhibited an outstanding **99% precision** on spam, recording **only a single false positive** across 965 legitimate test messages. RFC is the preferred estimator for mission-critical deployments where incorrectly flagging a legitimate email is severely penalized.

---

## 💾 Pre-Trained Model Artifacts

Pre-trained weights and estimators are serialized in `.pkl` format for direct production deployment:

* `MNB.pkl` — Multinomial Naïve Bayes model (balanced high performance).
* `RFC.pkl` — Random Forest model (ultra-conservative false-positive rate).
* `DTC.pkl` — Decision Tree model (lightweight deterministic baseline).

---

## ⚡ Quickstart & Usage

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Adityamhalunge-Dev/Email-Spam-Detection.git
cd Email-Spam-Detection

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### 2. Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

### 3. Real-Time Inference Example

To test a new incoming message with the trained model:

```python
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Load trained model
with open('MNB.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Text preprocessing function
def clean_message(raw_text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    cleaned = re.sub('[^a-zA-Z]', ' ', raw_text).lower().split()
    stemmed = [ps.stem(word) for word in cleaned if word not in stop_words]
    return ' '.join(stemmed)

# 3. Predict sample message
sample_email = "URGENT! You have won a $1,000 Walmart Gift Card. Click here to claim your reward immediately!"
processed_text = clean_message(sample_email)

# Note: Pass through the fitted CountVectorizer before model prediction:
# vectorized_text = cv.transform([processed_text]).toarray()
# prediction = model.predict(vectorized_text)
# print("Spam" if prediction[0] == 1 else "Ham")
```

---

## 🛠️ Technology Stack

| Domain | Technology / Library |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Text Preprocessing** | NLTK (Stopwords, PorterStemmer), Regular Expressions (`re`) |
| **Feature Extraction** | Scikit-Learn `CountVectorizer` (Bag-of-Words, 4,000 features) |
| **Machine Learning** | Scikit-Learn (`MultinomialNB`, `RandomForestClassifier`, `DecisionTreeClassifier`) |
| **Metrics & Analytics** | Confusion Matrix, Classification Report, Precision-Recall, Seaborn, Matplotlib |
| **Serialization** | Pickle |

---

## 🚀 Future Roadmap

- [ ] **TF-IDF & N-Gram Integration**: Transition from raw term counts to TF-IDF with bi-gram and tri-gram vocabulary ranges.
- [ ] **Transformer Fine-Tuning**: Benchmark with lightweight LLM backbones such as `DistilBERT` or `RoBERTa` for context-aware spam detection.
- [ ] **Interactive Web Interface**: Deploy a live demo using `Streamlit` or `Gradio` with real-time confidence scores and spam keyword heatmaps.
- [ ] **REST API Service**: Containerize an inference microservice using `FastAPI` and `Docker`.

---

## 📜 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
