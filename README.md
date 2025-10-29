# AI-Powered Task Management System
**Internship Project – Zaalima Development Pvt. Ltd.**

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify technical task descriptions into categories like **Backend**, **Frontend**, and **Data Science**.

The model is built, trained, optimized, and deployed over four weeks using Python, Scikit-learn, and Streamlit.

---

## Project Timeline

| Week | Focus Area | Key Tasks |
|------|-------------|-----------|
| **Week 1** | Data Understanding & NLP Preprocessing | Data cleaning, text normalization, stopword removal, stemming |
| **Week 2** | Feature Extraction & Model Training | TF-IDF vectorization, training Naive Bayes and SVM models |
| **Week 3** | Model Optimization | Cross-validation, hyperparameter tuning, and performance comparison |
| **Week 4** | Final Evaluation & Deployment | Evaluation, visualization, and Streamlit app deployment |

---

## Tech Stack

- **Python 3.x**
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK
- **NLP Techniques:** Tokenization, Stopword Removal, Stemming, TF-IDF
- **ML Models:** Multinomial Naive Bayes, Support Vector Machine (SVM)
- **Deployment Tool:** Streamlit

---

## Files in this Repository

| File | Description |
|------|--------------|
| `AI_Powered_Task_Management_System.ipynb` | Main Google Colab notebook (Weeks 1 → 4) |
| `cleaned_task_dataset.csv` | Preprocessed dataset used for training |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `svm_model_tuned.pkl` | Final tuned SVM model |
| `naive_bayes_model.pkl` | Baseline Naive Bayes model |
| `app.py` | Streamlit deployment script |
| `README.md` | Project documentation file |

---

## How to Run This Project

### Option 1 — In Google Colab
1. Open the `.ipynb` file in Google Colab.  
2. Run all cells sequentially (Weeks 1–4).  
3. It will generate the models and deployment files automatically.

### Option 2 — Run Streamlit App Locally
```bash
pip install streamlit
streamlit run app.py
