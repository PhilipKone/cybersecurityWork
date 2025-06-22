# Model Results Log

This file tracks major experiments, model configurations, and results for the adaptive phishing detection project.

---

## Experiment: HTML Features Only (XGBoost)
- **Date:** 2025-04
- **Features Used:** HTML features (TF-IDF, hyperlink statistics)
- **Model:** XGBoost Classifier (default parameters)
- **Results:**
  - **Accuracy:** ~0.54
  - **ROC-AUC:** Not reported (performance was suboptimal)
- **Notes:**
  - Model performed poorly using only HTML features, indicating these features alone are not sufficient for robust phishing detection.

---

## Experiment: URL Features Only (XGBoost)
- **Date:** 2025-04
- **Features Used:** URL features (statistical, character-level)
- **Model:** XGBoost Classifier (default parameters)
- **Results:**
  - **Accuracy:** ~0.81
  - **ROC-AUC:** ~0.88
- **Notes:**
  - Model performance improved significantly with URL features, achieving strong accuracy and ROC-AUC.

---

## Experiment: Combined URL + HTML Features (XGBoost)
- **Date:** 2025-04-27
- **Features Used:** URL features (statistical, character-level), HTML features (TF-IDF, hyperlink statistics)
- **Model:** XGBoost Classifier (default parameters)
- **Train/Test Split:** As per project CSVs

### Results
- **Accuracy:** 0.80
- **ROC-AUC:** 0.88
- **Confusion Matrix:**
  - [[5968  627]
     [1752 3704]]
- **Classification Report:**
  - Class 0 (Benign): Precision: 0.77, Recall: 0.90, F1: 0.83
  - Class 1 (Phishing): Precision: 0.86, Recall: 0.68, F1: 0.76
  - Macro avg: Precision: 0.81, Recall: 0.79, F1: 0.80
  - Weighted avg: Precision: 0.81, Recall: 0.80, F1: 0.80

### Notes
- Combined features outperform HTML-only, but are similar to URL-only results.
- Feature importance analysis available in `feature_importance_report.md` and `README_feature_importance.md`.
- See README for next steps and further experiment plans.

---

## Experiment: Deep Learning on Raw .txt Content
- **Date:** 2025-06-22
- **Features Used:** Raw text content from .txt files (cleaned, tokenized, padded)
- **Model:** Simple Embedding + GlobalMaxPooling1D + Dense (Keras Sequential)
- **Train/Test Split:** 80/20 random split

### Results
- **Test Accuracy:** 0.859
- **Precision/Recall/F1 (macro avg):** 0.86 / 0.86 / 0.86
- **ROC-AUC:** 0.857

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.86      0.88      0.87      6572
           1       0.85      0.83      0.84      5479

    accuracy                           0.86     12051
   macro avg       0.86      0.86      0.86     12051
weighted avg       0.86      0.86      0.86     12051
```

### Notes
- This is the first deep learning experiment using raw text content instead of engineered features.
- The model achieves strong baseline performance, comparable to classical ML using URL features.
- Next steps: try more advanced architectures (LSTM, CNN), combine URL and text, and perform error analysis.

---

_Add new experiments below this line as the project progresses._
