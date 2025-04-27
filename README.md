# cybersecurityWork

This repository contains research and development for adaptive phishing detection using machine learning, as part of the PHconsult project.

## Project Overview
- **Goal:** Develop a robust, adaptive phishing detection model using advanced feature engineering and machine learning, building on and extending the state-of-the-art (Aljofey et al., 2022).
- **Current Focus:** Feature engineering from URL and HTML, XGBoost modeling, and interpretability.

## Progress So Far
- Implemented extraction of URL and HTML features (including TF-IDF and hyperlink statistics).
- Combined features for training an XGBoost classifier.
- Evaluated model performance (accuracy, ROC-AUC, confusion matrix, classification report).
- Generated and documented feature importance plots and markdown reports.
- Ensured all data files and sensitive research materials are excluded from version control.

## Next Steps (Based on Base Paper, Proposal, and Current Progress)

1. **Advanced Feature Engineering**
   - Review and expand hyperlink and login form features (ensure all 13+ HTML-based features from the base paper are included).
   - Add or refine character-level URL features (fixed-length, padded/truncated as in the base paper).
   - Enhance TF-IDF/text features with advanced NLP preprocessing (tokenization, lemmatization, stemming, cleaning JS/CSS).
   - (Optional for novelty) Engineer new features relating URL and HTML content or behavioral/user feedback features.

2. **Feature Selection & Dimensionality Reduction**
   - Apply and compare feature selection techniques (recursive elimination, genetic algorithms, PCA) to optimize the feature set.

3. **Model Evaluation & Experimentation**
   - Perform ablation studies: compare URL-only, HTML-only, and combined feature sets.
   - Benchmark against other classifiers (e.g., Random Forest, Logistic Regression, SVM).
   - Conduct cross-validation for robust performance metrics.

4. **User Feedback & Adaptation (Novelty)**
   - (If aligned with thesis) Simulate or integrate a feedback loop to adapt the model to new phishing tactics.

5. **Deployment & Real-Time Evaluation**
   - Prepare for deployment in a simulated real-time environment.
   - Measure system scalability, latency, and adaptability.

6. **Reporting & Documentation**
   - Continue documenting methodology, results, and insights in markdown and thesis files.
   - Compare results with the base paper and highlight novel contributions.

---

**For details on feature importance and interpretability, see:**
- `notebooks/feature_importance_report.md`
- `notebooks/README_feature_importance.md`

**For research context and thesis alignment, see:**
- `researchPP.md`

---

_This README will be updated as the project progresses. For any questions or contributions, please refer to the documentation or contact the project maintainer._
