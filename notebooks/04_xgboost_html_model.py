import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# --- Load Features and Labels ---
X_train_url = pd.read_csv('notebooks/X_train_url_features.csv')
X_test_url = pd.read_csv('notebooks/X_test_url_features.csv')
X_train_html = pd.read_csv('notebooks/X_train_html_features.csv')
X_test_html = pd.read_csv('notebooks/X_test_html_features.csv')
train_df = pd.read_csv('notebooks/y_train.csv')
test_df = pd.read_csv('notebooks/y_test.csv')

# --- Prepare Labels ---
y_train = train_df.iloc[:,0].reset_index(drop=True)
y_test = test_df.iloc[:,0].reset_index(drop=True)

# If your labels are -1/1 and you want 0/1, uncomment below:
y_train = y_train.map({-1: 0, 1: 1})
y_test = y_test.map({-1: 0, 1: 1})

# --- Combine URL and HTML features ---
X_train_combined = pd.concat([X_train_url, X_train_html], axis=1)
X_test_combined = pd.concat([X_test_url, X_test_html], axis=1)

# --- Train XGBoost Classifier ---
dtrain = xgb.DMatrix(X_train_combined, label=y_train)
dtest = xgb.DMatrix(X_test_combined, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42,
    'verbosity': 1
}

bst = xgb.train(params, dtrain, num_boost_round=100)

y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
except Exception as e:
    print("ROC-AUC could not be calculated:", e)

# --- Feature Importance Plot ---
ax = xgb.plot_importance(bst, max_num_features=15)
plt.tight_layout()
plt.savefig('notebooks/feature_importance_combined.png')
plt.show()

# --- Save Top Features to Markdown Report ---
import numpy as np
feature_scores = bst.get_score(importance_type='weight')
# Get feature names for combined features
feature_names = list(X_train_combined.columns)
# Map XGBoost's f0, f1, ... to actual names
score_items = [(k if k in feature_names else feature_names[int(k[1:])], v) for k, v in feature_scores.items()]
sorted_scores = sorted(score_items, key=lambda x: x[1], reverse=True)[:15]

with open('notebooks/feature_importance_report.md', 'w', encoding='utf-8') as f:
    f.write('# XGBoost Feature Importance Report\n\n')
    f.write('![Feature Importance Plot](feature_importance_combined.png)\n\n')
    f.write('## Top 15 Features (Combined URL + HTML)\n\n')
    f.write('| Rank | Feature Name | Importance Score |\n')
    f.write('|------|--------------|------------------|\n')
    for i, (name, score) in enumerate(sorted_scores, 1):
        f.write(f'| {i} | {name} | {score} |\n')
    f.write('\n')
    f.write('## Interpretation\n')
    f.write('The table and plot above show which features contributed most to the XGBoost model for phishing detection.\n')
    f.write('Features at the top are the most influential for the model\'s predictions.\n')
    f.write('URL features (e.g., suspicious keywords, length, special characters) often dominate, but HTML features may also appear if they provide additional signal.\n')
    f.write('Reviewing these features can guide further feature engineering and help explain the model\'s behavior in your report.\n')

