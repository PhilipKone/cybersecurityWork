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
xgb.plot_importance(bst, max_num_features=15)
plt.show()
