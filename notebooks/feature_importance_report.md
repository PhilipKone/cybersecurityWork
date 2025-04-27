# XGBoost Feature Importance Report

![Feature Importance Plot](feature_importance_combined.png)

## Top 15 Features (Combined URL + HTML)

| Rank | Feature Name | Importance Score |
|------|--------------|------------------|
| 1 | url_length | 407.0 |
| 2 | digit_ratio | 325.0 |
| 3 | letter_ratio | 305.0 |
| 4 | url_entropy | 284.0 |
| 5 | special_char_count | 239.0 |
| 6 | subdomain_count | 69.0 |
| 7 | num_scripts | 39.0 |
| 8 | num_links | 35.0 |
| 9 | num_imgs | 32.0 |
| 10 | num_external_links | 30.0 |
| 11 | num_internal_links | 27.0 |
| 12 | tfidf_255 | 19.0 |
| 13 | tfidf_264 | 19.0 |
| 14 | tfidf_13 | 15.0 |
| 15 | has_ip | 14.0 |

## Interpretation
The table and plot above show which features contributed most to the XGBoost model for phishing detection.
Features at the top are the most influential for the model's predictions.
URL features (e.g., suspicious keywords, length, special characters) often dominate, but HTML features may also appear if they provide additional signal.
Reviewing these features can guide further feature engineering and help explain the model's behavior in your report.
