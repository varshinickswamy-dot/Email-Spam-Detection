# Email Model Details

## Dataset

The email model uses the Apache SpamAssassin Public Corpus, parsed from RFC 5322/MIME files. Labels are standardized to `spam=1` and `legitimate=0`. The corpus provides no supervised phishing, promotional, or malicious-attachment classes. The UCI SMS Spam Collection and its existing model are kept separate.

The corpus inspection script extracts subject, sender, plain text, and HTML text while excluding attachments. Empty/corrupt messages are counted and skipped. Exact records and normalized duplicates are grouped; conflicting labels for an identical normalized group are removed. A normalized fingerprint also masks common variable numbers and URLs to reduce template leakage.

## Features

`backend/email_features.py` is shared by training and inference. It supplies:

- Subject word TF-IDF with unigrams/bigrams.
- Body word TF-IDF with unigrams/bigrams.
- Body character TF-IDF with character n-grams.
- Sender address/display-name character features.
- Structural features such as message/subject length, capitalization, punctuation, digit proportion, link count, sender presence, HTML presence, IP-host links, shorteners, attachment mentions, and currency symbols.
- Visible HTML text extraction and URL/domain extraction. Scripts and styles are excluded from visible text.

Fitted TF-IDF vocabularies and IDF weights are learned only from the respective training partition. At inference, the saved model pipeline applies the same fitted transforms.

## Split, class balancing, and model selection

The pipeline uses group-aware stratified splits. A realistic test split is reserved before balancing and is not used to select the classifier. Selection runs on training and validation data. Each candidate is trained both on original training labels and on a randomly undersampled training subset. Balancing is never applied to validation or test data.

The five candidates are:

| Classifier | Features |
|---|---|
| Multinomial Naive Bayes | Word, character, sender, and structural features |
| Logistic Regression | Word, character, sender, and structural features |
| Linear SVM | Word, character, sender, and structural features |
| Random Forest | Word, character, sender, and structural features |
| HistGradientBoosting | Structural features (dense representation) |

Model selection uses a spam F1/PR-AUC composite and penalizes validation false-positive rate. Full comparison results on validation and untouched test data are saved to `reports/email/email_evaluation_metrics.json` and `reports/email/email_model_comparison.csv`. The test comparison includes original and balanced training modes with their natural test prevalence.

## Calibration and reported metrics

The selected model is calibrated by sigmoid scaling on a separate unbalanced development subset. A second group-aware development holdout selects the operating threshold to maximize spam recall subject to precision >= 0.95 and false-positive rate <= 0.02; if no candidate threshold meets those constraints, the script uses a documented F1/precision/recall composite with an FPR penalty. Neither development holdout is part of the untouched test set. The final test set reports accuracy, spam precision/recall/F1, ROC-AUC, PR-AUC, false-positive and false-negative counts/rates, confusion matrix, classification report, and Brier score. The artifact metadata records model choice, split counts, balancing method, calibration method, threshold-selection method, threshold, cleaning counts, dataset distribution, and scikit-learn version.

The API identifies the returned `spam_probability` as a calibrated sigmoid probability when the trained email artifact is loaded. It is an estimate for the `spam` class, not the probability that a message is malicious or a guarantee of calibration on future data. The separate `risk_score` is a transparent rule-based score from 0 to 100 and is not a probability.

## Current corpus results

The generated report currently records 5,838 deduplicated emails (4,097 legitimate and 1,741 spam). The untouched test split contains 1,168 messages (820 legitimate, 348 spam). The deployed calibrated model is the original-imbalance Linear SVM with a 0.26 threshold. On that test split it achieved 98.29% accuracy, 96.33% spam precision, 97.99% spam recall, 97.15% spam F1, 99.87% ROC-AUC, 99.71% PR-AUC, and a 0.0129 Brier score. Its confusion matrix is TN=807, FP=13, FN=7, TP=341.

The threshold-development holdout (467 messages, separate from both model fitting and calibration) measured 98.54% precision, 97.12% recall, 0.61% false-positive rate, and 2 false positives among 328 legitimate messages. This development score guided threshold selection and is not the final test result.

For the Linear SVM candidate comparison on the same natural-prevalence test set, original-imbalance training measured 97.70% spam precision and recall with 8 false positives; train-only undersampling measured 96.33% precision, 97.99% recall, and 13 false positives. This illustrates the recall/false-alarm tradeoff; the selected deployed pipeline retains the original training distribution. These historical SpamAssassin results do not guarantee current-email performance.

## Thresholds and secondary labels

The 0.26 decision threshold is tuned only on a development holdout, not the test set. The detection layer labels scores within 0.12 of its configured decision boundary as `uncertain`. Other UI categories—`phishing_suspicious` and `promotional`—are supplementary rule-assisted categories, not trained classes from the email corpus. False alarms and missed detections remain possible.

## Artifacts and visualizations

Training creates:

- `models/email_security_model.joblib` — complete calibrated feature/model pipeline and metadata.
- `data/processed/email_train.csv`, `email_validation.csv`, and `email_test.csv` — grouped, unbalanced split records.
- `reports/email/email_evaluation_metrics.json` — model comparison, balancing effects, and final held-out metrics.
- `reports/email/email_model_comparison.csv` — compact per-model/per-mode metrics.
- `reports/email/figures/` — SVG dataset distribution, word frequency, algorithm comparison, test metrics, confusion matrix, ROC, and precision-recall plots.

The separate legacy SMS model remains `models/spam_classifier.joblib`; it must not be used for email inference.

## Reproducibility and limitations

Training uses fixed split and undersampling seeds. Results depend on the exact downloaded corpus and installed scikit-learn version; consult the generated report rather than assuming a target accuracy. SpamAssassin is historical, and external or temporal validation is needed before real-world use.
