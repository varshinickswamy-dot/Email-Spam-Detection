import math
from pathlib import Path

import joblib

from backend.email_features import prepare_model_frame


class SpamModel:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {path}. Run "
                "'python scripts/train_model.py' before starting the application."
            )
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError(f"Model artifact at {path} has an unsupported format.")
        self.pipeline = bundle["pipeline"]
        self.info = bundle.get("metadata", {})

    def predict(self, text):
        label = int(self.pipeline.predict([text])[0])
        classifier = self.pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            probabilities = self.pipeline.predict_proba([text])[0]
            class_positions = {int(value): index for index, value in enumerate(classifier.classes_)}
            spam_score = float(probabilities[class_positions[1]])
            confidence = float(probabilities[class_positions[label]])
        else:
            margin = float(self.pipeline.decision_function([text])[0])
            spam_score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, margin))))
            confidence = spam_score if label == 1 else 1.0 - spam_score

        return {
            "prediction": "spam" if label == 1 else "not_spam",
            "label": "spam" if label == 1 else "not_spam",
            "confidence": round(confidence, 4),
            "spam_score": round(spam_score, 4),
            "score_type": "uncalibrated_model_score",
            "explanation": (
                "The message resembles patterns learned from labeled spam examples."
                if label == 1
                else "The message resembles patterns learned from legitimate examples."
            ),
        }


class EmailSecurityModel:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Email model not found at {path}. Run "
                "'python scripts/train_email_model.py' before starting the application."
            )
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError(f"Email model artifact at {path} has an unsupported format.")
        self.pipeline = bundle["pipeline"]
        self.info = bundle.get("metadata", {})
        self.threshold = float(self.info.get("threshold", 0.5))

    def predict_spam_probability(self, record):
        features = prepare_model_frame([record])
        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError("The email model must provide calibrated class probabilities.")
        probabilities = self.pipeline.predict_proba(features)[0]
        classes = list(self.pipeline.classes_)
        if 1 not in classes:
            raise ValueError("The trained email model does not contain a spam class.")
        return float(probabilities[classes.index(1)])
