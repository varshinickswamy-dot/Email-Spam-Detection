import argparse
from collections import Counter
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.email_features import (  # noqa: E402
    build_feature_transformer,
    normalize_text,
    prepare_model_frame,
    structural_features,
)

DEFAULT_DATASET = ROOT_DIR / "data" / "raw" / "spamassassin" / "spamassassin_emails.jsonl"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_MODEL = ROOT_DIR / "models" / "email_security_model.joblib"
DEFAULT_REPORTS = ROOT_DIR / "reports" / "email"
RANDOM_STATE = 42


def _record_group(record):
    fingerprint_text = normalize_text(
        f"{record.get('subject', '')} {record.get('body', '')}"
    )
    fingerprint_text = re.sub(r"\b\d+\b", " number ", fingerprint_text)
    fingerprint_text = re.sub(r"https?://\S+", " url ", fingerprint_text)
    return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()


def load_email_dataset(path):
    records = []
    skipped = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                label = str(record.get("label", "")).strip().lower()
                subject = record.get("subject", "")
                body = record.get("body", "")
                sender = record.get("sender", "")
                if label not in {"spam", "legitimate"}:
                    skipped += 1
                    continue
                if not all(isinstance(value, str) for value in (subject, body, sender)):
                    skipped += 1
                    continue
                if not subject.strip() and not body.strip():
                    skipped += 1
                    continue
                normalized = {
                    "subject": subject[:500],
                    "body": body[:50_000],
                    "html_body": str(record.get("html_body", "") or "")[:50_000],
                    "sender": sender[:320],
                    "links": record.get("links", []),
                    "label": label,
                    "source_id": str(record.get("source_id", "")),
                }
                normalized["target"] = int(label == "spam")
                normalized["group"] = _record_group(normalized)
                records.append(normalized)
            except (json.JSONDecodeError, TypeError, ValueError):
                skipped += 1

    frame = pd.DataFrame(records)
    if frame.empty or frame["target"].nunique() != 2:
        raise ValueError("The email dataset must contain spam and legitimate examples.")

    conflicting = (
        frame.groupby("group")["target"].nunique().loc[lambda values: values > 1].index
    )
    conflict_count = int(frame["group"].isin(conflicting).sum())
    frame = frame[~frame["group"].isin(conflicting)].copy()
    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["group", "target"]).reset_index(drop=True)
    return frame, {
        "invalid_or_corrupt_rows_skipped": skipped,
        "conflicting_duplicate_rows_removed": conflict_count,
        "duplicate_rows_removed": before_dedup - len(frame),
    }


def _split_indices(frame):
    y = frame["target"].to_numpy()
    groups = frame["group"].to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    development_idx, test_idx = next(outer.split(frame, y, groups))
    development = frame.iloc[development_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
    train_relative, validation_relative = next(
        inner.split(
            development,
            development["target"].to_numpy(),
            development["group"].to_numpy(),
        )
    )
    selection_train_idx = development_idx[train_relative]
    validation_idx = development_idx[validation_relative]
    return selection_train_idx, validation_idx, development_idx, test_idx


def _save_splits(frame, processed_dir, train_idx, validation_idx, test_idx):
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = ["subject", "body", "html_body", "sender", "label", "group", "source_id"]
    for name, indices in (
        ("email_train.csv", train_idx),
        ("email_validation.csv", validation_idx),
        ("email_test.csv", test_idx),
    ):
        frame.iloc[indices][columns].to_csv(processed_dir / name, index=False)


def _undersample_indices(y, seed):
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    target_size = min(map(len, class_indices))
    sampled = np.concatenate(
        [rng.choice(indices, size=target_size, replace=False) for indices in class_indices]
    )
    return np.sort(sampled)


def _candidate_estimators():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError as error:
        raise RuntimeError(
            "The selected Python environment blocked scikit-learn tree extensions. "
            "Random Forest and HistGradientBoosting are required for the requested "
            "five-model comparison; use an administrator-approved environment that "
            "can import sklearn.ensemble before training."
        ) from error

    return {
        "Multinomial Naive Bayes": (
            lambda: MultinomialNB(alpha=0.5),
            "text_and_structural",
        ),
        "Logistic Regression": (
            lambda: LogisticRegression(
                C=2.0,
                max_iter=1600,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "text_and_structural",
        ),
        "Linear Support Vector Machine": (
            lambda: LinearSVC(max_iter=20_000, random_state=RANDOM_STATE),
            "text_and_structural",
        ),
        "Random Forest": (
            lambda: RandomForestClassifier(
                n_estimators=120,
                max_depth=24,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "text_and_structural",
        ),
        "HistGradientBoosting": (
            lambda: HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=15,
                learning_rate=0.08,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            "structural_only",
        ),
    }


def _get_scores(estimator, features):
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        class_index = list(estimator.classes_).index(1)
        scores = probabilities[:, class_index]
        predictions = (scores >= 0.5).astype(int)
    else:
        scores = estimator.decision_function(features)
        predictions = (scores >= 0).astype(int)
    return predictions, scores


def _metrics(y_true, predictions, scores):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "spam_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "spam_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "spam_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _features(frame, training=False, text_and_structural=True, transformer=None):
    if text_and_structural:
        if transformer is None:
            transformer = build_feature_transformer()
    else:
        prepared = prepare_model_frame(frame.to_dict("records"))
        structural = structural_features(
            prepared[["subject", "body", "sender", "html_body", "links"]]
        )
        return structural, transformer

    prepared = prepare_model_frame(frame.to_dict("records"))
    if training:
        return transformer.fit_transform(prepared), transformer
    return transformer.transform(prepared), transformer


def _fit_candidates(train_frame, validation_frame, model_factories):
    x_train, transformer = _features(train_frame, training=True)
    x_validation, _ = _features(
        validation_frame, text_and_structural=True, transformer=transformer
    )
    y_train = train_frame["target"].to_numpy()
    y_validation = validation_frame["target"].to_numpy()
    structural_train, _ = _features(
        train_frame, training=True, text_and_structural=False
    )
    structural_validation, _ = _features(
        validation_frame, text_and_structural=False
    )
    results = []
    for model_name, (factory, feature_type) in model_factories.items():
        for training_mode in ("original_imbalanced", "balanced_undersampled"):
            indices = (
                np.arange(len(y_train))
                if training_mode == "original_imbalanced"
                else _undersample_indices(y_train, RANDOM_STATE)
            )
            train_features = (
                x_train if feature_type == "text_and_structural" else structural_train
            )
            validation_features = (
                x_validation
                if feature_type == "text_and_structural"
                else structural_validation
            )
            estimator = factory()
            estimator.fit(train_features[indices], y_train[indices])
            predictions, scores = _get_scores(estimator, validation_features)
            metrics = _metrics(y_validation, predictions, scores)
            selection_score = (
                0.55 * metrics["spam_f1"]
                + 0.35 * metrics["pr_auc"]
                - 0.10 * metrics["false_positive_rate"]
            )
            results.append(
                {
                    "model": model_name,
                    "feature_type": feature_type,
                    "training_mode": training_mode,
                    "selection_score": selection_score,
                    **metrics,
                }
            )
    return results


def _test_all_candidates(development, test, model_factories):
    x_development, transformer = _features(development, training=True)
    x_test, _ = _features(test, text_and_structural=True, transformer=transformer)
    structural_development, _ = _features(
        development, training=True, text_and_structural=False
    )
    structural_test, _ = _features(test, text_and_structural=False)
    y_development = development["target"].to_numpy()
    y_test = test["target"].to_numpy()
    results = []
    for model_name, (factory, feature_type) in model_factories.items():
        development_features = (
            x_development
            if feature_type == "text_and_structural"
            else structural_development
        )
        test_features = x_test if feature_type == "text_and_structural" else structural_test
        for training_mode in ("original_imbalanced", "balanced_undersampled"):
            indices = (
                np.arange(len(y_development))
                if training_mode == "original_imbalanced"
                else _undersample_indices(y_development, RANDOM_STATE)
            )
            estimator = factory()
            estimator.fit(development_features[indices], y_development[indices])
            predictions, scores = _get_scores(estimator, test_features)
            results.append(
                {
                    "model": model_name,
                    "feature_type": feature_type,
                    "training_mode": training_mode,
                    **_metrics(y_test, predictions, scores),
                }
            )
    return results


def _select_operating_threshold(y_true, probabilities):
    candidates = []
    for threshold in np.linspace(0.20, 0.90, 71):
        predictions = (probabilities >= threshold).astype(int)
        metrics = _metrics(y_true, predictions, probabilities)
        candidates.append((threshold, metrics))
    acceptable = [
        item
        for item in candidates
        if item[1]["spam_precision"] >= 0.95
        and item[1]["false_positive_rate"] <= 0.02
    ]
    if acceptable:
        threshold, metrics = max(
            acceptable,
            key=lambda item: (
                item[1]["spam_recall"],
                item[1]["spam_f1"],
                -item[1]["false_positive_rate"],
                item[0],
            ),
        )
        return float(threshold), metrics, "maximum recall with validation precision >= 0.95 and FPR <= 0.02"

    threshold, metrics = max(
        candidates,
        key=lambda item: (
            0.55 * item[1]["spam_f1"]
            + 0.35 * item[1]["spam_precision"]
            + 0.10 * item[1]["spam_recall"]
            - 0.10 * item[1]["false_positive_rate"],
            item[1]["spam_f1"],
            -item[1]["false_positive_rate"],
        ),
    )
    return float(threshold), metrics, "F1/precision/recall composite with false-positive-rate penalty"


def _final_calibrated_model(development, model_name, training_mode, model_factories):
    factory, feature_type = model_factories[model_name]
    threshold_split = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_STATE + 2
    )
    fit_calibration_idx, threshold_idx = next(
        threshold_split.split(
            development,
            development["target"].to_numpy(),
            development["group"].to_numpy(),
        )
    )
    fit_calibration = development.iloc[fit_calibration_idx].reset_index(drop=True)
    threshold_validation = development.iloc[threshold_idx].reset_index(drop=True)
    calibration_split = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=RANDOM_STATE + 2
    )
    base_idx, calibration_idx = next(
        calibration_split.split(
            fit_calibration,
            fit_calibration["target"].to_numpy(),
            fit_calibration["group"].to_numpy(),
        )
    )
    base = fit_calibration.iloc[base_idx]
    calibration = fit_calibration.iloc[calibration_idx]
    y_base = base["target"].to_numpy()
    base_fit_idx = (
        np.arange(len(y_base))
        if training_mode == "original_imbalanced"
        else _undersample_indices(y_base, RANDOM_STATE + 2)
    )

    if feature_type == "text_and_structural":
        features, transformer = _features(base, training=True)
        model_pipeline = Pipeline(
            [("features", transformer), ("classifier", factory())]
        )
        model_pipeline.named_steps["classifier"].fit(
            features[base_fit_idx], y_base[base_fit_idx]
        )
        calibration_frame = prepare_model_frame(calibration.to_dict("records"))
    else:
        base_features = prepare_model_frame(base.to_dict("records"))
        model_pipeline = Pipeline(
            [
                (
                    "features",
                    FunctionTransformer(structural_features, validate=False),
                ),
                ("classifier", factory()),
            ]
        )
        model_pipeline.fit(
            base_features.iloc[base_fit_idx], y_base[base_fit_idx]
        )
        calibration_frame = prepare_model_frame(calibration.to_dict("records"))

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model_pipeline),
        method="sigmoid",
    )
    calibrated.fit(calibration_frame, calibration["target"].to_numpy())
    threshold_frame = prepare_model_frame(threshold_validation.to_dict("records"))
    threshold_probabilities = calibrated.predict_proba(threshold_frame)[:, 1]
    threshold, threshold_metrics, threshold_method = _select_operating_threshold(
        threshold_validation["target"].to_numpy(), threshold_probabilities
    )
    return calibrated, threshold, {
        "base_training_rows": int(len(base_fit_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "threshold_validation_rows": int(len(threshold_idx)),
        "calibration_method": "sigmoid on a separate, unbalanced development calibration split",
        "threshold_method": threshold_method,
        "threshold_validation_metrics": threshold_metrics,
    }


def _svg_header(title, width=900, height=440):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" rx="14" fill="#ffffff"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#334155}'
        '.muted{fill:#738096;font-size:12px}.title{font-size:20px;font-weight:700}'
        '.label{font-size:12px}.value{font-size:12px;font-weight:700}</style>'
        f'<text class="title" x="30" y="38">{html.escape(title)}</text>'
    )


def _save_comparison_chart(path, results):
    data = [item for item in results if item["training_mode"] == "original_imbalanced"]
    width, height = 980, 100 + 50 * len(data)
    svg = [_svg_header("Email model comparison — untouched test set", width, height)]
    max_value = max(
        max((item["pr_auc"] for item in data), default=0.01),
        max((item["spam_f1"] for item in data), default=0.01),
    )
    for index, item in enumerate(data):
        y = 73 + index * 50
        svg.append(
            f'<text class="label" x="30" y="{y + 15}">{html.escape(item["model"])}</text>'
        )
        for offset, key, color, label in (
            (0, "spam_f1", "#6366f1", "F1"),
            (20, "pr_auc", "#16a34a", "PR-AUC"),
        ):
            value = item[key]
            bar_width = 400 * value / max(max_value, 0.01)
            svg.extend(
                [
                    f'<text class="muted" x="275" y="{y + offset + 12}">{label}</text>',
                    f'<rect x="340" y="{y + offset}" width="400" height="14" rx="5" fill="#eef0f6"/>',
                    f'<rect x="340" y="{y + offset}" width="{bar_width:.2f}" height="14" rx="5" fill="{color}"/>',
                    f'<text class="value" x="755" y="{y + offset + 12}">{value:.3f}</text>',
                ]
            )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _save_metrics_chart(path, metrics):
    items = [
        ("Accuracy", metrics["accuracy"]),
        ("Spam precision", metrics["spam_precision"]),
        ("Spam recall", metrics["spam_recall"]),
        ("Spam F1", metrics["spam_f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
        ("PR-AUC", metrics["pr_auc"]),
    ]
    svg = [_svg_header("Selected model — untouched test metrics", 900, 380)]
    for index, (label, score) in enumerate(items):
        y = 70 + index * 46
        svg.extend(
            [
                f'<text class="label" x="35" y="{y + 16}">{label}</text>',
                f'<rect x="220" y="{y}" width="560" height="22" rx="6" fill="#eef0f6"/>',
                f'<rect x="220" y="{y}" width="{560 * score:.2f}" height="22" rx="6" fill="#6366f1"/>',
                f'<text class="value" x="798" y="{y + 16}">{score:.4f}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _save_curves(path, title, x_values, y_values, x_label, y_label):
    width, height = 640, 480
    left, top, plot_width, plot_height = 72, 70, 510, 330
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + float(x_value) * plot_width
        y = top + (1 - float(y_value)) * plot_height
        points.append(f"{x:.2f},{y:.2f}")
    svg = [_svg_header(title, width, height)]
    svg.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#64748b"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>',
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#6366f1" stroke-width="3"/>',
            f'<text class="muted" x="{left + plot_width // 2 - 35}" y="445">{x_label}</text>',
            f'<text class="muted" transform="translate(20 {top + plot_height // 2}) rotate(-90)" text-anchor="middle">{y_label}</text>',
        ]
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = left + fraction * plot_width
        y = top + (1 - fraction) * plot_height
        svg.append(
            f'<text class="muted" x="{x - 9}" y="{top + plot_height + 20}">{fraction:.2g}</text>'
        )
        svg.append(
            f'<text class="muted" x="{left - 30}" y="{y + 4}">{fraction:.2g}</text>'
        )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _save_word_frequency_chart(dataset, path):
    class_words = {}
    for label in ("legitimate", "spam"):
        counts = Counter()
        for record in dataset.loc[dataset["label"] == label, ["subject", "body"]].itertuples(
            index=False, name=None
        ):
            words = re.findall(r"\b[a-z]{3,}\b", normalize_text(" ".join(record)))
            counts.update(word for word in words if word not in ENGLISH_STOP_WORDS)
        class_words[label] = counts.most_common(12)
    maximum = max(
        (count for words in class_words.values() for _, count in words),
        default=1,
    )
    svg = [_svg_header("Frequent words by email label", 980, 460)]
    for column, (label, color) in enumerate(
        (("legitimate", "#6366f1"), ("spam", "#e87979"))
    ):
        origin_x = 35 + column * 485
        svg.append(
            f'<text class="value" x="{origin_x}" y="73">{label.title()}</text>'
        )
        for index, (word, count) in enumerate(class_words[label]):
            y = 93 + index * 28
            width = 300 * count / maximum
            svg.extend(
                [
                    f'<text class="label" x="{origin_x}" y="{y + 14}">{html.escape(word)}</text>',
                    f'<rect x="{origin_x + 115}" y="{y}" width="300" height="17" rx="5" fill="#eef0f6"/>',
                    f'<rect x="{origin_x + 115}" y="{y}" width="{width:.2f}" height="17" rx="5" fill="{color}"/>',
                    f'<text class="muted" x="{origin_x + 424}" y="{y + 13}">{count}</text>',
                ]
            )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def save_visualizations(dataset, test_labels, test_predictions, test_probabilities, comparisons, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = dataset["label"].value_counts()
    rows = "".join(
        f'<text class="label" x="35" y="{85 + index * 52}">{label}</text>'
        f'<rect x="220" y="{65 + index * 52}" width="500" height="26" rx="6" fill="#eef0f6"/>'
        f'<rect x="220" y="{65 + index * 52}" width="{500 * count / max(counts):.1f}" height="26" rx="6" fill="{color}"/>'
        f'<text class="value" x="735" y="{85 + index * 52}">{count:,}</text>'
        for index, (label, count, color) in enumerate(
            [
                ("Legitimate", int(counts.get("legitimate", 0)), "#6366f1"),
                ("Spam", int(counts.get("spam", 0)), "#e87979"),
            ]
        )
    )
    (output_dir / "dataset_distribution.svg").write_text(
        _svg_header("SpamAssassin email label distribution", 900, 190)
        + rows
        + "</svg>",
        encoding="utf-8",
    )
    _save_comparison_chart(output_dir / "algorithm_comparison.svg", comparisons)
    _save_metrics_chart(output_dir / "test_metrics.svg", test_predictions)
    _save_word_frequency_chart(dataset, output_dir / "word_frequency.svg")

    matrix = confusion_matrix(test_labels, test_predictions["predicted_labels"], labels=[0, 1])
    svg = [_svg_header("Selected model — untouched test confusion matrix", 620, 340)]
    svg.extend(
        [
            '<text class="muted" x="252" y="82">Predicted</text>',
            '<text class="muted" x="180" y="112">Legitimate</text>',
            '<text class="muted" x="318" y="112">Spam</text>',
            '<text class="muted" x="42" y="177">Actual</text>',
            '<text class="muted" x="36" y="198">legitimate</text>',
            '<text class="muted" x="42" y="257">spam</text>',
        ]
    )
    colors = [["#e9eaff", "#fff0ef"], ["#fff0ef", "#eaf8ef"]]
    for row in range(2):
        for column in range(2):
            x, y = 155 + column * 128, 132 + row * 60
            svg.extend(
                [
                    f'<rect x="{x}" y="{y}" width="116" height="48" rx="8" fill="{colors[row][column]}"/>',
                    f'<text x="{x + 58}" y="{y + 30}" text-anchor="middle" style="font-size:18px;font-weight:700">{int(matrix[row, column])}</text>',
                ]
            )
    svg.append("</svg>")
    (output_dir / "confusion_matrix.svg").write_text("".join(svg), encoding="utf-8")

    fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
    _save_curves(
        output_dir / "roc_curve.svg",
        "Selected model ROC curve",
        fpr,
        tpr,
        "False positive rate",
        "True positive rate",
    )
    precision, recall, _ = precision_recall_curve(test_labels, test_probabilities)
    _save_curves(
        output_dir / "precision_recall_curve.svg",
        "Selected model precision-recall curve",
        recall,
        precision,
        "Recall",
        "Precision",
    )


def train(dataset_path, processed_dir, model_path, reports_dir):
    dataset, cleaning = load_email_dataset(dataset_path)
    selection_train_idx, validation_idx, development_idx, test_idx = _split_indices(dataset)
    _save_splits(dataset, processed_dir, selection_train_idx, validation_idx, test_idx)

    selection_train = dataset.iloc[selection_train_idx].reset_index(drop=True)
    validation = dataset.iloc[validation_idx].reset_index(drop=True)
    development = dataset.iloc[development_idx].reset_index(drop=True)
    test = dataset.iloc[test_idx].reset_index(drop=True)
    model_factories = _candidate_estimators()

    validation_results = _fit_candidates(selection_train, validation, model_factories)
    selected = max(
        validation_results,
        key=lambda item: (
            item["selection_score"],
            item["spam_f1"],
            item["pr_auc"],
            -item["false_positive_rate"],
        ),
    )
    test_comparisons = _test_all_candidates(development, test, model_factories)
    calibrated_model, threshold, calibration_metadata = _final_calibrated_model(
        development,
        selected["model"],
        selected["training_mode"],
        model_factories,
    )

    prepared_test = prepare_model_frame(test.to_dict("records"))
    test_probabilities = calibrated_model.predict_proba(prepared_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    final_metrics = _metrics(test["target"].to_numpy(), test_predictions, test_probabilities)
    final_metrics["classification_report"] = classification_report(
        test["target"],
        test_predictions,
        labels=[0, 1],
        target_names=["legitimate", "spam"],
        output_dict=True,
        zero_division=0,
    )
    final_metrics["brier_score"] = float(
        np.mean((test_probabilities - test["target"].to_numpy()) ** 2)
    )
    final_metrics["decision_threshold"] = threshold
    final_metrics["predicted_labels"] = test_predictions
    final_metadata = {
        "model_name": selected["model"],
        "training_mode": selected["training_mode"],
        "feature_type": selected["feature_type"],
        "dataset_name": "Apache SpamAssassin Public Corpus",
        "dataset_scope": "Email-only; SMS corpus is trained/evaluated separately and never mixed.",
        "dataset_records_after_deduplication": int(len(dataset)),
        "dataset_label_distribution": {
            str(label): int(count) for label, count in dataset["label"].value_counts().items()
        },
        "split_counts": {
            "selection_train": int(len(selection_train)),
            "validation": int(len(validation)),
            "development": int(len(development)),
            "test": int(len(test)),
        },
        "selection_metric": "validation F1/PR-AUC composite, with false-positive-rate penalty",
        "selection_score": selected["selection_score"],
        "random_state": RANDOM_STATE,
        "balancing": "Random undersampling is applied only to training indices. Validation and test keep their natural class distribution.",
        "calibration": calibration_metadata,
        "threshold": threshold,
        "metrics": {key: value for key, value in final_metrics.items() if key != "predicted_labels"},
        "software_versions": {"scikit_learn": sklearn.__version__},
        "cleaning": cleaning,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "email_evaluation_metrics.json").write_text(
        json.dumps(
            {
                "dataset": final_metadata,
                "selected_model": selected["model"],
                "selected_training_mode": selected["training_mode"],
                "selection_validation_metrics": validation_results,
                "untouched_test_comparisons": test_comparisons,
                "final_calibrated_test_metrics": {
                    key: value
                    for key, value in final_metrics.items()
                    if key != "predicted_labels"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "classification_report"}
            for item in test_comparisons
        ]
    ).to_csv(reports_dir / "email_model_comparison.csv", index=False)
    save_visualizations(
        dataset,
        test["target"].to_numpy(),
        final_metrics,
        test_probabilities,
        test_comparisons,
        reports_dir / "figures",
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(".tmp")
    joblib.dump(
        {
            "pipeline": calibrated_model,
            "metadata": final_metadata,
        },
        temporary_model_path,
    )
    temporary_model_path.replace(model_path)
    print(f"Email corpus records: {len(dataset)}")
    print(f"Selected model: {selected['model']} ({selected['training_mode']})")
    print(
        "Test accuracy/precision/recall/F1: "
        f"{final_metrics['accuracy']:.4f}/"
        f"{final_metrics['spam_precision']:.4f}/"
        f"{final_metrics['spam_recall']:.4f}/"
        f"{final_metrics['spam_f1']:.4f}"
    )
    print(f"Test confusion matrix [TN,FP;FN,TP]: {final_metrics['confusion_matrix']}")
    print(f"Saved email model: {model_path}")
    print(f"Saved processed splits/reports: {processed_dir}, {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train an email-specific spam/phishing model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports-output", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise SystemExit(
            f"Email dataset not found: {args.dataset}. Run "
            "'python scripts/inspect_email_corpus.py' after downloading the archives."
        )
    train(args.dataset, args.processed_dir, args.model_output, args.reports_output)


if __name__ == "__main__":
    main()
