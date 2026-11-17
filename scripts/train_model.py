import argparse
import html
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from backend.text_processing import clean_text


DEFAULT_DATASET = ROOT_DIR / "data" / "raw" / "SMSSpamCollection"
RANDOM_STATE = 42


def load_dataset(path):
    dataset = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        dtype=str,
        on_bad_lines="skip",
    )
    dataset = dataset.dropna(subset=["label", "text"])
    dataset["label"] = dataset["label"].str.strip().str.lower()
    dataset["text"] = dataset["text"].str.strip()
    dataset = dataset[dataset["label"].isin(["ham", "spam"]) & dataset["text"].ne("")]
    dataset = dataset.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    if dataset["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam messages.")
    dataset["target"] = dataset["label"].map({"ham": 0, "spam": 1})
    return dataset


def build_pipeline(classifier):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    stop_words="english",
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["not_spam", "spam"],
        output_dict=True,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    spam_precision = precision_score(y_test, predictions, pos_label=1, zero_division=0)
    spam_recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    spam_f1 = f1_score(y_test, predictions, pos_label=1, zero_division=0)
    beta = 1.5
    spam_f_beta = (
        (1 + beta**2) * spam_precision * spam_recall
        / (beta**2 * spam_precision + spam_recall)
        if (beta**2 * spam_precision + spam_recall)
        else 0.0
    )
    return pipeline, {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(spam_precision),
        "spam_recall": float(spam_recall),
        "spam_f1": float(spam_f1),
        "spam_f2": float(spam_f_beta),
        "false_positives": int(fp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": report,
    }


def save_visualizations(dataset, results, y_test, predictions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = dataset["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    _write_horizontal_bars(
        output_dir / "dataset_distribution.svg",
        "Dataset label distribution",
        [("Not spam", int(counts["ham"]), "#6366f1"), ("Spam", int(counts["spam"]), "#e87979")],
        value_suffix=" messages",
        max_value=max(counts.values) or 1,
    )
    _write_horizontal_bars(
        output_dir / "model_comparison.svg",
        "Algorithm comparison — Spam F2",
        [(item["model"], item["spam_f2"], "#6366f1") for item in results],
        value_suffix="",
        max_value=1,
    )
    _write_metric_bars(output_dir / "metrics_by_model.svg", results)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    _write_confusion_matrix(output_dir / "confusion_matrix.svg", matrix)

    spam_words = " ".join(dataset.loc[dataset["target"] == 1, "text"].map(clean_text))
    legitimate_words = " ".join(dataset.loc[dataset["target"] == 0, "text"].map(clean_text))
    word_counts = {}
    for label, text in (("Spam", spam_words), ("Not spam", legitimate_words)):
        frequencies = pd.Series(text.split()).value_counts().head(12)
        for word, frequency in frequencies.items():
            word_counts[(label, word)] = int(frequency)
    words = pd.DataFrame(
        [(label, word, count) for (label, word), count in word_counts.items()],
        columns=["class", "word", "count"],
    )
    _write_word_frequencies(output_dir / "word_frequency.svg", words)


def _svg_header(title, width, height):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" rx="14" fill="#ffffff"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#334155}'
        '.muted{fill:#7b8798;font-size:12px}.title{font-size:20px;font-weight:700}'
        '.label{font-size:12px}.value{font-size:12px;font-weight:700}</style>'
        f'<text class="title" x="28" y="38">{html.escape(title)}</text>'
    )


def _write_horizontal_bars(path, title, rows, value_suffix="", max_value=1):
    width, height = 840, max(160, 82 + 42 * len(rows))
    svg = [_svg_header(title, width, height)]
    bar_start, bar_width = 250, 500
    for index, (label, value, color) in enumerate(rows):
        y = 72 + index * 42
        amount = float(value)
        length = max(2, min(bar_width, amount / max_value * bar_width))
        value_text = f"{amount:.3f}" if isinstance(value, float) else f"{int(value):,}"
        svg.extend(
            [
                f'<text class="label" x="28" y="{y + 15}">{html.escape(str(label))}</text>',
                f'<rect x="{bar_start}" y="{y}" width="{bar_width}" height="22" rx="6" fill="#f1f3f8"/>',
                f'<rect x="{bar_start}" y="{y}" width="{length:.1f}" height="22" rx="6" fill="{color}"/>',
                f'<text class="value" x="{bar_start + bar_width + 14}" y="{y + 15}">'
                f"{value_text}{html.escape(value_suffix)}</text>",
            ]
        )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _write_metric_bars(path, results):
    width, height = 900, 110 + 94 * len(results)
    colors = {"spam_precision": "#6366f1", "spam_recall": "#16a34a", "spam_f1": "#e87979"}
    labels = {"spam_precision": "Precision", "spam_recall": "Recall", "spam_f1": "F1"}
    svg = [_svg_header("Spam-class evaluation metrics", width, height)]
    for index, result in enumerate(results):
        top = 72 + index * 94
        svg.append(
            f'<text class="label" x="28" y="{top + 17}">{html.escape(result["model"])}</text>'
        )
        for offset, key in enumerate(colors):
            y = top + offset * 22
            score = result[key]
            svg.extend(
                [
                    f'<text class="muted" x="270" y="{y + 13}">{labels[key]}</text>',
                    f'<rect x="365" y="{y}" width="390" height="16" rx="5" fill="#f1f3f8"/>',
                    f'<rect x="365" y="{y}" width="{390 * score:.1f}" height="16" rx="5" fill="{colors[key]}"/>',
                    f'<text class="value" x="770" y="{y + 13}">{score:.3f}</text>',
                ]
            )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _write_confusion_matrix(path, matrix):
    svg = [_svg_header("Selected model confusion matrix", 620, 340)]
    svg.extend(
        [
            '<text class="muted" x="266" y="82">Predicted label</text>',
            '<text class="muted" x="184" y="112">Not spam</text>',
            '<text class="muted" x="310" y="112">Spam</text>',
            '<text class="muted" x="28" y="177">Actual</text>',
            '<text class="muted" x="28" y="196">not spam</text>',
            '<text class="muted" x="28" y="253">spam</text>',
        ]
    )
    colors = [["#e9eaff", "#fff0ef"], ["#fff0ef", "#eaf8ef"]]
    for row in range(2):
        for column in range(2):
            x, y = 150 + column * 128, 132 + row * 58
            svg.extend(
                [
                    f'<rect x="{x}" y="{y}" width="116" height="48" rx="8" fill="{colors[row][column]}"/>',
                    f'<text x="{x + 58}" y="{y + 30}" text-anchor="middle" '
                    f'style="font-size:18px;font-weight:700">{int(matrix[row, column])}</text>',
                ]
            )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def _write_word_frequencies(path, words):
    rows = words.sort_values(["class", "count"], ascending=[True, False]).to_dict("records")
    width, height = 900, max(180, 90 + 28 * len(rows))
    svg = [_svg_header("Most frequent terms by class", width, height)]
    maximum = max((row["count"] for row in rows), default=1)
    colors = {"Spam": "#e87979", "Not spam": "#6366f1"}
    for index, row in enumerate(rows):
        y = 67 + index * 28
        bar_width = 450 * row["count"] / maximum
        svg.extend(
            [
                f'<text class="label" x="28" y="{y + 14}">{html.escape(row["word"])}</text>',
                f'<text class="muted" x="160" y="{y + 14}">{html.escape(row["class"])}</text>',
                f'<rect x="275" y="{y}" width="450" height="18" rx="5" fill="#f1f3f8"/>',
                f'<rect x="275" y="{y}" width="{bar_width:.1f}" height="18" rx="5" '
                f'fill="{colors[row["class"]]}"/>',
                f'<text class="value" x="740" y="{y + 14}">{row["count"]}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def train(dataset_path, model_path, reports_dir):
    dataset = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["target"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset["target"],
    )
    candidates = [
        ("Multinomial Naive Bayes", MultinomialNB(alpha=0.5)),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Linear Support Vector Machine",
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000),
        ),
    ]

    fitted = []
    results = []
    for name, classifier in candidates:
        pipeline, metrics = evaluate_model(
            name, build_pipeline(classifier), x_train, x_test, y_train, y_test
        )
        fitted.append((pipeline, metrics))
        results.append(metrics)

    best_pipeline, best_metrics = max(
        fitted,
        key=lambda item: (
            item[1]["spam_f2"],
            item[1]["spam_precision"],
            item[1]["accuracy"],
        ),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "metadata": {
                "model_name": best_metrics["model"],
                "dataset": "UCI SMS Spam Collection",
                "dataset_rows_after_deduplication": int(len(dataset)),
                "test_rows": int(len(y_test)),
                "random_state": RANDOM_STATE,
                "selection_metric": "spam F2 (beta=1.5), then spam precision, then accuracy",
                "metrics": best_metrics,
            },
        },
        model_path,
    )
    with (reports_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_rows": int(len(dataset)),
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "random_state": RANDOM_STATE,
                "selected_model": best_metrics["model"],
                "selection_metric": "spam F2 (beta=1.5), then spam precision, then accuracy",
                "models": results,
            },
            file,
            indent=2,
        )
    pd.DataFrame(results).drop(columns="classification_report").to_csv(
        reports_dir / "model_comparison.csv", index=False
    )
    selected_predictions = best_pipeline.predict(x_test)
    save_visualizations(
        dataset,
        results,
        y_test,
        selected_predictions,
        reports_dir / "figures",
    )

    print(f"Dataset rows after cleanup: {len(dataset)}")
    print(f"Training/test rows: {len(y_train)}/{len(y_test)}")
    print(f"Selected model: {best_metrics['model']}")
    print(
        "Spam precision/recall/F1: "
        f"{best_metrics['spam_precision']:.4f}/"
        f"{best_metrics['spam_recall']:.4f}/"
        f"{best_metrics['spam_f1']:.4f}"
    )
    print(f"False positives on test set: {best_metrics['false_positives']}")
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation report and charts: {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SMS spam classifiers.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--model-output", type=Path, default=ROOT_DIR / "models" / "spam_classifier.joblib"
    )
    parser.add_argument("--reports-output", type=Path, default=ROOT_DIR / "reports")
    args = parser.parse_args()
    if not args.dataset.is_file():
        print(
            f"Dataset not found: {args.dataset}\n"
            "Download the UCI SMS Spam Collection and extract its SMSSpamCollection file "
            "to data/raw/SMSSpamCollection.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    train(args.dataset, args.model_output, args.reports_output)


if __name__ == "__main__":
    main()
