import json
import sqlite3
from pathlib import Path

import pandas as pd

from backend.database import init_db
from backend.detection_engine import analyze_email
from backend.email_features import inspect_urls, parse_html, parse_raw_email
from backend.model_service import EmailSecurityModel
from scripts.train_email_model import (
    _select_operating_threshold,
    _split_indices,
    _undersample_indices,
    load_email_dataset,
)


def test_bundled_email_model_loads_and_scores_email():
    model = EmailSecurityModel(
        Path(__file__).resolve().parents[1] / "models" / "email_security_model.joblib"
    )
    probability = model.predict_spam_probability(
        {
            "subject": "Updated schedule",
            "body": "The updated schedule is available in the shared folder.",
        }
    )
    assert 0 <= probability <= 1
    assert 0 <= model.threshold <= 1


def test_html_parser_ignores_script_content_and_extracts_anchor_text():
    text, anchors = parse_html(
        '<p>Review your account</p><script>window.secret = "ignored";</script>'
        '<a href="https://bad.example/login">Trusted bank</a>'
    )
    assert "Review your account" in text
    assert "ignored" not in text
    assert anchors == [{"url": "https://bad.example/login", "text": " Trusted bank"}]


def test_url_inspection_flags_ip_and_misleading_display_domain_without_fetching():
    urls = inspect_urls(
        {
            "body": "Visit http://192.0.2.1/verify",
            "html_body": '<a href="https://other.example/login">https://bank.example</a>',
        }
    )
    assert len(urls) == 3
    assert "IP-address host" in urls[0]["flags"]
    assert "unencrypted HTTP link" in urls[0]["flags"]
    assert "displayed domain differs from link destination" in urls[1]["flags"]


def test_raw_mime_parser_extracts_headers_and_plain_body():
    parsed = parse_raw_email(
        "From: Team <team@example.org>\r\n"
        "Subject: Weekly notes\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Notes are ready."
    )
    assert parsed["subject"] == "Weekly notes"
    assert parsed["sender"] == "Team <team@example.org>"
    assert parsed["body"] == "Notes are ready."


def test_uncertain_model_score_is_not_forced_to_safe_or_spam():
    result = analyze_email(
        {"subject": "Unexpected message", "body": "Please review this note."},
        0.5,
    )
    assert result["classification"] == "uncertain"
    assert result["probability_type"] == "calibrated_sigmoid_probability"
    assert any("decision boundary" in reason for reason in result["reasons"])


def test_credential_request_with_ip_link_is_flagged_even_at_low_spam_score():
    result = analyze_email(
        {
            "sender": "Bank Support <alerts@unknown.example>",
            "subject": "Verify your password",
            "body": "Verify your password at http://192.0.2.1/login",
        },
        0.1,
    )
    assert result["classification"] == "phishing_suspicious"
    assert result["risk_score"] >= 35


def test_group_split_keeps_near_duplicates_together_and_balances_only_training():
    frame = pd.DataFrame(
        {
            "target": [0] * 80 + [1] * 20,
            "group": [f"ham-{index}" for index in range(80)]
            + [f"spam-{index}" for index in range(20)],
        }
    )
    train_idx, validation_idx, _, test_idx = _split_indices(frame)
    train_groups = set(frame.iloc[train_idx]["group"])
    validation_groups = set(frame.iloc[validation_idx]["group"])
    test_groups = set(frame.iloc[test_idx]["group"])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    sampled = _undersample_indices(frame.iloc[train_idx]["target"].to_numpy(), 7)
    labels = frame.iloc[train_idx]["target"].to_numpy()[sampled]
    assert labels.sum() == len(labels) / 2


def test_operating_threshold_limits_validation_false_positives():
    threshold, metrics, method = _select_operating_threshold(
        [0, 0, 0, 1, 1, 1],
        [0.1, 0.4, 0.7, 0.6, 0.8, 0.9],
    )
    assert threshold == 0.8
    assert metrics["spam_precision"] == 1
    assert metrics["false_positive_rate"] == 0
    assert "precision >= 0.95" in method


def test_email_dataset_loader_removes_exact_duplicate_records(tmp_path):
    source = tmp_path / "emails.jsonl"
    record = {
        "label": "spam",
        "subject": "Offer",
        "body": "A message",
        "sender": "offers@example.org",
        "source_id": "one",
    }
    with source.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
        file.write(json.dumps({**record, "source_id": "duplicate"}) + "\n")
        file.write(
            json.dumps(
                {
                    "label": "legitimate",
                    "subject": "Meeting",
                    "body": "See you tomorrow",
                    "sender": "team@example.org",
                    "source_id": "two",
                }
            )
            + "\n"
        )
    frame, cleaning = load_email_dataset(source)
    assert len(frame) == 2
    assert cleaning["duplicate_rows_removed"] == 1


def test_database_additive_migration_preserves_existing_sms_history(tmp_path):
    database = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT NOT NULL CHECK (label IN ('spam', 'not_spam')),
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO predictions (text, label, confidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("legacy SMS", "not_spam", 0.8, "2024-01-01T00:00:00+00:00"),
        )
    init_db(str(database))
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT text, label, user_id, classification FROM predictions"
        ).fetchone()
    assert row == ("legacy SMS", "not_spam", None, "")
