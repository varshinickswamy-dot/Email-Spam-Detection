import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


@contextmanager
def _connect(database_path):
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(database_path):
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT NOT NULL CHECK (label IN ('spam', 'not_spam')),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        migrations = {
            "user_id": "INTEGER REFERENCES users(id) ON DELETE CASCADE",
            "subject": "TEXT NOT NULL DEFAULT ''",
            "sender": "TEXT NOT NULL DEFAULT ''",
            "classification": "TEXT NOT NULL DEFAULT ''",
            "risk_level": "TEXT NOT NULL DEFAULT 'low'",
            "risk_score": "INTEGER NOT NULL DEFAULT 0",
            "details_json": "TEXT NOT NULL DEFAULT '{}'",
            "batch_id": "TEXT",
            "scan_kind": "TEXT NOT NULL DEFAULT 'single'",
        }
        for column, definition in migrations.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {definition}"
                )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_batches (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_user_created "
            "ON predictions(user_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_user_batch "
            "ON predictions(user_id, batch_id)"
        )


def create_user(database_path, email, password):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email.strip().lower(), generate_password_hash(password), created_at),
        )
        return cursor.lastrowid


def get_user_by_email(database_path, email):
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(database_path, user_id):
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def verify_user(database_path, email, password):
    user = get_user_by_email(database_path, email)
    if user and check_password_hash(user["password_hash"], password):
        return {key: user[key] for key in ("id", "email", "created_at")}
    return None


def create_scan_batch(database_path, batch_id, user_id, total):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(database_path) as connection:
        connection.execute(
            "INSERT INTO scan_batches (id, user_id, created_at, total) VALUES (?, ?, ?, ?)",
            (batch_id, user_id, created_at, total),
        )
    return created_at


def record_scan(database_path, user_id, email, result, batch_id=None, scan_kind="single"):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = "\n".join(
        part
        for part in (
            f"Subject: {email.get('subject', '')}" if email.get("subject") else "",
            email.get("body", ""),
            email.get("html_body", ""),
        )
        if part
    )[:50_000]
    label = (
        "spam"
        if result["spam_probability"] >= result["classification_threshold"]
        else "not_spam"
    )
    details = {
        key: result[key]
        for key in (
            "classification_label",
            "probability_type",
            "classification_threshold",
            "uncertainty_band",
            "risk_level",
            "risk_score",
            "reasons",
            "suspicious_phrases",
            "suspicious_urls",
            "recommended_action",
        )
    }
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                user_id, text, label, confidence, created_at, subject, sender,
                classification, risk_level, risk_score, details_json, batch_id, scan_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                text,
                label,
                result["spam_probability"],
                created_at,
                email.get("subject", "")[:500],
                email.get("sender", "")[:320],
                result["classification"],
                result["risk_level"],
                result["risk_score"],
                json.dumps(details, ensure_ascii=False),
                batch_id,
                scan_kind,
            ),
        )
        return cursor.lastrowid


def get_history(database_path, user_id, query="", classification="", limit=100):
    clauses = ["user_id = ?"]
    parameters = [user_id]
    if query:
        clauses.append("(text LIKE ? OR sender LIKE ?)")
        parameters.extend((f"%{query}%", f"%{query}%"))
    if classification:
        clauses.append("classification = ?")
        parameters.append(classification)
    parameters.append(limit)
    with _connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT id, text, label, confidence, created_at, subject, sender,
                   classification, risk_level, risk_score, details_json, batch_id
            FROM predictions WHERE {' AND '.join(clauses)}
            ORDER BY id DESC LIMIT ?
            """,
            parameters,
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        items.append(item)
    return items


def get_batch_history(database_path, user_id, batch_id):
    with _connect(database_path) as connection:
        batch = connection.execute(
            "SELECT id, created_at, total FROM scan_batches WHERE id = ? AND user_id = ?",
            (batch_id, user_id),
        ).fetchone()
    if not batch:
        return None
    return {
        "id": batch["id"],
        "created_at": batch["created_at"],
        "total": batch["total"],
        "items": _get_batch_items(database_path, user_id, batch_id),
    }


def _get_batch_items(database_path, user_id, batch_id):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, text, label, confidence, created_at, subject, sender,
                   classification, risk_level, risk_score, details_json, batch_id
            FROM predictions
            WHERE user_id = ? AND batch_id = ?
            ORDER BY id
            """,
            (user_id, batch_id),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        items.append(item)
    return items


def get_statistics(database_path, user_id):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT classification, label, COUNT(*) AS count
            FROM predictions WHERE user_id = ?
            GROUP BY classification, label
            """,
            (user_id,),
        ).fetchall()
        trends = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS total,
                   SUM(CASE WHEN label = 'spam' THEN 1 ELSE 0 END) AS spam
            FROM predictions WHERE user_id = ?
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day DESC LIMIT 14
            """,
            (user_id,),
        ).fetchall()
        recent = connection.execute(
            """
            SELECT id, subject, sender, classification, risk_level, confidence, created_at
            FROM predictions WHERE user_id = ?
            ORDER BY id DESC LIMIT 8
            """,
            (user_id,),
        ).fetchall()

    distribution = {
        "safe": 0,
        "promotional": 0,
        "spam": 0,
        "phishing_suspicious": 0,
        "uncertain": 0,
    }
    total = spam = legitimate = 0
    for row in rows:
        key = row["classification"] or ("spam" if row["label"] == "spam" else "safe")
        if key in distribution:
            distribution[key] += row["count"]
        total += row["count"]
        if row["label"] == "spam":
            spam += row["count"]
        else:
            legitimate += row["count"]
    return {
        "total": total,
        "spam": spam,
        "legitimate": legitimate,
        "phishing": distribution["phishing_suspicious"],
        "promotional": distribution["promotional"],
        "suspicious": distribution["phishing_suspicious"],
        "uncertain": distribution["uncertain"],
        "spam_rate": round(spam / total, 4) if total else 0,
        "distribution": distribution,
        "trends": [dict(row) for row in reversed(trends)],
        "recent_scans": [dict(row) for row in recent],
    }


def clear_history(database_path, user_id):
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE user_id = ?", (user_id,)
        )
        connection.execute(
            "DELETE FROM scan_batches WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount


def record_prediction(database_path, text, label, confidence):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (text, label, confidence, created_at, classification)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text, label, confidence, created_at, "spam" if label == "spam" else "safe"),
        )
        return cursor.lastrowid
