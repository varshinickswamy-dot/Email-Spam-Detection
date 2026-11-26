import csv
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from backend import create_app
from backend import database
from backend.detection_engine import analyze_email, detect_phishing_campaigns
from backend.email_features import inspect_urls


class StubEmailModel:
    threshold = 0.5
    info = {
        "model_name": "Test email model",
        "metrics": {"spam_precision": 0.9, "spam_recall": 0.8, "spam_f1": 0.84},
    }

    def predict_spam_probability(self, record):
        content = f"{record['subject']} {record['body']} {record['html_body']}".lower()
        return 0.95 if "prize" in content or "password" in content else 0.05


@pytest.fixture()
def app(tmp_path):
    sent_messages = []
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "history.sqlite3"),
            "EMAIL_MODEL": StubEmailModel(),
            "EMAIL_SENDER": lambda _recipient, _subject, _body: sent_messages.append(
                {
                    "recipient": _recipient,
                    "subject": _subject,
                    "body": _body,
                }
            ),
        }
    )
    app.extensions["sent_messages"] = sent_messages
    return app


@pytest.fixture()
def client(app):
    client = app.test_client()
    register(client, "analyst@example.org")
    return client


def register(client, email, password="secure-demo-password-123"):
    response = client.post(
        "/register",
        data={
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )
    assert response.status_code == 200
    sent_message = client.application.extensions["sent_messages"][-1]
    link = sent_message["body"].splitlines()[2]
    token = urlsplit(link).fragment
    verified = client.post("/verify-email", data={"token": token})
    assert verified.status_code == 200
    assert b"Your email is verified" in verified.data
    signed_in = client.post(
        "/login", data={"email": email, "password": password}
    )
    assert signed_in.status_code == 302
    return client


def test_public_pages_auth_and_model_info(app):
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 200
    assert client.get("/privacy").status_code == 200
    assert client.get("/dashboard").status_code == 302
    assert client.get("/api/model-info").status_code == 401
    register(client, "reader@example.org")
    assert client.get("/api/model-info").get_json()["model_name"] == "Test email model"


def test_registration_requires_one_time_email_verification(app):
    client = app.test_client()
    response = client.post(
        "/register",
        data={
            "email": "verify@example.org",
            "password": "secure-demo-password-123",
            "confirm_password": "secure-demo-password-123",
        },
    )
    assert response.status_code == 200
    assert b"Check your inbox" in response.data
    assert client.get("/dashboard").status_code == 302
    login = client.post(
        "/login",
        data={"email": "verify@example.org", "password": "secure-demo-password-123"},
    )
    assert b"Verify your email" in login.data

    message = app.extensions["sent_messages"][-1]
    token = urlsplit(message["body"].splitlines()[2]).fragment
    assert client.get("/verify-email").status_code == 200
    verified = client.post("/verify-email", data={"token": token})
    assert verified.status_code == 200
    assert b"Your email is verified" in verified.data
    assert client.post("/verify-email", data={"token": token}).status_code == 400
    assert client.post(
        "/login",
        data={"email": "verify@example.org", "password": "secure-demo-password-123"},
    ).status_code == 302


def test_password_recovery_is_generic_one_time_and_revokes_sessions(client, app):
    client.post(
        "/login",
        data={"email": "analyst@example.org", "password": "secure-demo-password-123"},
    )
    client.post("/forgot-password", data={"email": "analyst@example.org"})
    assert b"If a verified account matches" in client.post(
        "/forgot-password", data={"email": "unknown@example.org"}
    ).data
    sent_message = app.extensions["sent_messages"][-1]
    token = urlsplit(sent_message["body"].splitlines()[2]).fragment
    assert client.get("/reset-password").status_code == 200
    reset = client.post(
        "/reset-password",
        data={
            "token": token,
            "password": "another-secure-password-456",
            "confirm_password": "another-secure-password-456",
        },
    )
    assert reset.status_code == 200
    assert b"Your password has been updated" in reset.data
    assert client.get("/dashboard").status_code == 302
    assert client.post(
        "/reset-password",
        data={
            "token": token,
            "password": "third-secure-password-789",
            "confirm_password": "third-secure-password-789",
        },
    ).status_code == 400
    assert client.post(
        "/login",
        data={"email": "analyst@example.org", "password": "another-secure-password-456"},
    ).status_code == 302


def test_account_deletion_requires_password_and_removes_associated_data(client, app):
    client.post(
        "/login",
        data={"email": "analyst@example.org", "password": "secure-demo-password-123"},
    )
    client.post("/api/scan", json={"body": "Saved message"})
    user = database.get_user_by_email(
        app.config["DATABASE"], "analyst@example.org"
    )
    failed = client.post(
        "/account",
        data={"password": "wrong-password", "confirm_delete": "DELETE"},
    )
    assert failed.status_code == 400
    assert database.get_history(app.config["DATABASE"], user["id"])

    deleted = client.post(
        "/account",
        data={
            "password": "secure-demo-password-123",
            "confirm_delete": "DELETE",
        },
    )
    assert deleted.status_code == 302
    assert database.get_user_by_id(app.config["DATABASE"], user["id"]) is None
    assert database.get_history(app.config["DATABASE"], user["id"]) == []
    assert client.get("/api/history").status_code == 401


def test_registration_and_scan_rate_limits(app):
    client = app.test_client()
    registration = {
        "password": "secure-demo-password-123",
        "confirm_password": "secure-demo-password-123",
    }
    for index in range(5):
        response = client.post(
            "/register",
            data={**registration, "email": f"limit-{index}@example.org"},
        )
        assert response.status_code == 200
    limited_registration = client.post(
        "/register",
        data={**registration, "email": "limit-six@example.org"},
    )
    assert limited_registration.status_code == 429
    assert int(limited_registration.headers["Retry-After"]) > 0

    scan_client = app.test_client()
    scan_client.environ_base["REMOTE_ADDR"] = "198.51.100.10"
    register(scan_client, "scanner@example.org")
    for _ in range(5):
        assert scan_client.post("/api/scan", json={"body": "Hello"}).status_code == 200
    assert scan_client.post("/api/scan", json={"body": "Hello"}).status_code == 429


def test_failed_sign_in_rate_limit_is_per_ip(app):
    client = app.test_client()
    for _ in range(10):
        response = client.post(
            "/login",
            data={"email": "unknown@example.org", "password": "wrong-password"},
        )
        assert response.status_code == 200
    limited = client.post(
        "/login",
        data={"email": "unknown@example.org", "password": "wrong-password"},
    )
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_registration_fails_explicitly_without_mail_configuration(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "no-mail.sqlite3"),
            "EMAIL_MODEL": StubEmailModel(),
        }
    )
    client = app.test_client()
    response = client.post(
        "/register",
        data={
            "email": "no-mail@example.org",
            "password": "secure-demo-password-123",
            "confirm_password": "secure-demo-password-123",
        },
    )
    assert response.status_code == 200
    assert b"Account creation is temporarily unavailable" in response.data
    assert b"INBOXGUARD_SMTP_PASSWORD" in response.data
    assert (
        database.get_user_by_email(
            app.config["DATABASE"], "no-mail@example.org"
        )
        is None
    )


def test_first_request_prunes_scan_history_older_than_retention(
    tmp_path, monkeypatch
):
    database_path = str(tmp_path / "retention.sqlite3")
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": database_path,
            "EMAIL_MODEL": StubEmailModel(),
            "RETENTION_DAYS": 30,
        }
    )
    user_id = database.create_user(
        database_path, "retention@example.org", "secure-demo-password-123"
    )
    expired_user_id = database.create_user(
        database_path, "expired@example.org", "secure-demo-password-123"
    )
    recent = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(
        timespec="seconds"
    )
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO predictions (text, label, confidence, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("old message", "not_spam", 0.1, old, user_id),
                ("recent message", "not_spam", 0.1, recent, user_id),
            ],
        )
        connection.execute(
            "UPDATE users SET created_at = ? WHERE id = ?",
            (old, expired_user_id),
        )
    monkeypatch.setattr("backend.time.monotonic", lambda: 10.0)
    app.test_client().get("/api/history")
    items = database.get_history(database_path, user_id)
    assert [item["text"] for item in items] == ["recent message"]
    assert database.get_user_by_id(database_path, expired_user_id) is None


def test_static_assets_are_served_from_public_directory(app):
    response = app.test_client().get("/static/js/security.js")
    assert response.status_code == 200
    assert b"loadDashboard();" in response.data


def test_authenticated_header_does_not_expose_account_email(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Account" in response.data
    assert b"analyst@example.org" not in response.data


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"subject": "", "body": "   "},
        {"subject": ["bad"], "body": "message"},
        {"body": "x" * 50_001},
        {"body": "content", "links": [12]},
    ],
)
def test_scan_rejects_invalid_input(client, payload):
    assert client.post("/api/scan", json=payload).status_code == 400


def test_scan_persists_results_and_updates_user_stats(client):
    response = client.post(
        "/api/scan",
        json={
            "sender": "Notices <alerts@example.test>",
            "subject": "Claim your prize",
            "body": "Claim your free cash prize now.",
        },
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["classification"] == "spam"
    assert result["probability_type"] == "calibrated_sigmoid_probability"
    assert 0 <= result["spam_probability"] <= 1
    history = client.get("/api/history?classification=spam").get_json()["items"]
    assert len(history) == 1
    assert "Claim your prize" in history[0]["text"]
    stats = client.get("/api/stats").get_json()
    assert stats["total"] == 1
    assert stats["spam"] == 1


def test_history_filter_search_clear_and_export(client):
    client.post("/api/scan", json={"body": "free prize cash claim"})
    client.post("/api/scan", json={"subject": "Project meeting", "body": "Notes for tomorrow"})
    assert len(client.get("/api/history?q=meeting").get_json()["items"]) == 1
    assert len(client.get("/api/history?classification=safe").get_json()["items"]) == 1
    assert client.get("/api/history?classification=unknown").status_code == 400
    assert client.get("/api/history?limit=invalid").status_code == 400
    export = client.get("/api/history/export")
    assert export.status_code == 200
    assert export.mimetype == "text/csv"
    assert "attachment" in export.headers.get("Content-Disposition", "")
    browser_export = client.get("/history/export")
    assert browser_export.status_code == 200
    assert browser_export.mimetype == "text/csv"
    assert "inboxguard-history.csv" in browser_export.headers.get("Content-Disposition", "")
    dedicated_export = client.get("/api/history/export.csv")
    assert dedicated_export.status_code == 200
    assert dedicated_export.mimetype == "text/csv"
    assert dedicated_export.headers.get("Cache-Control", "").startswith("no-store")
    assert "nosniff" in dedicated_export.headers.get("X-Content-Type-Options", "")
    rows = list(csv.reader(io.StringIO(export.data.decode("utf-8-sig"), newline="")))
    assert rows[0] == [
        "id",
        "subject",
        "sender",
        "message",
        "classification",
        "risk_level",
        "spam_probability",
        "created_at",
    ]
    assert any("free prize cash claim" in row[3] for row in rows[1:])
    response = client.delete("/api/history")
    assert response.get_json()["deleted"] == 2
    assert client.get("/api/stats").get_json()["total"] == 0


def test_url_inspection_flags_deceptive_url_structure():
    urls = inspect_urls({
        "body": "Review https://trusted.example@evil.test:8443/login and https://a.b.c.d.evil.test/path",
        "html_body": "",
        "links": [],
    })
    first = next(item for item in urls if "trusted.example@evil.test" in item["url"])
    second = next(item for item in urls if "a.b.c.d.evil.test" in item["url"])
    assert "URL contains embedded user information" in first["flags"]
    assert "non-standard URL port" in first["flags"]
    assert "excessive subdomain depth" in second["flags"]


def test_url_brand_impersonation_in_subdomain_is_flagged():
    result = analyze_email(
        {"subject": "Account notice", "sender": "Support <help@example.org>", "body": "Visit https://amazon.login.evil.test/verify", "html_body": "", "links": []},
        0.05,
    )
    assert any("resembles the amazon domain" in reason.lower() for reason in result["reasons"])

def test_phishing_campaign_detection_groups_repeated_suspicious_sender_domains():
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    history = [
        {
            "sender": f"Alert {index} <notice@fake-bank.example>",
            "subject": f"Urgent account notice {index}",
            "classification": "phishing_suspicious",
            "risk_level": "high",
            "created_at": timestamp,
        }
        for index in range(3)
    ]
    alerts = detect_phishing_campaigns(history)
    domain_alert = next(
        alert for alert in alerts
        if alert["type"] == "repeated_sender_domain"
    )
    assert domain_alert["indicator"] == "fake-bank.example"
    assert domain_alert["count"] == 3


def test_phishing_campaign_detection_ignores_safe_and_old_scans():
    alerts = detect_phishing_campaigns([
        {
            "sender": "person@example.org",
            "subject": "Routine update",
            "classification": "safe",
            "risk_level": "low",
            "created_at": "2020-01-01T00:00:00+00:00",
        }
        for _ in range(5)
    ])
    assert alerts == []


def test_brand_lookalike_sender_domain_is_flagged():
    result = analyze_email(
        {"subject": "Account notice", "sender": "Support <security@arnazon.com>", "body": "Please review this routine notice.", "html_body": "", "links": []},
        0.05,
    )
    assert any("resembles the amazon domain" in reason.lower() for reason in result["reasons"])
    assert result["risk_score"] >= 15

def test_raw_email_scan_extracts_subject_and_sender(client):
    raw_email = (
        "From: Example Sender <hello@example.org>\r\n"
        "Subject: Project update\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Here are the notes from our meeting."
    )
    response = client.post("/api/scan/raw", json={"raw_email": raw_email})
    assert response.status_code == 200
    assert response.get_json()["classification"] == "safe"
    history = client.get("/api/history").get_json()["items"][0]
    assert history["subject"] == "Project update"
    assert history["sender"] == "Example Sender <hello@example.org>"



def test_raw_email_authentication_headers_and_reply_to_mismatch(client):
    raw_email = (
        "From: Billing <billing@trusted.example>\r\n"
        "Reply-To: attacker@outside.example\r\n"
        "Authentication-Results: mx.example; spf=fail; dkim=pass; dmarc=fail\r\n"
        "Subject: Invoice update\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Please review the attached invoice."
    )
    response = client.post("/api/scan/raw", json={"raw_email": raw_email})
    assert response.status_code == 200
    result = response.get_json()
    authentication = result["email_authentication"]
    assert authentication["spf"] == "fail"
    assert authentication["dkim"] == "pass"
    assert authentication["dmarc"] == "fail"
    assert authentication["status"] == "warning"
    assert any("Reply-To domain" in finding for finding in authentication["findings"])
    assert any("SPF authentication reported fail" in reason for reason in result["reasons"])
    assert result["risk_score"] > 0


def test_raw_email_without_authentication_headers_reports_unavailable(client):
    raw_email = (
        "From: Sender <sender@example.org>\r\n"
        "Subject: Routine note\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "A routine project update."
    )
    response = client.post("/api/scan/raw", json={"raw_email": raw_email})
    assert response.status_code == 200
    authentication = response.get_json()["email_authentication"]
    assert authentication["status"] == "not_available"
    assert authentication["spf"] == "not_provided"
    assert authentication["findings"] == []


def test_raw_email_flags_risky_mime_attachment_metadata(client):
    raw_email = (
        "From: Sender <sender@example.org>\r\n"
        "Subject: Document attached\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/mixed; boundary=BOUNDARY\r\n\r\n"
        "--BOUNDARY\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Please review this document.\r\n"
        "--BOUNDARY\r\n"
        "Content-Type: application/octet-stream; name=invoice.js\r\n"
        "Content-Disposition: attachment; filename=invoice.js\r\n"
        "Content-Transfer-Encoding: base64\r\n\r\n"
        "Y29uc29sZS5sb2coJ2hpJyk=\r\n"
        "--BOUNDARY--\r\n"
    )
    response = client.post("/api/scan/raw", json={"raw_email": raw_email})
    assert response.status_code == 200
    analysis = response.get_json()["attachment_analysis"]
    assert analysis["items"][0]["filename"] == "invoice.js"
    assert analysis["items"][0]["risk"] == "high"
    assert any("invoice.js" in finding for finding in analysis["findings"])


def test_raw_email_without_attachments_returns_empty_attachment_analysis(client):
    raw_email = (
        "From: Sender <sender@example.org>\r\n"
        "Subject: Routine note\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "A routine project update."
    )
    response = client.post("/api/scan/raw", json={"raw_email": raw_email})
    assert response.status_code == 200
    assert response.get_json()["attachment_analysis"]["items"] == []

def test_bulk_upload_and_batch_csv_export(client):
    content = io.BytesIO(
        b"subject,sender,body\r\n"
        b"Prize,offers@example.test,Claim a prize\r\n"
        b"Meeting,team@example.org,Notes for tomorrow\r\n"
    )
    response = client.post(
        "/api/bulk",
        data={"file": (content, "emails.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    batch = response.get_json()
    assert batch["total"] == 2
    assert len(batch["results"]) == 2
    exported = client.get(f"/api/bulk/{batch['batch_id']}/export")
    assert exported.status_code == 200
    assert exported.mimetype == "text/csv"
    rows = list(csv.reader(io.StringIO(exported.data.decode("utf-8-sig"), newline="")))
    assert rows[0][3] == "message"
    assert any("Claim a prize" in row[3] for row in rows[1:])


def test_authentication_and_history_are_user_scoped(app):
    first = app.test_client()
    register(first, "first@example.org")
    first.post("/api/scan", json={"subject": "First", "body": "Message"})

    second = app.test_client()
    register(second, "second@example.org")
    assert second.get("/api/history").get_json()["items"] == []
    assert second.get("/api/stats").get_json()["total"] == 0
    assert second.post("/api/history").status_code == 405


def test_registration_rejects_weak_password_and_duplicate_email(app):
    client = app.test_client()
    weak = client.post(
        "/register",
        data={
            "email": "weak@example.org",
            "password": "short",
            "confirm_password": "short",
        },
    )
    assert weak.status_code == 200
    assert b"at least 12 characters" in weak.data
    register(client, "duplicate@example.org")
    client.post("/logout")
    duplicate = client.post(
        "/register",
        data={
            "email": "duplicate@example.org",
            "password": "secure-demo-password-123",
            "confirm_password": "secure-demo-password-123",
        },
    )
    assert duplicate.status_code == 200
    assert b"already exists" in duplicate.data


def test_domain_rules_can_be_created_listed_and_removed(client):
    created = client.post(
        "/api/domain-rules",
        json={"domain": "www.Evil.Example.com", "action": "block"},
    )
    assert created.status_code == 201
    rule = created.get_json()["item"]
    assert rule["domain"] == "evil.example.com"
    assert rule["action"] == "block"

    listed = client.get("/api/domain-rules")
    assert [item["id"] for item in listed.get_json()["items"]] == [rule["id"]]

    removed = client.delete(f"/api/domain-rules/{rule['id']}")
    assert removed.status_code == 200
    assert removed.get_json()["deleted"] == 1
    assert client.get("/api/domain-rules").get_json()["items"] == []


def test_invalid_domain_rule_is_rejected(client):
    response = client.post(
        "/api/domain-rules",
        json={"domain": "https://evil.example/path", "action": "block"},
    )
    assert response.status_code == 400


def test_personal_blocklist_affects_future_scans(client):
    saved = client.post(
        "/api/domain-rules",
        json={"domain": "evil.example", "action": "block"},
    )
    assert saved.status_code == 201

    scanned = client.post(
        "/api/scan",
        json={
            "sender": "Attacker <alert@evil.example>",
            "subject": "Routine notice",
            "body": "Please review this ordinary message.",
        },
    )
    result = scanned.get_json()
    assert scanned.status_code == 200
    assert result["classification"] == "phishing_suspicious"
    assert "evil.example" in result["domain_rule_matches"]["blocked"]
    assert any("personal blocklist" in reason.lower() for reason in result["reasons"])


def test_bulk_requires_csv_message_column(client):
    file_data = io.BytesIO(b"sender,subject\r\nperson@example.org,Hello\r\n")
    response = client.post(
        "/api/bulk",
        data={"file": (file_data, "emails.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "body" in response.get_json()["error"]


def test_state_changes_require_csrf_outside_test_mode(tmp_path):
    protected_app = create_app(
        {
            "TESTING": False,
            "DATABASE": str(tmp_path / "csrf.sqlite3"),
            "EMAIL_MODEL": StubEmailModel(),
            "EMAIL_SENDER": lambda recipient, subject, body: None,
            "SECRET_KEY": "test-secret",
        }
    )
    client = protected_app.test_client()
    client.get("/register")
    with client.session_transaction() as state:
        token = state["csrf_token"]
    form = {
        "email": "csrf@example.org",
        "password": "secure-demo-password-123",
        "confirm_password": "secure-demo-password-123",
    }
    assert client.post("/register", data=form).status_code == 400
    assert client.post("/register", data={**form, "csrf_token": token}).status_code == 200
    assert client.post("/logout").status_code == 400


def test_deployment_database_and_secure_cookie_environment(monkeypatch, tmp_path):
    database_path = tmp_path / "persistent" / "history.sqlite3"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("INBOXGUARD_DATABASE", str(database_path))
    monkeypatch.setenv("INBOXGUARD_SECURE_COOKIES", "true")
    app = create_app({"TESTING": True, "EMAIL_MODEL": StubEmailModel()})
    assert app.config["DATABASE"] == str(database_path)
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert database_path.is_file()


def test_vercel_requires_database_and_stable_secret(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INBOXGUARD_DATABASE", raising=False)
    monkeypatch.delenv("INBOXGUARD_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_app()

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.example/app")
    with pytest.raises(RuntimeError, match="INBOXGUARD_SECRET_KEY"):
        create_app()


def test_vercel_uses_persistent_database_and_secure_cookie(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("INBOXGUARD_DATABASE", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.example/app")
    monkeypatch.setenv("INBOXGUARD_SECRET_KEY", "a-stable-test-secret")
    monkeypatch.setattr("backend.init_db", lambda database_url: None)
    app = create_app(
        {
            "TESTING": True,
            "EMAIL_MODEL": StubEmailModel(),
        }
    )
    assert app.config["DATABASE"] == "postgresql://user:pass@db.example/app"
    assert app.config["SESSION_COOKIE_SECURE"] is True


def test_missing_email_model_returns_explicit_service_unavailable(tmp_path):
    sent_messages = []
    unavailable_app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "missing-model.sqlite3"),
            "EMAIL_MODEL_PATH": str(tmp_path / "not-trained.joblib"),
            "EMAIL_SENDER": lambda _recipient, _subject, _body: sent_messages.append(
                {
                    "recipient": _recipient,
                    "subject": _subject,
                    "body": _body,
                }
            ),
        }
    )
    unavailable_app.extensions["sent_messages"] = sent_messages
    client = unavailable_app.test_client()
    register(client, "no-model@example.org")
    response = client.post("/api/scan", json={"body": "Message"})
    assert response.status_code == 503
    assert "Run 'python scripts/train_email_model.py'" in response.get_json()["error"]


def test_history_export_neutralizes_spreadsheet_formulas(client):
    client.post(
        "/api/scan",
        json={"subject": "=HYPERLINK(\"https://example.test\")", "body": "A message"},
    )
    client.post("/api/scan", json={"body": "=2+2"})
    response = client.get("/api/history/export")
    assert b"'=HYPERLINK" in response.data
    rows = list(csv.reader(io.StringIO(response.data.decode("utf-8-sig"), newline="")))
    assert any(row[3] == "'=2+2" for row in rows[1:])



def test_scan_pdf_report_is_downloadable_and_account_scoped(client, app):
    scan = client.post(
        "/api/scan",
        json={"sender": "alerts@example.test", "subject": "Prize", "body": "Claim your prize now"},
    ).get_json()
    response = client.get(f"/api/history/{scan['id']}/report.pdf")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")
    assert f"inboxguard-scan-{scan['id']}-report.pdf" in response.headers["Content-Disposition"]
    assert "no-store" in response.headers["Cache-Control"]

    other = app.test_client()
    register(other, "other-analyst@example.org")
    assert other.get(f"/api/history/{scan['id']}/report.pdf").status_code == 404


def test_scan_pdf_report_requires_authentication(app):
    response = app.test_client().get("/api/history/1/report.pdf")
    assert response.status_code == 401

def test_security_activity_is_persisted_and_scoped_to_account(client, app):
    items = client.get("/api/security-events").get_json()["items"]
    assert any(item["event"] == "email_verification" for item in items)
    assert any(item["event"] == "login" and item["outcome"] == "success" for item in items)
    other = app.test_client()
    register(other, "another-user@example.org")
    other_items = other.get("/api/security-events").get_json()["items"]
    assert other_items
    assert {item["id"] for item in items}.isdisjoint({item["id"] for item in other_items})
    assert all("user_id" not in item for item in other_items)

def test_security_activity_requires_authentication(app):
    assert app.test_client().get("/api/security-events").status_code == 401

def test_notifications_are_account_scoped_and_can_be_read_or_dismissed(client, app):
    from backend.database import create_notification
    with app.app_context():
        create_notification(app.config["DATABASE"], 1, "Test warning", "Review activity.", "warning")
    items = client.get("/api/notifications").get_json()["items"]
    item = next(x for x in items if x["title"] == "Test warning")
    assert client.patch(f"/api/notifications/{item['id']}", json={"action":"read"}).status_code == 200
    other = app.test_client()
    register(other, "notification-owner@example.org")
    assert other.patch(f"/api/notifications/{item['id']}", json={"action":"dismiss"}).status_code == 404
    assert client.patch(f"/api/notifications/{item['id']}", json={"action":"dismiss"}).status_code == 200


def test_notifications_require_authentication(app):
    assert app.test_client().get("/api/notifications").status_code == 401
