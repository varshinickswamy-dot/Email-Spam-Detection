import csv
import io

import pytest

from backend import create_app


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
    return create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "history.sqlite3"),
            "EMAIL_MODEL": StubEmailModel(),
        }
    )


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
    assert response.status_code == 302
    return client


def test_public_pages_auth_and_model_info(app):
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 200
    assert client.get("/dashboard").status_code == 302
    assert client.get("/api/model-info").status_code == 401
    register(client, "reader@example.org")
    assert client.get("/api/model-info").get_json()["model_name"] == "Test email model"


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
    assert client.post("/register", data={**form, "csrf_token": token}).status_code == 302
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
    unavailable_app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "missing-model.sqlite3"),
            "EMAIL_MODEL_PATH": str(tmp_path / "not-trained.joblib"),
        }
    )
    client = unavailable_app.test_client()
    register(client, "no-model@example.org")
    response = client.post("/api/scan", json={"body": "Message"})
    assert response.status_code == 503
    assert "Run 'python scripts/train_email_model.py'" in response.get_json()["error"]


def test_history_export_neutralizes_spreadsheet_formulas(client):
    client.post(
        "/api/scan",
        json={"subject": "=HYPERLINK(\"https://example.test\")", "body": "=2+2"},
    )
    response = client.get("/api/history/export")
    assert b"'=HYPERLINK" in response.data
    assert b"',=2+2" not in response.data
    assert b"'=2+2" in response.data
