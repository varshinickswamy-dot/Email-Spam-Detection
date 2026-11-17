import csv
import io
import os
import sqlite3
import re
import secrets
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.exceptions import BadRequest

from backend.database import (
    clear_history,
    create_scan_batch,
    create_user,
    get_batch_history,
    get_history,
    get_statistics,
    get_user_by_id,
    get_user_by_email,
    init_db,
    record_scan,
    verify_user,
)
from backend.detection_engine import analyze_email
from backend.email_features import parse_raw_email
from backend.model_service import EmailSecurityModel


ROOT_DIR = Path(__file__).resolve().parent.parent
MAX_EMAIL_CHARS = 50_000
MAX_BULK_ROWS = 500
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


def create_app(test_config=None):
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "frontend" / "templates"),
        static_folder=str(ROOT_DIR / "frontend" / "static"),
    )
    app.config.from_mapping(
        DATABASE=os.environ.get(
            "INBOXGUARD_DATABASE",
            str(ROOT_DIR / "instance" / "spam_history.sqlite3"),
        ),
        EMAIL_MODEL_PATH=str(ROOT_DIR / "models" / "email_security_model.joblib"),
        SECRET_KEY=os.environ.get("INBOXGUARD_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get(
            "INBOXGUARD_SECURE_COOKIES", ""
        ).lower()
        in {"1", "true", "yes"},
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    init_db(app.config["DATABASE"])
    email_model = app.config.get("EMAIL_MODEL")
    if email_model is None:
        try:
            email_model = EmailSecurityModel(app.config["EMAIL_MODEL_PATH"])
        except FileNotFoundError as error:
            app.extensions["email_model_error"] = str(error)
    app.extensions["email_model"] = email_model

    def current_user():
        user_id = session.get("user_id")
        if not user_id:
            return None
        user = get_user_by_id(app.config["DATABASE"], user_id)
        if user is None:
            session.clear()
        return user

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if current_user() is None:
                if request.path.startswith("/api/"):
                    return jsonify(error="Please sign in to continue."), 401
                return redirect(url_for("login", next=request.full_path.rstrip("?")))
            return view(*args, **kwargs)

        return wrapped

    @app.context_processor
    def inject_template_state():
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return {"current_user": current_user(), "csrf_token": token}

    @app.before_request
    def protect_state_changes():
        if request.method in {"GET", "HEAD", "OPTIONS"} or app.testing:
            return None
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            if request.path.startswith("/api/"):
                return jsonify(error="The request token is missing or invalid. Refresh and retry."), 400
            abort(400, description="The request token is missing or invalid. Refresh and retry.")
        return None

    def email_model_unavailable_response():
        model = app.extensions.get("email_model")
        if model is None:
            return (
                jsonify(
                    error=app.extensions.get(
                        "email_model_error",
                        "The email model is unavailable. Train the email model before scanning.",
                    )
                ),
                503,
            )
        return None

    def validate_email_record(payload):
        if not isinstance(payload, dict):
            return None, "Send an object containing an email subject and body."
        fields = ("subject", "body", "sender", "html_body")
        if any(not isinstance(payload.get(key, ""), str) for key in fields):
            return None, "Subject, body, sender, and HTML body must be text."
        record = {
            key: payload.get(key, "").strip()
            for key in fields
        }
        links = payload.get("links", [])
        if isinstance(links, str):
            links = [link.strip() for link in links.splitlines() if link.strip()]
        if not isinstance(links, list) or len(links) > 50:
            return None, "Links must be a list containing no more than 50 items."
        if any(
            not isinstance(item, str)
            and not (
                isinstance(item, dict)
                and isinstance(item.get("url", ""), str)
                and isinstance(item.get("text", ""), str)
            )
            for item in links
        ):
            return None, "Each link must be text or an object containing text URL fields."
        record["links"] = links
        if not any(record[key] for key in ("subject", "body", "html_body")):
            return None, "Enter an email subject, body, or HTML content to analyze."
        if any(len(record[key]) > MAX_EMAIL_CHARS for key in fields):
            return None, f"Email fields cannot exceed {MAX_EMAIL_CHARS:,} characters."
        return record, None

    def safe_csv_value(value):
        value = str(value or "")
        if value[:1] in {"=", "+", "-", "@"} or value.startswith(("\t", "\r")):
            return f"'{value}"
        return value

    def scan_record(record, user_id, batch_id=None, scan_kind="single"):
        model = app.extensions["email_model"]
        probability = model.predict_spam_probability(record)
        result = analyze_email(record, probability, model.threshold)
        result["id"] = record_scan(
            app.config["DATABASE"],
            user_id,
            record,
            result,
            batch_id=batch_id,
            scan_kind=scan_kind,
        )
        return result

    @app.get("/")
    def home():
        if current_user():
            return redirect(url_for("dashboard"))
        return render_template("landing.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user():
            return redirect(url_for("dashboard"))
        error = None
        if request.method == "POST":
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            user = verify_user(app.config["DATABASE"], email, password)
            if user is None:
                error = "Email or password is incorrect."
            else:
                session.clear()
                session["user_id"] = user["id"]
                session["csrf_token"] = secrets.token_urlsafe(32)
                target = request.args.get("next", "")
                parsed_target = urlsplit(target)
                if (
                    not target.startswith("/")
                    or target.startswith("//")
                    or parsed_target.netloc
                ):
                    target = url_for("dashboard")
                return redirect(target)
        return render_template("login.html", error=error)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user():
            return redirect(url_for("dashboard"))
        error = None
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirmation = request.form.get("confirm_password", "")
            if not EMAIL_RE.fullmatch(email):
                error = "Enter a valid email address."
            elif len(password) < 12:
                error = "Use a password with at least 12 characters."
            elif len(password) > 256:
                error = "Password must be 256 characters or fewer."
            elif password != confirmation:
                error = "The password confirmation does not match."
            elif get_user_by_email(app.config["DATABASE"], email):
                error = "An account with that email already exists."
            else:
                try:
                    user_id = create_user(app.config["DATABASE"], email, password)
                except sqlite3.IntegrityError:
                    error = "An account with that email already exists."
                    return render_template("register.html", error=error)
                session.clear()
                session["user_id"] = user_id
                session["csrf_token"] = secrets.token_urlsafe(32)
                return redirect(url_for("dashboard"))
        return render_template("register.html", error=error)

    @app.post("/logout")
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/scanner")
    @login_required
    def scanner():
        return render_template("scanner.html")

    @app.get("/bulk")
    @login_required
    def bulk_scanner():
        return render_template("bulk.html")

    @app.get("/history")
    @login_required
    def history_page():
        return render_template("history.html")

    @app.get("/about")
    @login_required
    def about():
        return render_template("about.html")

    @app.post("/api/scan")
    @login_required
    def scan():
        unavailable = email_model_unavailable_response()
        if unavailable:
            return unavailable
        payload = request.get_json(silent=True)
        record, error = validate_email_record(payload)
        if error:
            return jsonify(error=error), 400
        result = scan_record(record, current_user()["id"])
        return jsonify(result)

    @app.post("/api/scan/raw")
    @login_required
    def scan_raw():
        unavailable = email_model_unavailable_response()
        if unavailable:
            return unavailable
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("raw_email"), str):
            return jsonify(error="Send a raw_email string in the JSON request."), 400
        raw_email = payload["raw_email"]
        if not raw_email.strip():
            return jsonify(error="Paste a complete email to analyze."), 400
        if len(raw_email) > MAX_EMAIL_CHARS:
            return jsonify(error=f"Raw email cannot exceed {MAX_EMAIL_CHARS:,} characters."), 400
        record, error = validate_email_record(parse_raw_email(raw_email))
        if error:
            return jsonify(error=error), 400
        return jsonify(scan_record(record, current_user()["id"], scan_kind="raw_email"))

    @app.get("/api/model-info")
    @login_required
    def model_info():
        model = app.extensions.get("email_model")
        unavailable = email_model_unavailable_response()
        if unavailable:
            return unavailable
        return jsonify(model.info)

    @app.get("/api/history")
    @login_required
    def history():
        query = request.args.get("q", "").strip()[:200]
        classification = request.args.get("classification", "").strip().lower()
        allowed = {"", "safe", "promotional", "spam", "phishing_suspicious", "uncertain"}
        if classification not in allowed:
            return jsonify(error="Unknown classification filter."), 400
        try:
            limit = max(1, min(int(request.args.get("limit", 100)), 500))
        except ValueError:
            return jsonify(error="limit must be an integer."), 400
        return jsonify(
            items=get_history(
                app.config["DATABASE"],
                current_user()["id"],
                query=query,
                classification=classification,
                limit=limit,
            )
        )

    @app.delete("/api/history")
    @login_required
    def delete_history():
        return jsonify(deleted=clear_history(app.config["DATABASE"], current_user()["id"]))

    @app.get("/api/history/export")
    @login_required
    def export_history():
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            ("id", "subject", "sender", "classification", "risk_level", "spam_probability", "created_at")
        )
        for item in get_history(
            app.config["DATABASE"], current_user()["id"], limit=100_000
        ):
            writer.writerow(
                (
                    item["id"],
                    safe_csv_value(item["subject"]),
                    safe_csv_value(item["sender"]),
                    safe_csv_value(item["classification"]),
                    safe_csv_value(item["risk_level"]),
                    item["confidence"],
                    item["created_at"],
                )
            )
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="inboxguard-history.csv",
        )

    @app.get("/api/stats")
    @login_required
    def stats():
        return jsonify(get_statistics(app.config["DATABASE"], current_user()["id"]))

    @app.post("/api/bulk")
    @login_required
    def bulk_scan():
        unavailable = email_model_unavailable_response()
        if unavailable:
            return unavailable
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a CSV file to scan."), 400
        if Path(upload.filename).suffix.lower() != ".csv":
            return jsonify(error="Upload a .csv file."), 400
        try:
            content = upload.stream.read(5 * 1024 * 1024 + 1)
            if len(content) > 5 * 1024 * 1024:
                return jsonify(error="CSV files cannot exceed 5 MB."), 413
            decoded = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(decoded, newline=""))
            fields = {field.strip().lower() for field in (reader.fieldnames or [])}
            if not fields.intersection({"body", "message"}):
                return jsonify(error="CSV must include a 'body' or 'message' column."), 400
            records = []
            row_errors = []
            for row_number, row in enumerate(reader, start=2):
                if row_number > MAX_BULK_ROWS + 1:
                    return jsonify(error=f"CSV cannot contain more than {MAX_BULK_ROWS} data rows."), 400
                normalized = {
                    str(key or "").strip().lower(): value or ""
                    for key, value in row.items()
                    if key is not None
                }
                record, error = validate_email_record(
                    {
                        "subject": normalized.get("subject", ""),
                        "body": normalized.get("body", normalized.get("message", "")),
                        "sender": normalized.get("sender", ""),
                        "html_body": normalized.get("html_body", ""),
                        "links": normalized.get("links", ""),
                    }
                )
                if error:
                    row_errors.append({"row": row_number, "error": error})
                    continue
                records.append(record)
        except (UnicodeDecodeError, csv.Error, BadRequest) as error:
            return jsonify(error=f"Could not read the CSV file: {error}"), 400
        if not records:
            return jsonify(error="The CSV contains no valid email rows.", row_errors=row_errors[:20]), 400

        user_id = current_user()["id"]
        batch_id = uuid4().hex
        create_scan_batch(app.config["DATABASE"], batch_id, user_id, len(records))
        results = [
            scan_record(record, user_id, batch_id=batch_id, scan_kind="bulk")
            for record in records
        ]
        return jsonify(
            batch_id=batch_id,
            total=len(results),
            results=results,
            row_errors=row_errors[:20],
            skipped_rows=len(row_errors),
        )

    @app.get("/api/bulk/<batch_id>/export")
    @login_required
    def export_bulk(batch_id):
        batch = get_batch_history(
            app.config["DATABASE"], current_user()["id"], batch_id
        )
        if batch is None:
            abort(404)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            ("id", "subject", "sender", "classification", "risk_level", "spam_probability", "created_at")
        )
        for item in batch["items"]:
            writer.writerow(
                (
                    item["id"],
                    safe_csv_value(item["subject"]),
                    safe_csv_value(item["sender"]),
                    safe_csv_value(item["classification"]),
                    safe_csv_value(item["risk_level"]),
                    item["confidence"],
                    item["created_at"],
                )
            )
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"inboxguard-batch-{batch_id[:8]}.csv",
        )

    @app.errorhandler(413)
    def request_too_large(error):
        return jsonify(
            error="Request is too large.",
            detail=error.description,
        ), 413

    return app
