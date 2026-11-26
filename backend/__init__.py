import csv
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import smtplib
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
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
    send_from_directory,
    session,
    url_for,
)
from werkzeug.exceptions import BadRequest

from backend.database import (
    clear_history,
    delete_domain_rule,
    get_domain_rules,
    save_domain_rule,
    consume_account_token,
    consume_rate_limit,
    create_scan_batch,
    create_user,
    delete_user,
    get_batch_history,
    get_history,
    get_statistics,
    get_user_by_id,
    get_user_by_email,
    mark_email_verified,
    is_integrity_error,
    is_postgresql_database,
    init_db,
    prune_expired_data,
    record_scan,
    set_user_password,
    store_account_token,
    verify_user,
)
from backend.detection_engine import analyze_email, detect_phishing_campaigns
from backend.email_features import analyze_attachments, analyze_email_authentication, parse_raw_email
from backend.model_service import EmailSecurityModel


ROOT_DIR = Path(__file__).resolve().parent.parent
MAX_EMAIL_CHARS = 50_000
MAX_BULK_ROWS = 500
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
SECURITY_LOGGER = logging.getLogger("inboxguard.security")
SECURITY_LOGGER.setLevel(logging.INFO)
SECURITY_LOGGER.propagate = False
if not SECURITY_LOGGER.handlers:
    SECURITY_LOGGER.addHandler(logging.StreamHandler())


def create_app(test_config=None):
    is_vercel = os.environ.get("VERCEL") == "1"
    database_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("INBOXGUARD_DATABASE")
    )
    secret_key = os.environ.get("INBOXGUARD_SECRET_KEY")
    public_url = (
        os.environ.get("INBOXGUARD_PUBLIC_URL")
        or (
            f"https://{os.environ['VERCEL_URL']}"
            if os.environ.get("VERCEL_URL")
            else os.environ.get("RENDER_EXTERNAL_URL")
        )
    )
    if is_vercel and not database_url:
        raise RuntimeError(
            "Set DATABASE_URL to a persistent PostgreSQL database before deploying to Vercel."
        )
    if is_vercel and not secret_key:
        raise RuntimeError(
            "Set INBOXGUARD_SECRET_KEY in the Vercel project environment variables."
        )
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "frontend" / "templates"),
        static_folder=None,
    )
    app.config.from_mapping(
        DATABASE=database_url or str(ROOT_DIR / "instance" / "spam_history.sqlite3"),
        EMAIL_MODEL_PATH=str(ROOT_DIR / "models" / "email_security_model.joblib"),
        SECRET_KEY=secret_key or secrets.token_hex(32),
        PUBLIC_URL=public_url,
        SMTP_HOST=os.environ.get("INBOXGUARD_SMTP_HOST", ""),
        SMTP_PORT=int(os.environ.get("INBOXGUARD_SMTP_PORT", "587")),
        SMTP_USER=os.environ.get("INBOXGUARD_SMTP_USER", ""),
        SMTP_PASSWORD=os.environ.get("INBOXGUARD_SMTP_PASSWORD", ""),
        SMTP_FROM=os.environ.get("INBOXGUARD_SMTP_FROM", ""),
        RETENTION_DAYS=int(os.environ.get("INBOXGUARD_RETENTION_DAYS", "30")),
        MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=is_vercel
        or os.environ.get("INBOXGUARD_SECURE_COOKIES", "").lower()
        in {"1", "true", "yes"},
    )
    if test_config:
        app.config.update(test_config)
    if app.config["RETENTION_DAYS"] < 1:
        raise ValueError("INBOXGUARD_RETENTION_DAYS must be at least 1.")

    def serve_static(filename):
        return send_from_directory(ROOT_DIR / "public" / "static", filename)

    app.add_url_rule(
        "/static/<path:filename>",
        endpoint="static",
        view_func=serve_static,
    )

    if not is_postgresql_database(app.config["DATABASE"]):
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
        if (
            user is None
            or not user["email_verified"]
            or session.get("auth_version") != user["auth_version"]
        ):
            session.clear()
            return None
        return user

    def security_event(event, outcome, user_id=None):
        remote_addr = request.remote_addr or ""
        client_digest = hmac.new(
            app.secret_key.encode(), remote_addr.encode(), hashlib.sha256
        ).hexdigest()[:16]
        SECURITY_LOGGER.info(
            "%s",
            json.dumps(
                {
                    "event": event,
                    "outcome": outcome,
                    "user_id": user_id,
                    "client_id": client_digest,
                },
                separators=(",", ":"),
            ),
        )

    def enforce_rate_limit(scope, identity, limit, window_seconds, message):
        bucket_digest = hmac.new(
            app.secret_key.encode(),
            f"{scope}:{identity}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        allowed, retry_after, _ = consume_rate_limit(
            app.config["DATABASE"],
            bucket_digest,
            limit,
            window_seconds,
            datetime.now(timezone.utc),
        )
        if allowed:
            return None
        security_event(f"rate_limit_{scope}", "rejected")
        response = jsonify(error=message)
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    def mail_is_configured():
        return bool(
            app.config.get("EMAIL_SENDER")
            or (
                app.config["SMTP_HOST"]
                and app.config["SMTP_USER"]
                and app.config["SMTP_PASSWORD"]
                and (app.config["SMTP_FROM"] or app.config["SMTP_USER"])
            )
        )

    def send_account_email(recipient, subject, body):
        custom_sender = app.config.get("EMAIL_SENDER")
        if custom_sender:
            custom_sender(recipient, subject, body)
            return
        if not mail_is_configured():
            raise RuntimeError(
                "Email delivery is not configured. Set the InboxGuard SMTP environment variables."
            )
        message = EmailMessage()
        message["From"] = app.config["SMTP_FROM"] or app.config["SMTP_USER"]
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)
        context = ssl.create_default_context()
        if app.config["SMTP_PORT"] == 465:
            with smtplib.SMTP_SSL(
                app.config["SMTP_HOST"], app.config["SMTP_PORT"], context=context, timeout=15
            ) as server:
                server.login(app.config["SMTP_USER"], app.config["SMTP_PASSWORD"])
                server.send_message(message)
        else:
            with smtplib.SMTP(
                app.config["SMTP_HOST"], app.config["SMTP_PORT"], timeout=15
            ) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(app.config["SMTP_USER"], app.config["SMTP_PASSWORD"])
                server.send_message(message)

    def issue_account_token(user, purpose, endpoint, subject, expires_in):
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        store_account_token(
            app.config["DATABASE"],
            user["id"],
            hashlib.sha256(token.encode("ascii")).hexdigest(),
            purpose,
            (now + expires_in).isoformat(timespec="seconds"),
        )
        root = (app.config["PUBLIC_URL"] or request.url_root).rstrip("/")
        link = f"{root}{url_for(endpoint)}#{token}"
        send_account_email(
            user["email"],
            subject,
            f"Use this one-time link within {int(expires_in.total_seconds() // 3600)} "
            f"hours:\n\n{link}\n\nIf you did not request this, you can ignore this message.",
        )

    last_cleanup = [None]
    cleanup_lock = threading.Lock()

    @app.before_request
    def purge_expired_records():
        now_monotonic = time.monotonic()
        if (
            last_cleanup[0] is not None
            and now_monotonic - last_cleanup[0] < 3600
        ):
            return None
        with cleanup_lock:
            now_monotonic = time.monotonic()
            if (
                last_cleanup[0] is None
                or now_monotonic - last_cleanup[0] >= 3600
            ):
                now = datetime.now(timezone.utc)
                prune_expired_data(
                    app.config["DATABASE"],
                    (now - timedelta(days=app.config["RETENTION_DAYS"])).isoformat(
                        timespec="seconds"
                    ),
                    now.isoformat(timespec="seconds"),
                    (now - timedelta(days=1)).isoformat(timespec="seconds"),
                )
                last_cleanup[0] = now_monotonic
        return None

    @app.after_request
    def secure_response_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.endpoint in {"verify_email", "reset_password"}:
            response.headers["Cache-Control"] = "no-store"
        return response

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
        return {
            "current_user": current_user(),
            "csrf_token": token,
            "retention_days": app.config["RETENTION_DAYS"],
            "public_page": request.endpoint
            in {
                "home",
                "login",
                "register",
                "privacy",
                "resend_verification",
                "verify_email",
                "forgot_password",
                "reset_password",
            },
        }

    @app.before_request
    def protect_state_changes():
        if request.method in {"GET", "HEAD", "OPTIONS"} or app.testing:
            return None
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            security_event("csrf_validation", "rejected")
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
        rules = get_domain_rules(app.config["DATABASE"], user_id)
        sender_match = re.search(r"@([A-Z0-9.-]+\\.[A-Z]{2,})", record.get("sender", ""), re.I)
        sender_domain = sender_match.group(1).lower().strip(".") if sender_match else ""
        observed_domains = {item.get("domain", "").lower().strip(".") for item in result.get("links", [])}
        observed_domains.discard("")
        matched_blocks = [
            rule["domain"] for rule in rules
            if rule["action"] == "block"
            and (
                sender_domain == rule["domain"]
                or sender_domain.endswith("." + rule["domain"])
                or any(domain == rule["domain"] or domain.endswith("." + rule["domain"]) for domain in observed_domains)
            )
        ]
        matched_allows = [
            rule["domain"] for rule in rules
            if rule["action"] == "allow"
            and (
                sender_domain == rule["domain"]
                or sender_domain.endswith("." + rule["domain"])
                or any(domain == rule["domain"] or domain.endswith("." + rule["domain"]) for domain in observed_domains)
            )
        ]
        if matched_blocks:
            result["reasons"] = list(dict.fromkeys(
                result["reasons"] + [f"Domain matches your personal blocklist: {domain}" for domain in matched_blocks]
            ))
            result["risk_score"] = min(100, result["risk_score"] + 30)
            result["risk_level"] = "critical" if result["risk_score"] >= 75 else "high" if result["risk_score"] >= 45 else "medium"
            result["classification"] = "phishing_suspicious"
            result["classification_label"] = "PHISHING / SUSPICIOUS"
            result["recommended_action"] = "This sender or link matches your personal blocklist. Do not interact with it unless you have independently verified it."
        if matched_allows:
            result["reasons"] = list(dict.fromkeys(
                result["reasons"] + [f"Domain is on your personal allowlist: {domain}" for domain in matched_allows]
            ))
        result["domain_rule_matches"] = {
            "blocked": matched_blocks,
            "allowed": matched_allows,
            "note": "Allowlist matches are informational and do not suppress other security warnings."
        }
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

    @app.get("/privacy")
    def privacy():
        return render_template("privacy.html")

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
                limited = enforce_rate_limit(
                    "login",
                    request.remote_addr or "unknown",
                    10,
                    15 * 60,
                    "Too many failed sign-in attempts. Try again later.",
                )
                if limited:
                    return limited
                error = "Email or password is incorrect."
                security_event("login", "failure")
            elif not user["email_verified"]:
                error = "Verify your email before signing in. You can request a new verification link below."
                security_event("login", "unverified", user["id"])
            else:
                session.clear()
                session["user_id"] = user["id"]
                session["auth_version"] = user["auth_version"]
                session["csrf_token"] = secrets.token_urlsafe(32)
                security_event("login", "success", user["id"])
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

    @app.route("/resend-verification", methods=["GET", "POST"])
    def resend_verification():
        if request.method == "POST":
            limited = enforce_rate_limit(
                "resend_verification",
                request.remote_addr or "unknown",
                5,
                60 * 60,
                "Too many requests. Try again later.",
            )
            if limited:
                return limited
            email = request.form.get("email", "").strip().lower()
            user = get_user_by_email(app.config["DATABASE"], email)
            if user and not user["email_verified"]:
                try:
                    issue_account_token(
                        user,
                        "verify_email",
                        "verify_email",
                        "Verify your InboxGuard account",
                        timedelta(hours=24),
                    )
                except (OSError, RuntimeError, smtplib.SMTPException):
                    security_event("verification_email", "delivery_failed", user["id"])
                else:
                    security_event("verification_email", "sent", user["id"])
            return render_template("resend_verification.html", sent=True)
        return render_template("resend_verification.html")

    @app.route("/verify-email", methods=["GET", "POST"])
    def verify_email():
        token = (
            request.args.get("token", "")
            if request.method == "GET"
            else request.form.get("token", "")
        )
        if request.method == "GET":
            return render_template("verify_email.html", token=token)
        user_id = consume_account_token(
            app.config["DATABASE"],
            hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "verify_email",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        if user_id is None:
            security_event("email_verification", "invalid_token")
            return render_template(
                "verify_email.html",
                token="",
                error="This verification link is invalid or has expired.",
            ), 400
        mark_email_verified(app.config["DATABASE"], user_id)
        security_event("email_verification", "success", user_id)
        return render_template("verify_email.html", token="", verified=True)

    @app.route("/forgot-password", methods=["GET", "POST"])
    def forgot_password():
        if request.method == "POST":
            limited = enforce_rate_limit(
                "password_reset_request",
                request.remote_addr or "unknown",
                5,
                60 * 60,
                "Too many requests. Try again later.",
            )
            if limited:
                return limited
            email = request.form.get("email", "").strip().lower()
            user = get_user_by_email(app.config["DATABASE"], email)
            if user and user["email_verified"]:
                try:
                    issue_account_token(
                        user,
                        "reset_password",
                        "reset_password",
                        "Reset your InboxGuard password",
                        timedelta(hours=1),
                    )
                except (OSError, RuntimeError, smtplib.SMTPException):
                    security_event("password_reset_email", "delivery_failed", user["id"])
                else:
                    security_event("password_reset_email", "sent", user["id"])
            return render_template("forgot_password.html", sent=True)
        return render_template("forgot_password.html")

    @app.route("/reset-password", methods=["GET", "POST"])
    def reset_password():
        token = (
            request.args.get("token", "")
            if request.method == "GET"
            else request.form.get("token", "")
        )
        error = None
        if request.method == "POST":
            password = request.form.get("password", "")
            confirmation = request.form.get("confirm_password", "")
            if len(password) < 12:
                error = "Use a password with at least 12 characters."
            elif len(password) > 256:
                error = "Password must be 256 characters or fewer."
            elif password != confirmation:
                error = "The password confirmation does not match."
            else:
                user_id = consume_account_token(
                    app.config["DATABASE"],
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    "reset_password",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                if user_id is None:
                    security_event("password_reset", "invalid_token")
                    return render_template(
                        "reset_password.html",
                        token="",
                        error="This password-reset link is invalid or has expired.",
                    ), 400
                else:
                    set_user_password(app.config["DATABASE"], user_id, password)
                    security_event("password_reset", "success", user_id)
                    return render_template("reset_password.html", token="", reset=True)
        return render_template("reset_password.html", token=token, error=error)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user():
            return redirect(url_for("dashboard"))
        error = None
        if request.method == "POST":
            limited = enforce_rate_limit(
                "registration",
                request.remote_addr or "unknown",
                5,
                60 * 60,
                "Too many registration attempts. Try again later.",
            )
            if limited:
                return limited
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirmation = request.form.get("confirm_password", "")
            if not mail_is_configured():
                error = (
                    "Account creation is temporarily unavailable because email "
                    "delivery is not configured. The site owner must add "
                    "INBOXGUARD_SMTP_HOST, INBOXGUARD_SMTP_USER, and "
                    "INBOXGUARD_SMTP_PASSWORD in Vercel Settings > Environment "
                    "Variables, then redeploy."
                )
            elif not EMAIL_RE.fullmatch(email):
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
                except Exception as error:
                    if not is_integrity_error(error):
                        raise
                    error = "An account with that email already exists."
                    return render_template("register.html", error=error)
                user = get_user_by_id(app.config["DATABASE"], user_id)
                try:
                    issue_account_token(
                        user,
                        "verify_email",
                        "verify_email",
                        "Verify your InboxGuard account",
                        timedelta(hours=24),
                    )
                except (OSError, RuntimeError, smtplib.SMTPException):
                    security_event("registration", "verification_delivery_failed", user_id)
                    return render_template(
                        "register.html",
                        error="Your account was created, but the verification email could not be sent. Request another link later.",
                    ), 503
                security_event("registration", "verification_sent", user_id)
                return render_template("register.html", sent=True)
        return render_template("register.html", error=error)

    @app.post("/logout")
    @login_required
    def logout():
        security_event("logout", "success", session.get("user_id"))
        session.clear()
        return redirect(url_for("home"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/account", methods=["GET", "POST"])
    @login_required
    def account():
        user = current_user()
        if request.method == "POST":
            confirmation = request.form.get("confirm_delete", "")
            password = request.form.get("password", "")
            verified_user = verify_user(
                app.config["DATABASE"], user["email"], password
            )
            if (
                confirmation != "DELETE"
                or verified_user is None
                or verified_user["id"] != user["id"]
            ):
                return render_template(
                    "account.html",
                    user=user,
                    error="Confirm deletion by typing DELETE and entering your current password.",
                ), 400
            user_id = user["id"]
            delete_user(app.config["DATABASE"], user_id)
            security_event("account_delete", "success", user_id)
            session.clear()
            return redirect(url_for("home"))
        return render_template("account.html", user=user)

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
        limited = enforce_rate_limit(
            "email_scan",
            str(current_user()["id"]),
            5,
            60,
            "Too many scans. Try again in a minute.",
        )
        if limited:
            return limited
        unavailable = email_model_unavailable_response()
        if unavailable:
            return unavailable
        payload = request.get_json(silent=True)
        record, error = validate_email_record(payload)
        if error:
            return jsonify(error=error), 400
        result = scan_record(record, current_user()["id"])
        security_event("email_scan", "success", current_user()["id"])
        return jsonify(result)

    @app.post("/api/scan/raw")
    @login_required
    def scan_raw():
        limited = enforce_rate_limit(
            "email_scan",
            str(current_user()["id"]),
            5,
            60,
            "Too many scans. Try again in a minute.",
        )
        if limited:
            return limited
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
        parsed_email = parse_raw_email(raw_email)
        record, error = validate_email_record(parsed_email)
        if error:
            return jsonify(error=error), 400
        user_id = current_user()["id"]
        result = scan_record(record, user_id, scan_kind="raw_email")
        authentication = analyze_email_authentication(
            parsed_email.get("headers", {}), record.get("sender", "")
        )
        attachment_analysis = analyze_attachments(parsed_email.get("attachments", []))
        result["email_authentication"] = authentication
        result["attachment_analysis"] = attachment_analysis
        additional_findings = authentication["findings"] + attachment_analysis["findings"]
        additional_risk = authentication["risk_points"] + attachment_analysis["risk_points"]
        if additional_findings:
            result["reasons"] = list(dict.fromkeys(result["reasons"] + additional_findings))
        if additional_risk:
            result["risk_score"] = min(100, result["risk_score"] + additional_risk)
            result["risk_level"] = (
                "critical" if result["risk_score"] >= 75 else
                "high" if result["risk_score"] >= 45 else
                "medium" if result["risk_score"] >= 20 else "low"
            )
        security_event("email_scan", "success", user_id)
        return jsonify(result)

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
        user_id = current_user()["id"]
        deleted = clear_history(app.config["DATABASE"], user_id)
        security_event("history_clear", "success", user_id)
        return jsonify(deleted=deleted)

    @app.get("/api/history/export")
    @app.get("/api/history/export.csv")
    @app.get("/history/export")
    @login_required
    def export_history():
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            (
                "id",
                "subject",
                "sender",
                "message",
                "classification",
                "risk_level",
                "spam_probability",
                "created_at",
            )
        )
        for item in get_history(
            app.config["DATABASE"], current_user()["id"], limit=100_000
        ):
            writer.writerow(
                (
                    item["id"],
                    safe_csv_value(item["subject"]),
                    safe_csv_value(item["sender"]),
                    safe_csv_value(item["text"]),
                    safe_csv_value(item["classification"]),
                    safe_csv_value(item["risk_level"]),
                    item["confidence"],
                    item["created_at"],
                )
            )
        security_event("history_export", "success", current_user()["id"])
        response = send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name="inboxguard-history.csv",
            max_age=0,
        )
        response.headers["Content-Disposition"] = 'attachment; filename="inboxguard-history.csv"'
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/domain-rules")
    @login_required
    def list_domain_rules():
        return jsonify(items=get_domain_rules(app.config["DATABASE"], current_user()["id"]))

    @app.post("/api/domain-rules")
    @login_required
    def create_domain_rule():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a domain and an allow/block action."), 400
        domain = str(payload.get("domain", "")).strip().lower().rstrip(".")
        action = str(payload.get("action", "")).strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if len(domain) > 253 or not re.fullmatch(
            r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}",
            domain,
        ):
            return jsonify(error="Enter a valid domain name, without a URL or path."), 400
        if action not in {"allow", "block"}:
            return jsonify(error="Action must be 'allow' or 'block'."), 400
        rule = save_domain_rule(
            app.config["DATABASE"], current_user()["id"], domain, action
        )
        security_event("domain_rule_saved", action, current_user()["id"])
        return jsonify(item=rule), 201

    @app.delete("/api/domain-rules/<int:rule_id>")
    @login_required
    def remove_domain_rule(rule_id):
        deleted = delete_domain_rule(
            app.config["DATABASE"], current_user()["id"], rule_id
        )
        if not deleted:
            return jsonify(error="Domain rule not found."), 404
        security_event("domain_rule_deleted", "success", current_user()["id"])
        return jsonify(deleted=deleted)

    @app.get("/api/stats")
    @login_required
    def stats():
        user_id = current_user()["id"]
        statistics = get_statistics(app.config["DATABASE"], user_id)
        recent_history = get_history(app.config["DATABASE"], user_id, limit=500)
        statistics["campaign_alerts"] = detect_phishing_campaigns(recent_history)
        return jsonify(statistics)

    @app.post("/api/bulk")
    @login_required
    def bulk_scan():
        limited = enforce_rate_limit(
            "email_scan",
            str(current_user()["id"]),
            5,
            60,
            "Too many scans. Try again in a minute.",
        )
        if limited:
            return limited
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
        security_event("bulk_email_scan", "success", user_id)
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
            (
                "id",
                "subject",
                "sender",
                "message",
                "classification",
                "risk_level",
                "spam_probability",
                "created_at",
            )
        )
        for item in batch["items"]:
            writer.writerow(
                (
                    item["id"],
                    safe_csv_value(item["subject"]),
                    safe_csv_value(item["sender"]),
                    safe_csv_value(item["text"]),
                    safe_csv_value(item["classification"]),
                    safe_csv_value(item["risk_level"]),
                    item["confidence"],
                    item["created_at"],
                )
            )
        security_event("bulk_export", "success", current_user()["id"])
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
