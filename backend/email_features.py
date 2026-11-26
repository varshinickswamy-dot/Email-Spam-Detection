import html
import ipaddress
import re
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>'\"()\[\]{}]+"
)
HTML_PATTERN = re.compile(r"(?is)<\s*(?:html|body|div|p|a|table|script|style)\b")
URL_SHORTENERS = {
    "bit.ly",
    "buff.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rb.gy",
    "rebrand.ly",
    "t.co",
    "tinyurl.com",
    "shorturl.at",
}
SUSPICIOUS_TLDS = {".zip", ".mov", ".click", ".top", ".work", ".country", ".gq"}
ATTACHMENT_PATTERN = re.compile(
    r"(?i)\b[\w.-]+\.(?:exe|scr|js|vbs|bat|cmd|iso|zip|html|hta)\b"
)


class _EmailHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.anchors = []
        self._active_anchor = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}:
            self._skip_depth += 1
        if tag == "a":
            href = dict(attrs).get("href", "")
            self._active_anchor = {"url": href, "text": ""}

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "a" and self._active_anchor is not None:
            self.anchors.append(self._active_anchor)
            self._active_anchor = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        value = data.strip()
        if value:
            self.text_parts.append(value)
            if self._active_anchor is not None:
                self._active_anchor["text"] += f" {value}"


def parse_html(value):
    parser = _EmailHTMLParser()
    parser.feed(str(value or ""))
    parser.close()
    return " ".join(parser.text_parts), parser.anchors


def normalize_text(value):
    text = html.unescape(str(value or ""))
    if HTML_PATTERN.search(text):
        text, _ = parse_html(text)
    text = text.lower()
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"[^a-z0-9\s:/.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_url(value):
    url = str(value or "").strip().rstrip(".,;:!?)]}")
    if not url or any(ord(character) < 32 for character in url) or any(character.isspace() for character in url):
        return ""
    if url.lower().startswith("www."):
        url = f"http://{url}"
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            return ""
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return ""
        if len(host) > 253 or any(not label or len(label) > 63 for label in host.split(".")):
            return ""
        return url, host
    except ValueError:
        return ""


def _is_ip_address(host):
    try:
        ipaddress.ip_address(str(host).strip("[]"))
        return True
    except ValueError:
        return False


def inspect_urls(record):
    body = str(record.get("body", "") or "")
    html_body = str(record.get("html_body", "") or "")
    _, html_anchors = parse_html(html_body)
    if HTML_PATTERN.search(body):
        _, body_anchors = parse_html(body)
        html_anchors.extend(body_anchors)

    candidates = []
    for candidate in URL_PATTERN.findall(body):
        candidates.append({"url": candidate, "text": ""})
    candidates.extend(html_anchors)
    for candidate in URL_PATTERN.findall(html_body):
        candidates.append({"url": candidate, "text": ""})
    supplied = record.get("links", []) or []
    if isinstance(supplied, str):
        supplied = [supplied]
    if isinstance(supplied, (list, tuple)):
        for item in supplied[:50]:
            if isinstance(item, str):
                candidates.append({"url": item, "text": ""})
            elif isinstance(item, dict):
                candidates.append(
                    {"url": item.get("url", ""), "text": item.get("text", "")}
                )

    urls = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_url(candidate.get("url"))
        if not normalized:
            continue
        url, host = normalized
        if url in seen:
            continue
        seen.add(url)
        flags = []
        try:
            ipaddress.ip_address(host.strip("[]"))
            flags.append("IP-address host")
        except ValueError:
            pass
        if host in URL_SHORTENERS:
            flags.append("URL shortener")
        if "xn--" in host:
            flags.append("internationalized/punycode domain")
        if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            flags.append("uncommon high-risk TLD")
        if url.lower().startswith("http://"):
            flags.append("unencrypted HTTP link")
        try:
            parsed_url = urlsplit(url)
            if parsed_url.username is not None or parsed_url.password is not None:
                flags.append("URL contains embedded user information")
            if parsed_url.port and parsed_url.port not in {80, 443}:
                flags.append("non-standard URL port")
        except ValueError:
            flags.append("malformed URL authority")
        if not _is_ip_address(host) and len(host.split(".")) >= 5:
            flags.append("excessive subdomain depth")
        anchor_text = str(candidate.get("text", "") or "").strip()
        displayed_hosts = re.findall(
            r"(?i)(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})",
            anchor_text,
        )
        if displayed_hosts:
            displayed_host = displayed_hosts[0].lower().removeprefix("www.")
            if displayed_host != host and not host.endswith(f".{displayed_host}"):
                flags.append("displayed domain differs from link destination")
        urls.append(
            {
                "url": url,
                "domain": host,
                "anchor_text": anchor_text[:200],
                "flags": flags,
                "suspicious": bool(flags),
            }
        )
    return urls


def _sender_parts(sender):
    display, address = parseaddr(str(sender or ""))
    domain = address.rsplit("@", 1)[-1].lower() if "@" in address else ""
    return display, address, domain


def prepare_email(record):
    if not isinstance(record, dict):
        raise TypeError("Email must be provided as an object.")
    subject = str(record.get("subject", "") or "")[:500]
    body = str(record.get("body", "") or "")[:50_000]
    html_body = str(record.get("html_body", "") or "")[:50_000]
    sender = str(record.get("sender", "") or "")[:320]
    prepared = {
        "subject": subject,
        "body": body,
        "html_body": html_body,
        "sender": sender,
        "links": record.get("links", []),
    }
    clean_body = body
    if HTML_PATTERN.search(clean_body):
        clean_body, _ = parse_html(clean_body)
    if html_body:
        html_text, _ = parse_html(html_body)
        clean_body = f"{clean_body}\n{html_text}".strip()
    url_items = inspect_urls(prepared)
    domains = " ".join(item["domain"] for item in url_items)
    display, address, sender_domain = _sender_parts(sender)
    return {
        "subject": normalize_text(subject),
        "body": normalize_text(f"{clean_body}\n{domains}"),
        "sender": normalize_text(f"{display} {address} {sender_domain}"),
        "html_body": " ".join([html_body, "html"] if html_body or HTML_PATTERN.search(body) else []),
        "links": " ".join(f"{item['domain']} {item['url']}" for item in url_items),
        "sender_domain": sender_domain,
        "sender_address": address,
        "url_items": url_items,
        "raw_subject": subject,
        "raw_body": body,
        "raw_html_body": html_body,
        "raw_sender": sender,
    }


def prepare_model_frame(records):
    rows = []
    for record in records:
        prepared = prepare_email(record)
        rows.append(
            {
                "subject": prepared["subject"],
                "body": prepared["body"],
                "sender": prepared["sender"],
                "html_body": prepared["html_body"],
                "links": prepared["links"],
            }
        )
    return pd.DataFrame(
        rows, columns=["subject", "body", "sender", "html_body", "links"]
    )


def analyze_email_authentication(headers, sender=""):
    """Summarize sender-authentication headers without treating missing data as failure."""
    headers = headers or {}
    auth_results = str(headers.get("authentication_results", "") or "").lower()
    statuses = {}
    for mechanism in ("spf", "dkim", "dmarc"):
        matches = re.findall(r"(?<![a-z0-9_-])" + mechanism + r"\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)", auth_results)
        statuses[mechanism] = matches[0] if matches else "not_provided"
    findings = []
    risk_points = 0
    for mechanism, status in statuses.items():
        if status in {"fail", "softfail", "permerror"}:
            findings.append(f"{mechanism.upper()} authentication reported {status}.")
            risk_points += 12 if mechanism == "dmarc" else 8
    _, _, from_domain = _sender_parts(sender)
    reply_to = str(headers.get("reply_to", "") or "")
    _, _, reply_domain = _sender_parts(reply_to)
    if from_domain and reply_domain and from_domain != reply_domain:
        findings.append(f"Reply-To domain {reply_domain} differs from the visible sender domain {from_domain}.")
        risk_points += 15
    return {
        "available": bool(auth_results or reply_to or headers.get("return_path")),
        "spf": statuses["spf"], "dkim": statuses["dkim"], "dmarc": statuses["dmarc"],
        "reply_to": reply_to, "return_path": str(headers.get("return_path", "") or ""),
        "findings": findings, "risk_points": min(35, risk_points),
        "status": "warning" if findings else ("provided" if auth_results else "not_available"),
        "note": "Authentication-Results headers are supplied by the receiving mail system; pasted or forged headers cannot independently prove authenticity.",
    }

def analyze_attachments(attachments):
    """Flag risky attachment types from MIME metadata; never execute or extract files."""
    dangerous_extensions = {".exe", ".scr", ".js", ".jse", ".vbs", ".vbe", ".bat", ".cmd", ".ps1", ".hta", ".iso", ".img", ".lnk", ".jar", ".msi", ".docm", ".xlsm"}
    reviewed = []
    findings = []
    for item in (attachments or [])[:50]:
        filename = str(item.get("filename", "unnamed attachment"))[:255]
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        dangerous = extension in dangerous_extensions
        try:
            size_bytes = max(0, int(item.get("size_bytes", 0) or 0))
        except (TypeError, ValueError):
            size_bytes = 0
        reviewed.append({
            "filename": filename,
            "content_type": str(item.get("content_type", "unknown"))[:120],
            "size_bytes": size_bytes,
            "risk": "high" if dangerous else "review",
        })
        if dangerous:
            findings.append(f"Potentially dangerous attachment type: {filename}.")
    return {"items": reviewed, "findings": findings, "risk_points": min(20, len(findings) * 10)}

def parse_raw_email(raw_email):
    message = BytesParser(policy=policy.default).parsebytes(
        str(raw_email).encode("utf-8", errors="replace")
    )
    plain_parts = []
    html_parts = []
    attachments = []
    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment" or part.get_filename():
            payload = part.get_payload(decode=True) or b""
            attachments.append({
                "filename": str(part.get_filename() or "unnamed attachment")[:255],
                "content_type": str(part.get_content_type() or "application/octet-stream")[:120],
                "size_bytes": len(payload),
            })
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeDecodeError, KeyError):
            payload = part.get_payload(decode=True) or b""
            content = payload.decode(
                part.get_content_charset() or "utf-8", errors="replace"
            )
        if part.get_content_type() == "text/html":
            html_parts.append(content)
        else:
            plain_parts.append(content)
    return {
        "subject": str(message.get("subject", "") or ""),
        "sender": str(message.get("from", "") or ""),
        "body": "\n".join(plain_parts or html_parts),
        "html_body": "\n".join(html_parts),
        "links": [],
        "attachments": attachments[:50],
        "headers": {
            "authentication_results": " ".join(str(value) for value in message.get_all("Authentication-Results", [])),
            "reply_to": str(message.get("Reply-To", "") or ""),
            "return_path": str(message.get("Return-Path", "") or ""),
        },
    }


def structural_features(values):
    if isinstance(values, pd.DataFrame):
        rows = values.itertuples(index=False, name=None)
    else:
        rows = values
    features = []
    for subject, body, sender, html_body, links in rows:
        subject = str(subject or "")
        body = str(body or "")
        sender = str(sender or "")
        links = str(links or "")
        text = f"{subject} {body}"
        upper_count = sum(character.isupper() for character in text)
        alpha_count = sum(character.isalpha() for character in text)
        urls = URL_PATTERN.findall(f"{body} {links}")
        ip_links = 0
        short_links = 0
        for url in urls:
            normalized = _normalize_url(url)
            if not normalized:
                continue
            _, host = normalized
            try:
                ipaddress.ip_address(host.strip("[]"))
                ip_links += 1
            except ValueError:
                pass
            short_links += host in URL_SHORTENERS
        _, sender_address, sender_domain = _sender_parts(sender)
        features.append(
            [
                len(subject),
                len(body),
                len(body.split()),
                upper_count / max(alpha_count, 1),
                text.count("!"),
                text.count("?"),
                sum(character.isdigit() for character in text) / max(len(text), 1),
                len(urls),
                len(set(re.findall(r"(?i)(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", links))),
                ip_links,
                short_links,
                "xn--" in links,
                bool(html_body),
                bool(sender_address),
                len(sender_domain),
                len(re.findall(r"(?i)\b(?:attachment|attached|enclosed)\b", body)),
                bool(ATTACHMENT_PATTERN.search(text)),
                text.count("$") + text.count("€") + text.count("£"),
            ]
        )
    return np.asarray(features, dtype=float)


def build_feature_transformer(max_word_features=30_000, max_char_features=25_000):
    return ColumnTransformer(
        transformers=[
            (
                "subject_word",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=8_000,
                    sublinear_tf=True,
                ),
                "subject",
            ),
            (
                "body_word",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_word_features,
                    sublinear_tf=True,
                ),
                "body",
            ),
            (
                "body_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_char_features,
                    sublinear_tf=True,
                ),
                "body",
            ),
            (
                "sender_char",
                TfidfVectorizer(
                    analyzer="char",
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=(2, 4),
                    min_df=2,
                    max_features=4_000,
                    sublinear_tf=True,
                ),
                "sender",
            ),
            (
                "structure",
                Pipeline(
                    [
                        (
                            "extract",
                            FunctionTransformer(structural_features, validate=False),
                        ),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                ["subject", "body", "sender", "html_body", "links"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def summarize_url_risk(url_items):
    reasons = []
    for item in url_items:
        reasons.extend(f"{item['domain']}: {flag}" for flag in item["flags"])
    return list(dict.fromkeys(reasons))
