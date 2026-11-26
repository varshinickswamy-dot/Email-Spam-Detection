import re
from difflib import SequenceMatcher

from backend.email_features import ATTACHMENT_PATTERN, prepare_email


SIGNAL_PATTERNS = {
    "credential_request": re.compile(
        r"(?i)\b(?:verify|confirm|reset|update)\s+(?:your\s+)?(?:password|credentials|account|login)\b"
        r"|\benter\s+(?:your\s+)?(?:password|credentials)\b"
        r"|\bconfirm\s+your\s+identity\b"
    ),
    "financial_urgency": re.compile(
        r"(?i)\b(?:wire transfer|gift card|urgent payment|pay immediately|"
        r"bank account|payment is overdue|avoid suspension|invoice due)\b"
    ),
    "account_threat": re.compile(
        r"(?i)\b(?:account (?:will be|has been) suspended|unusual sign.?in|"
        r"security alert|unauthorized access|account will be closed)\b"
    ),
    "prize_claim": re.compile(
        r"(?i)\b(?:you(?:'ve| have) won|claim your prize|selected as (?:a )?winner|"
        r"claim your reward|lottery winner)\b"
    ),
    "impersonation": re.compile(
        r"(?i)\b(?:IT help ?desk|technical support|billing department|"
        r"security team|account verification)\b"
    ),
    "promotional_language": re.compile(
        r"(?i)\b(?:unsubscribe|limited.time offer|special offer|discount|"
        r"sale ends|subscribe now|save up to|free trial|new collection)\b"
    ),
    "urgent_language": re.compile(r"(?i)\b(?:act now|immediately|urgent|within 24 hours|today only)\b"),
}

BRAND_DOMAINS = {
    "amazon": {"amazon.com"},
    "apple": {"apple.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com"},
    "paypal": {"paypal.com"},
    "google": {"google.com", "google.co.uk"},
    "bank": set(),
}


def _matches(pattern, text):
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def _sender_impersonation(sender, urls):
    lowered = sender.lower()
    reasons = []
    sender_match = re.search(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", sender, re.I)
    sender_domain = sender_match.group(1).lower() if sender_match else ""

    def trusted_domain(domain, known_domains):
        return any(domain == known or domain.endswith(f".{known}") for known in known_domains)

    def lookalike_brand(domain, brand, known_domains):
        if not domain or not known_domains or trusted_domain(domain, known_domains):
            return False
        labels = domain.lower().split(".")
        normalized_labels = [label.replace("-", "").replace("_", "") for label in labels]
        return any(
            len(label) >= 4 and SequenceMatcher(None, label, brand).ratio() >= 0.72
            for label in normalized_labels[:-1]
        )

    for brand, domains in BRAND_DOMAINS.items():
        if brand in lowered and domains and sender_domain and not trusted_domain(sender_domain, domains):
            reasons.append(f"Sender display/address uses {brand} but is not from its known domain")
        elif lookalike_brand(sender_domain, brand, domains):
            reasons.append(f"Sender domain {sender_domain} resembles the {brand} domain")
    for item in urls:
        domain = item["domain"].lower()
        for brand, domains in BRAND_DOMAINS.items():
            if brand in item["anchor_text"].lower() and domains and not trusted_domain(domain, domains):
                reasons.append(f"Link text invokes {brand} but points to {item['domain']}")
            elif lookalike_brand(domain, brand, domains):
                reasons.append(f"Link domain {domain} resembles the {brand} domain")
    return list(dict.fromkeys(reasons))


def analyze_email(record, spam_probability, classification_threshold=0.5):
    classification_threshold = float(classification_threshold)
    if not 0 <= classification_threshold <= 1:
        raise ValueError("Classification threshold must be between zero and one.")
    uncertainty_margin = 0.12
    uncertainty_low = max(0.0, classification_threshold - uncertainty_margin)
    uncertainty_high = min(1.0, classification_threshold + uncertainty_margin)
    prepared = prepare_email(record)
    text = f"{prepared['raw_subject']}\n{prepared['raw_body']}\n{prepared['raw_html_body']}"
    signals = {
        name: _matches(pattern, text)
        for name, pattern in SIGNAL_PATTERNS.items()
        if _matches(pattern, text)
    }
    url_reasons = [
        f"{item['domain']}: {flag}"
        for item in prepared["url_items"]
        for flag in item["flags"]
    ]
    impersonation_reasons = _sender_impersonation(
        prepared["raw_sender"], prepared["url_items"]
    )
    suspicious_phrases = list(
        dict.fromkeys(phrase for matches in signals.values() for phrase in matches)
    )
    reasons = []
    if spam_probability >= classification_threshold:
        reasons.append("Email-content model found patterns associated with spam.")
    else:
        reasons.append("Email-content model score is below the spam threshold.")
    if url_reasons:
        reasons.extend(url_reasons)
    if impersonation_reasons:
        reasons.extend(impersonation_reasons)
    if "credential_request" in signals:
        reasons.append("Message asks the recipient to verify, reset, or provide account credentials.")
    if "financial_urgency" in signals:
        reasons.append("Message includes urgent payment or financial-transfer language.")
    if "account_threat" in signals:
        reasons.append("Message threatens account restriction or claims suspicious account access.")
    if "prize_claim" in signals:
        reasons.append("Message uses an unrequested prize or reward claim.")
    if "promotional_language" in signals and spam_probability < uncertainty_high:
        reasons.append("Message contains promotional or subscription language.")
    if ATTACHMENT_PATTERN.search(text):
        reasons.append("Message mentions an executable or archive-type attachment.")

    risk_points = 0
    if spam_probability >= uncertainty_high:
        risk_points += 25
    elif spam_probability >= classification_threshold:
        risk_points += 12
    elif spam_probability >= uncertainty_low:
        risk_points += 5
    risk_points += min(20, len(url_reasons) * 10)
    risk_points += min(20, len(impersonation_reasons) * 15)
    risk_points += min(20, 20 if "credential_request" in signals else 0)
    risk_points += min(12, 12 if "financial_urgency" in signals else 0)
    risk_points += min(10, 10 if "account_threat" in signals else 0)
    risk_points += min(6, 6 if ATTACHMENT_PATTERN.search(text) else 0)
    risk_score = min(100, risk_points)

    high_threat = bool(
        url_reasons
        or impersonation_reasons
        or "credential_request" in signals
        or "financial_urgency" in signals
        or "account_threat" in signals
    )
    uncertain = uncertainty_low <= spam_probability <= uncertainty_high
    if high_threat and (spam_probability >= uncertainty_low or risk_score >= 35):
        classification = "phishing_suspicious"
    elif uncertain:
        classification = "uncertain"
        reasons.append("Model score is near the decision boundary; review the message manually.")
    elif spam_probability > uncertainty_high:
        classification = "spam"
    elif "promotional_language" in signals:
        classification = "promotional"
    else:
        classification = "safe"

    if risk_score >= 75:
        risk_level = "critical"
    elif risk_score >= 45:
        risk_level = "high"
    elif risk_score >= 20:
        risk_level = "medium"
    else:
        risk_level = "low"

    recommendations = {
        "safe": "No immediate threat indicators were found. Verify unexpected requests through a known, trusted channel.",
        "promotional": "Treat as marketing. Use the sender's verified unsubscribe option if you no longer want these messages.",
        "spam": "Avoid replying, opening attachments, or following links. Report or delete if unsolicited.",
        "phishing_suspicious": "Do not click links or share credentials/payment details. Contact the organization using a trusted address or phone number.",
        "uncertain": "Review the sender and destination domains independently. Do not act on urgent requests until verified.",
    }
    return {
        "classification": classification,
        "classification_label": {
            "safe": "SAFE / LEGITIMATE",
            "promotional": "PROMOTIONAL",
            "spam": "SPAM",
            "phishing_suspicious": "PHISHING / SUSPICIOUS",
            "uncertain": "UNCERTAIN / REQUIRES REVIEW",
        }[classification],
        "spam_probability": round(float(spam_probability), 4),
        "classification_threshold": round(classification_threshold, 4),
        "uncertainty_band": [
            round(uncertainty_low, 4),
            round(uncertainty_high, 4),
        ],
        "probability_type": "calibrated_sigmoid_probability",
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": list(dict.fromkeys(reasons)),
        "suspicious_phrases": suspicious_phrases,
        "suspicious_urls": [
            item for item in prepared["url_items"] if item["suspicious"]
        ],
        "links": prepared["url_items"],
        "recommended_action": recommendations[classification],
        "signals": list(signals),
    }


def detect_phishing_campaigns(history_items, minimum_matches=3):
    """Find repeated suspicious sender domains or subjects in one user's scan history."""
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    domain_groups = defaultdict(list)
    subject_groups = defaultdict(list)
    for item in history_items or []:
        if item.get("classification") not in {"phishing_suspicious", "spam"} and item.get("risk_level") not in {"high", "critical"}:
            continue
        try:
            created = datetime.fromisoformat(str(item.get("created_at", "")).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                continue
        except (TypeError, ValueError):
            continue
        sender = str(item.get("sender", "") or "").lower()
        match = re.search(r"@([a-z0-9.-]+\.[a-z]{2,})", sender)
        if match:
            domain_groups[match.group(1)].append(item)
        subject = re.sub(r"[^a-z0-9]+", " ", str(item.get("subject", "") or "").lower()).strip()
        if len(subject) >= 8:
            subject_groups[subject].append(item)

    alerts = []
    for domain, items in domain_groups.items():
        if len(items) >= minimum_matches:
            alerts.append({
                "type": "repeated_sender_domain",
                "indicator": domain,
                "count": len(items),
                "message": f"{len(items)} suspicious emails from the same sender domain were detected in your last 30 days of scans.",
                "latest": max(str(item.get("created_at", "")) for item in items),
            })
    for subject, items in subject_groups.items():
        if len(items) >= minimum_matches:
            alerts.append({
                "type": "repeated_subject",
                "indicator": subject[:120],
                "count": len(items),
                "message": f"{len(items)} suspicious emails share the same normalized subject in your last 30 days of scans.",
                "latest": max(str(item.get("created_at", "")) for item in items),
            })
    return sorted(alerts, key=lambda alert: (alert["count"], alert["latest"]), reverse=True)[:20]
