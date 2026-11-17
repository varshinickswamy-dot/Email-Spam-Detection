import argparse
from email.errors import MessageError
import hashlib
import json
import tarfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT_DIR / "data" / "raw" / "spamassassin"
CORPORA = {
    "20030228_easy_ham": "legitimate",
    "20030228_easy_ham_2": "legitimate",
    "20030228_hard_ham": "legitimate",
    "20030228_spam": "spam",
    "20030228_spam_2": "spam",
}


def safe_member_name(name):
    member_path = PurePosixPath(name)
    return not member_path.is_absolute() and ".." not in member_path.parts


def extract_archives(corpus_dir):
    root = corpus_dir.resolve()
    for corpus_name in CORPORA:
        archive_path = corpus_dir / f"{corpus_name}.tar.bz2"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing corpus archive: {archive_path}")
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            safe_members = []
            for member in archive.getmembers():
                if not safe_member_name(member.name):
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not allowed: {member.name}")
                target = (root / Path(*PurePosixPath(member.name).parts)).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isfile() or member.isdir():
                    safe_members.append(member)
            archive.extractall(corpus_dir, members=safe_members)


def inspect_email(raw_bytes):
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = str(message.get("subject", "") or "")
    sender = str(message.get("from", "") or "")
    plain_parts = []
    html_parts = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeDecodeError, KeyError):
            payload = part.get_payload(decode=True) or b""
            content = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if content_type == "text/html":
            html_parts.append(content)
        else:
            plain_parts.append(content)
    body = "\n".join(plain_parts or html_parts)
    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "html_body": "\n".join(html_parts),
    }


def export_dataset(corpus_dir=CORPUS_DIR, output_path=None):
    extract_archives(corpus_dir)
    records = []
    seen = set()
    skipped = 0
    for corpus_name, label in CORPORA.items():
        corpus_path = corpus_dir / corpus_name.removeprefix("20030228_")
        if not corpus_path.is_dir():
            raise FileNotFoundError(f"Expected extracted corpus folder: {corpus_path}")
        for email_path in sorted(corpus_path.iterdir()):
            if not email_path.is_file() or email_path.name.startswith("cmds"):
                continue
            raw_bytes = email_path.read_bytes()
            if not raw_bytes:
                skipped += 1
                continue
            try:
                record = inspect_email(raw_bytes)
            except (MessageError, UnicodeError, ValueError):
                skipped += 1
                continue
            text = "\n".join((record["subject"], record["body"], record["sender"])).strip()
            if not text:
                skipped += 1
                continue
            fingerprint = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            records.append(
                {
                    **record,
                    "label": label,
                    "source": "SpamAssassin Public Corpus",
                    "source_id": f"{corpus_name}/{email_path.name}",
                }
            )
    output_path = output_path or corpus_dir / "spamassassin_emails.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    return {"records": len(records), "labels": counts, "skipped": skipped, "output": str(output_path)}


def main():
    parser = argparse.ArgumentParser(description="Inspect and normalize SpamAssassin email archives.")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.corpus_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
