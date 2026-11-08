import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["question_id", "type", "question", "expected_source_id", "expected_quote", "expected_answer"]


def prepare():
    catalogue = json.loads((ROOT / "reference_questions.json").read_text(encoding="utf-8"))
    data_path = ROOT / "data" / "jobs.jsonl"
    if not data_path.exists():
        raise SystemExit("Run prepare_data.py before preparing reference questions.")
    jobs = {job["source_id"]: job for job in (
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    planned = []
    for filename, entries in catalogue["sets"].items():
        if Path(filename).name != filename:
            raise SystemExit("Invalid question-sheet filename.")
        rows = []
        for entry in entries:
            row = {key: entry.get(key, "") for key in FIELDS}
            row["expected_quote"] = ""
            if entry["type"] == "answerable":
                job = jobs.get(entry["expected_source_id"])
                if job is None:
                    raise SystemExit(f"Missing reference description: {entry['expected_source_id']}")
                if job.get("revision") != catalogue["revision"]:
                    raise SystemExit("Dataset revision differs from the reference catalogue.")
                selector = entry["evidence_selector"]
                body = job["text"]
                if hashlib.sha256(body.encode("utf-8")).hexdigest() != selector["description_sha256"]:
                    raise SystemExit(f"Description changed: {entry['expected_source_id']}. Nothing overwritten.")
                start, end = selector["start"], selector["end"]
                if not (0 <= start < end <= len(body)):
                    raise SystemExit("Invalid evidence offsets.")
                row["expected_quote"] = body[start:end]
            rows.append(row)
        destination = ROOT / "evaluation" / filename
        if destination.exists():
            with destination.open(encoding="utf-8-sig", newline="") as handle:
                existing = list(csv.DictReader(handle))
            if existing != rows:
                raise SystemExit(f"Existing {filename} differs. Nothing overwritten; preserve it for review.")
            planned.append((destination, rows, True))
        else:
            planned.append((destination, rows, False))
    for destination, rows, existing in planned:
        if existing:
            print(f"Kept matching sheet: {destination.name} ({len(rows)} questions)")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Prepared: {destination.name} ({len(rows)} questions)")


if __name__ == "__main__":
    prepare()
