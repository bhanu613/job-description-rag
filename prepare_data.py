import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

DATASET = "vmal/jobs_dataset"
REVISION = "b40383db90f0bb8ecadc3f33794d9dfe65c085a6"
URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/"
    f"{REVISION}/data/train-00000-of-00001.parquet"
)

folder = Path("data")
folder.mkdir(exist_ok=True)
raw_path = folder / "jobs.parquet"

if not raw_path.exists():
    print("Downloading dataset...", flush=True)
    urllib.request.urlretrieve(URL, raw_path)

frame = pd.read_parquet(raw_path, columns=["source_text"])
jobs = []
seen = set()

for row_index, raw_text in enumerate(frame["source_text"]):
    if not isinstance(raw_text, str):
        continue

    text = re.sub(r"\s+", " ", html.unescape(raw_text)).strip()
    if len(text.split()) < 100 or text in seen:
        continue

    seen.add(text)
    title = re.split(
        r"\s+share link\b", text, maxsplit=1, flags=re.IGNORECASE
    )[0][:180]

    jobs.append({
        "source_id": f"JOB-{row_index:05d}",
        "title": title,
        "text": text,
        "dataset_row": row_index,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "revision": REVISION,
    })

    if len(jobs) == 40:
        break

if len(jobs) != 40:
    raise RuntimeError(f"Expected 40 usable descriptions, found {len(jobs)}")

with (folder / "jobs.jsonl").open("w", encoding="utf-8") as handle:
    for job in jobs:
        handle.write(json.dumps(job, ensure_ascii=False) + "\n")

provenance = {
    "dataset": DATASET,
    "revision": REVISION,
    "download_url": URL,
    "uploader_license": "unlicense",
    "download_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "selection": "First 40 unique rows with at least 100 words",
    "text_field": "source_text",
    "cleaning": "HTML entity decoding and whitespace normalization",
}

(folder / "provenance.json").write_text(
    json.dumps(provenance, indent=2), encoding="utf-8"
)

print(f"Prepared descriptions: {len(jobs)}")
for job in jobs[:5]:
    print(f"{job['source_id']} | {job['title']}")
print("Saved data/jobs.jsonl and data/provenance.json")