import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["candidate", "baseline"], default="candidate")
    parser.add_argument("--questions", default="evaluation/questions_final.csv")
    args = parser.parse_args()
    question_file = Path(args.questions)
    if not question_file.is_absolute():
        question_file = ROOT / question_file
    with question_file.open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))
    required = {"question_id", "type", "question", "expected_source_id", "expected_quote", "expected_answer"}
    if not questions or not required.issubset(questions[0]):
        raise SystemExit("Missing questions or required columns.")
    seen = set()
    for q in questions:
        if not q["question_id"].strip() or q["question_id"] in seen:
            raise SystemExit("Question IDs must be nonempty and unique.")
        seen.add(q["question_id"])
        if not q["question"].strip() or q["type"] not in {"answerable", "missing_information"}:
            raise SystemExit(f"Invalid question: {q['question_id']}")
        if q["type"] == "answerable" and any(not q[x].strip() for x in ["expected_source_id", "expected_quote", "expected_answer"]):
            raise SystemExit(f"Missing reference evidence: {q['question_id']}")
    code_file = ROOT / ("rag.py" if args.backend == "candidate" else "rag_m04_snapshot.py")
    spec = importlib.util.spec_from_file_location("evaluated_backend", code_file)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    folder = ROOT / "evaluation" / (args.backend + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    manifest = {"backend": args.backend, "code_file": code_file.name, "code_sha256": digest(code_file),
                "question_file": str(question_file), "questions_sha256": digest(question_file),
                "question_count": len(questions), "embedding_model": pipeline.EMBED_MODEL,
                "generation_model": pipeline.CHAT_MODEL, "collection": pipeline.COLLECTION_NAME,
                "top_k": 3, "generation_options": getattr(pipeline, "OPTIONS", "historical baseline settings"),
                "note": "Source-ID validity is not semantic citation support. Manual answer/citation review required."}
    if args.backend == "candidate":
        manifest["system_prompt"] = pipeline.SYSTEM_PROMPT
        manifest["max_generation_attempts"] = pipeline.MAX_ATTEMPTS
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["question_id", "type", "question", "expected_source_id", "expected_quote", "expected_answer",
              "retrieved_chunk_ids", "retrieved_source_ids", "source_hit_at3", "answer", "referenced_chunk_ids",
              "execution_status", "attempts", "elapsed_seconds", "retrieval_evidence", "answer_support",
              "answer_completeness", "citation_support", "notes"]
    with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as f, (folder / "contexts.jsonl").open("w", encoding="utf-8") as c, (folder / "diagnostics.jsonl").open("w", encoding="utf-8") as d:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            print(f"Running {q['question_id']}...", flush=True)
            started, passages, diagnostics, references = time.perf_counter(), [], [], []
            status, attempts = "service_error", 0
            try:
                passages = pipeline.retrieve(q["question"], top_k=3)
                response = pipeline.generate_answer(q["question"], passages)
                if args.backend == "candidate":
                    status, answer = response.status, response.answer
                    attempts, diagnostics, references = response.attempts, response.diagnostics, response.source_ids
                else:
                    answer, attempts = response, 1 if passages else 0
                    status = "validation_failed" if response.startswith("Answer validation failed:") else "insufficient_evidence" if response.strip() == "The supplied passages do not provide that information." else "answered"
            except Exception as error:
                answer = "The request could not be completed."
                diagnostics = [{"reason": f"{type(error).__name__}: {error}"}]
            sources = [p["metadata"]["source_id"] for p in passages]
            row = {field: q.get(field, "") for field in fields}
            row.update({"retrieved_chunk_ids": "|".join(p["chunk_id"] for p in passages),
                        "retrieved_source_ids": "|".join(sources),
                        "source_hit_at3": int(q["expected_source_id"] in sources) if q["type"] == "answerable" else "",
                        "answer": answer, "referenced_chunk_ids": "|".join(references), "execution_status": status,
                        "attempts": attempts, "elapsed_seconds": round(time.perf_counter() - started, 2)})
            writer.writerow(row); f.flush()
            c.write(json.dumps({"question_id": q["question_id"], "question": q["question"], "passages": passages}, ensure_ascii=False) + "\n"); c.flush()
            d.write(json.dumps({"question_id": q["question_id"], "diagnostics": diagnostics}, ensure_ascii=False) + "\n"); d.flush()
            print(f"{q['question_id']}: {status}", flush=True)
    manifest["code_unchanged_during_run"] = digest(code_file) == manifest["code_sha256"]
    manifest["questions_unchanged_during_run"] = digest(question_file) == manifest["questions_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results folder: {folder}")
    print("Answers and citation support still require review.")


if __name__ == "__main__":
    main()
