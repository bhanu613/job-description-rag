# Evaluation

## Method

Reference answers were fixed before the corresponding model runs. Answerable questions identify a target job and an evidence excerpt. Missing-information questions concern details not established by the relevant saved description.

The generator receives only its question and retrieved passages, not the reference fields. Answers were reviewed by comparing the recorded outputs against the recorded retrieved passages. The labels below are qualitative case judgments, not an independently annotated benchmark.

| Dimension | Definition |
| --- | --- |
| Source hit@3 | At least one of three retrieved passages belongs to the expected job. Measured only for answerable questions. |
| Answer correctness | The answer addresses the question and agrees with the source. |
| Completeness | The requested details and meaningful alternatives are preserved. |
| Citation support | A referenced passage actually establishes the relevant answer, rather than merely belonging to the same job. |
| Missing-information handling | The response acknowledges absent evidence instead of guessing. |

A valid execution status is not a correctness judgment. Source hit@3 does not establish that the answer-bearing passage was retrieved or that every cited passage supports the answer.

## Development set

The final simple-generation method was run on 15 familiar questions using saved top-three retrieval contexts. Among 12 answerable questions, review found 10 correct answers, one incomplete stakeholder list (Q04), and one answer selecting minimum-qualification developer tools instead of preferred accessible technologies (Q10). All three missing-information questions received an appropriate response.

Expected-job source hit@3 in those saved contexts was 12/12. Several outputs referenced an extra non-supporting passage. Q09 included a reference to a different job with matching salary text; that does not establish the target role's salary.

These questions informed development and method selection. Do not describe them as held-out evaluation. See `docs/evaluation/development_review.csv`.

## Fresh end-to-end check

Six new questions were fixed before their first run on the integrated backend. They used the same collection, but performed new question embedding, retrieval and generation.

- Expected-job source hit@3: 4/4 answerable questions.
- Complete correct answers: F01, F02 and F03.
- Incomplete answer: F04 named HSPICE but omitted the allowed equivalent simulator.
- Appropriate missing-information responses: F05 and F06.
- Extra non-supporting references: F01, F03 and F04.

The recorded fresh run used one generation call per question. The code and question sheet remained unchanged during the run. Six examples cannot establish general reliability. See `docs/evaluation/fresh_review.csv` and the sanitized fresh-run metadata.

## Reproduce locally

With Ollama running and the environment active:

```bat
python prepare_data.py
python build_index.py
python prepare_reference_questions.py
python -m unittest discover -s tests -p test_simple_generation.py -v
python run_evaluation.py --backend candidate --questions evaluation/questions_final.csv
python run_evaluation.py --backend candidate --questions evaluation/questions_fresh.csv
```

Each evaluation creates a separate local output folder with results, retrieved contexts, diagnostics and a manifest. Retain them locally. New runs may differ from recorded outputs; do not discard failures or substitute a better rerun without reporting the selection.

`reference_questions.json` preserves both sets. The earlier fresh set is no longer unseen once used for further tuning. The public review files contain short outputs and judgments, not full source passages or raw diagnostic traces.

## Application tests

The active implementation passed 24 deterministic tests using controlled model responses. These cover output structure, unknown or duplicate references, missing information, no-context behavior, malformed JSON, truncation, service errors and the one-call contract. They do not test LLM factual accuracy or prove semantic support.
