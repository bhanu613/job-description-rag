# Data and provenance

- Dataset: [vmal/jobs_dataset](https://huggingface.co/datasets/vmal/jobs_dataset).
- Revision: `b40383db90f0bb8ecadc3f33794d9dfe65c085a6`.
- Input: `source_text`; structured `target_text` annotations are not used.
- Selection: first 40 unique descriptions with at least 100 words after cleaning.
- Cleaning: HTML entity decoding and whitespace normalization.
- Local provenance: dataset revision, source URL, selection rule and downloaded-file SHA-256 in `data/provenance.json`.

The uploader labels the dataset `unlicense`. This label does not independently verify the uploader's rights to employer-authored descriptions. The repository does not redistribute the downloaded dataset, full source descriptions or raw retrieval contexts. Consult the source dataset's current terms before reuse.

These are historical examples, not verified current job openings. The sample is not a representative analysis of Germany's labour market. It is independent of JobFit Navigator and the skill-extraction project.

## Reference questions without redistributing full passages

`reference_questions.json` records the questions, short reference answers, expected job IDs and evidence character offsets. It does not contain the original evidence quotations or full descriptions.

After preparing the pinned dataset, run:

```bat
python prepare_reference_questions.py
```

The helper checks the full description's SHA-256 before extracting a reference quotation locally. It reconstructs `evaluation/questions_final.csv` and `evaluation/questions_fresh.csv`. Matching existing sheets are preserved; conflicting sheets are not overwritten.

Run metadata records question-file hashes. Reconstructed CSV formatting may differ from an original spreadsheet export, so a newly serialized file need not have the original byte hash even when its reference fields agree. Keep each run's own manifest with its raw local outputs.
