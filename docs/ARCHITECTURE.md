# Architecture

## Index preparation

```text
Pinned dataset
  -> source_text selection: first 40 unique cleaned rows with >=100 words
  -> words: windows of up to 220, overlap 40
  -> prepend job title
  -> nomic-embed-text through Ollama
  -> Chroma collection: jobs_m03_220_40
```

The recorded collection contains 89 chunks. The word limit applies to each source passage before the added title. Chunk count is determined by the selected texts, not fixed per job.

Each record stores its text, embedding, job ID, chunk index, title, original dataset row, source URL and dataset revision. Document inputs use `search_document:`; query inputs use `search_query:`.

## Question answering

```text
Streamlit or command line
  -> embed the question
  -> query Chroma for three passages
  -> construct messages containing instructions, passages and question
  -> one llama3.2:3b generation call
  -> parse answer and source_ids
  -> validate fields, known source IDs and missing-information consistency
  -> display answer and original passages
```

Top-k counts passages, not distinct jobs. The generator sees selected readable text, not the entire database or the retrieval vectors.

An empty passage list returns an application-level missing-information response without a model call. With passages present, the model decides whether the evidence establishes an answer. Malformed outputs and service failures are separate statuses.

## Boundaries

The final implementation does not ask the model to reproduce evidence quotations, run a second model reviewer, or retry generation. It checks source-ID membership, not semantic entailment. Neither a low vector distance nor a valid ID establishes factual correctness.

The current index builder stages embeddings before writes and removes obsolete IDs after successful upserts. It is not a transactional, crash-safe index rebuild. Changing the embedding model or representation requires a compatible index; do not mix incompatible vectors.
