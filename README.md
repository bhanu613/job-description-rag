# Job Description RAG

Ask questions about saved job descriptions and inspect the passages behind the answer.

This local Python application retrieves relevant passages from ChromaDB, sends them to a model running through Ollama, and displays the answer with source references in Streamlit.

## Screenshots

[Answer and references](docs/screenshots/answer.png)

[Inspecting the retrieved evidence](docs/screenshots/source-evidence.png)

[When the descriptions do not provide an answer](docs/screenshots/missing-information.png)


## Features

- Prepare 40 historical descriptions from a pinned dataset revision.
- Index overlapping passages with job IDs, titles and source metadata.
- Retrieve relevant passages using `nomic-embed-text`.
- Generate concise answers with `llama3.2:3b`, or acknowledge missing information.
- Inspect source passages in Streamlit and evaluate saved answers separately from retrieval.

No LangChain, LlamaIndex, fine-tuning or hosted inference API is used.

## Demo data

The demo uses 40 historical job descriptions selected from the
[vmal/jobs_dataset](https://huggingface.co/datasets/vmal/jobs_dataset)
dataset on Hugging Face. The preparation script downloads a pinned revision
and selects the first 40 unique cleaned descriptions containing at least
100 words.

See [data and provenance](docs/DATA.md) for the revision, selection process
and reuse information.

The current interface searches this prepared collection. It does not
provide document upload or editing.

## Run locally on Windows

Tested during development on Windows 11 with Python 3.12.10 and 16 GB RAM. Response time depends on available resources. Install and start [Ollama for Windows](https://docs.ollama.com/windows) first.

From the repository folder, use Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python prepare_data.py
python build_index.py
python -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

Initial package, model and dataset downloads require internet access. Question answering uses the local index and local models. The application opens in your browser; this is not public internet hosting.

## How it works

```text
Descriptions -> clean text -> overlapping chunks -> embeddings -> Chroma
Question -> query embedding -> top-3 passages -> local LLM
         -> response/source-ID checks -> answer and inspectable passages
```

The model returns an answer and passage IDs. Python checks the response format and verifies that referenced IDs were supplied. It does not certify that every reference supports the answer's meaning.

The application-specific Python code handles preparation, chunking, retrieval, prompt construction, response checks and display. Embedding and generation use pretrained models; ChromaDB and Streamlit provide storage/search and the interface.

## Evaluation

| Check | Answer review | Missing-information handling |
| --- | --- | --- |
| 15 development questions, replayed with saved retrieval contexts | 10 correct, 1 incomplete, 1 incorrect among 12 answerable questions | 3 appropriate responses out of 3 |
| 6 fresh questions, end-to-end retrieval and generation | 3 complete, 1 incomplete among 4 answerable questions | 2 appropriate responses out of 2 |

The fresh questions used existing collection documents and were fixed before their first run. These are small, single-run checks, not a general accuracy benchmark. Extra non-supporting references occurred in both sets. The active backend also passed 24 deterministic application tests on the development machine.

See [evaluation details](docs/EVALUATION.md) for definitions, per-question reviews and reproduction commands.

## Current scope

The application answers questions over a small historical collection. It does not check whether vacancies are still open.

Observed errors include incomplete answers, selecting the wrong qualification, and adding a source reference that does not support the answer. The evaluation notes show specific cases.

## Project files

- `prepare_data.py`: download, selection, cleaning and provenance.
- `build_index.py`: chunking, embedding and index synchronization.
- `rag.py`: retrieval, generation and response validation.
- `app.py`: local interface.
- `run_evaluation.py`: evaluation outputs and run metadata.
- `prepare_reference_questions.py`: reconstruct local question sheets from pinned descriptions.
- `tests/`: deterministic application tests.
- `docs/`: architecture, data provenance and evaluation review.

Downloaded descriptions, embeddings, local environments, raw diagnostic logs and development backups are excluded from version control.
