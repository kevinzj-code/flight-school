# constitution-reader-v1

The US Constitution and its 27 amendments (public domain, from the National
Archives transcriptions), used by the Reader retrieval tests in
`src-tauri/src/chunker.rs`.

| file | what it is |
|---|---|
| `constitution.md` | One `##` heading per Preamble, article section, article and amendment: 52 sections, 7,362 words, original spelling. |
| `build_md.py` | Builds `constitution.md` from the transcriptions, fetched beside it. |
| `questions.json` | Held-out questions, each with the section that answers it. |
| `queries.json` | The same questions with their embeddings, for the hybrid retrieval test. |
| `embeddings.json` | The embedding of each section. |
