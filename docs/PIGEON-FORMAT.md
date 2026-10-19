# The `.pigeon` file — for Pigeon's developers

Flight School trains a small adapter on top of an LLM someone already has and
packs the result into one file, `<name>.pigeon`. This document says what is in
that file, what an app must do before loading one, and where the reference
code and its tests live in this repository. Read it once end to end; then
keep `src-tauri/python/bench/test_pigeon.py` open — every rule below has a
test there, and **that test file is the agreement between the apps.**

Paths in this document are relative to the Flight School repository.

**If you have read this file before**, start with
the app brief instead: it says what is new (§8
grounded, §9 skill, the runtime stamp on a scorecard), what you must implement,
and what we measured that will bite you if you assume otherwise. This file stays
the specification.

---

## 1. Why a file, not an adapter

An adapter (`adapter.lora.gguf`) is a few megabytes of weights that only mean
something on the exact model they were trained against, asked in the exact
shape they were trained with. Copy the bare adapter anywhere and both facts
are lost. The first pilot lesson looked broken for a week for precisely this
reason: the weights were fine, the prompt shape at test time was not.

A `.pigeon` carries the adapter *and* those facts, so any app can load it
correctly without asking anyone. It is a ZIP in store mode (no compression —
the adapter is already dense) with every file at the root, no wrapper
folder. A person with no app can unzip it and see files they recognize.

```
note-to-title.pigeon
├── pigeon.json            required · the manifest (§2)
├── adapter.lora.gguf      required · the trained weights, llama.cpp LoRA format
├── projector.gguf         required when base_model.vision · the projector, carried (§2)
├── actions.json           optional, vision only · what each KEY(…) answer presses (§6a)
├── labels.txt             required when kind = classifier · one label per line
├── system_prompt.txt      required when kind = persona
├── material/              required when kind = grounded (§8)
│   └── chunks.json        the sections the answers come from
├── skill/                 required when the manifest names a skill (§9)
│   └── skill.json         the parts a recipe answer names, as data
├── icon.png               recommended · 512×512
├── README.md              recommended · plain-words description, drafted from the run
├── report.json            optional · the scorecard, control included (§5)
└── examples/              optional · 3–5 inputs with expected outputs (§6)
    ├── 001.txt / 001.expected.txt
    └── 001.jpg            when the base is a vision model
```

Writer: `src-tauri/python/pigeon/pack.py` (`pack_run`).
Test: `test_pack_writes_the_spec_layout_at_the_root_in_store_mode`,
`test_the_training_data_never_leaves` (train/valid rows are never packed).

## 2. `pigeon.json`

A real one, from a Pigeon trained on one Mac and opened on another:

```json
{
  "format_version": 1,
  "name": "note-to-title",
  "id": "note-to-title",
  "kind": "freeform",
  "base_model": {
    "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "sha256": "1adf0b11…c3370",
    "arch": "qwen2",
    "name": "Qwen2.5",
    "size": "1.5B",
    "repo": "https://huggingface.co/Qwen/Qwen2.5-1.5B",
    "quant": "Q4_K_M",
    "vision": false
  },
  "prompt_template": "{input}",
  "output_contract": { "type": "text", "max_tokens": 45 },
  "inputs": ["text"],
  "trained_on": "2026-09-10",
  "trained_with": "Flight School 0.1.0",
  "example_count": 75,
  "author": null,
  "license": "Apache-2.0",
  "signature": ""
}
```

| field | what it is for |
|---|---|
| `format_version` | Whole number. An app refuses anything newer than it reads, naming the version (§4 step 2). |
| `name`, `id` | Display name, and its slug. `id` is what a store keys on; the training itself is fingerprinted by the adapter's sha256, not by `id`. |
| `kind` | `classifier`, `freeform`, `persona`, `extractor`, `router`, `grounded`, `skillpack` (sealed code, §12). Decides the required files and whether a self-test may show pass/fail (§4 step 5). A reader MUST refuse a kind it does not know rather than treat it as freeform — that refusal is what lets new kinds ship without a `format_version` bump (§9). |
| `base_model.file` | The GGUF it was trained against. Named for the fast path and for the refusal message; **not** the identity rule. |
| `base_model.sha256` | Hash of that file. Matching it is a fast path (§4). May be `null` if the packer skipped hashing. |
| `base_model.arch` / `name` / `size` | **The identity.** Read from the base's GGUF header (`general.architecture`, `general.basename` or `general.name`, `general.size_label`), never typed. Two files that agree on all three are the same base at possibly different quantizations. |
| `base_model.quant`, `repo` | Informational. |
| `base_model.vision` | The base takes an image; `base_model.mmproj` `{file, sha256}` then names the projector, a second file with its own hash. **The projector travels in the file as `projector.gguf`** (decided 2026-09-21), so a picture Pigeon runs on a Mac that has only the base: `sha256` is then required, and a reader MUST refuse carried bytes that do not hash to it. A reader that already holds a projector with that hash may skip the carried copy. `file` keeps the projector's original name, for the refusal message. |
| `skill` | Present when the adapter answers with a **recipe** rather than the finished thing — `{name, version}`. The named skill turns that answer into the artefact, and `skill/skill.json` carries everything it needs (§9). A reader that does not know the name MUST refuse the file. |
| `prompt_template` | The **user turn** the adapter was trained and tested with, with `{input}` (and `{image}` for vision, `{material}` for grounded — §8) as slots. Reproduce it byte for byte, then render through the base's own chat template. Flight School strips the brief before training, so there is no system prompt to carry — the template is the whole shape. |
| `output_contract` | `type` ∈ `label`, `json`, `text`; `max_tokens` is the answer budget **as measured on the run's own test**, not as planned (a plan once claimed 120 against 303 real tokens and starved its own test). `json` also carries `fields`. A `classifier` is always `label`. |
| `inputs` | `["text"]` or `["image","text"]`. |
| `trained_on`, `trained_with`, `example_count` | Provenance. `example_count` is train + valid rows; held-out test rows are not counted or shipped. |
| `author`, `license` | Free text; `license` defaults to Apache-2.0. |
| `signature` | **Reserved, empty in v1.** Do not treat a non-empty value as meaningful. A file's seal lives in `seal.json`, not here (§12). |

Writer: `src-tauri/python/pigeon/manifest.py` (`build_manifest`). Every
value comes from the run's own config, header or measurement — the module
docstring lists which. Tests: `test_manifest_reads_base_identity_and_hash_from_the_run`,
`test_a_vision_run_names_the_image_in_its_template`,
`test_a_free_text_job_is_freeform_with_a_measured_budget`,
`test_a_voice_is_not_called_a_persona_when_no_prompt_was_trained_alongside`
(a `persona` is only a kind when a system prompt was trained alongside and
ships in the file; a run that merely asked for a voice is `freeform`).

Required files by kind (`validate.py:REQUIRED_FILES`): `classifier` →
`labels.txt`; `persona` → `system_prompt.txt`; `grounded` →
`material/chunks.json` (§8). Tests:
`test_a_classifier_without_labels_txt_is_rejected`,
`test_a_persona_without_its_prompt_is_rejected`.

### 2.1 `card` — what a reader shows before anyone uses it (optional)

```json
"card": {
  "does": "Sort coin photos by type.",
  "input_hint": "one photo",
  "output_noun": "label",
  "examples": [{ "input": "what coin is this?", "output": "penny", "group": "clear" }],
  "input_stats": { "words": { "p50": 4, "p95": 5 }, "lines": { "p50": 1, "p95": 1 } },
  "check": { "passed": 5, "total": 6 },
  "revision": 1
}
```

**A reader shows every card field as the maker's claim**, beside what it
measured itself. The card never changes how a Pigeon is run.

| field | |
|---|---|
| `does` | Required when there is a card. One line, **at most 60 characters.** From the plan's `one_line`, or typed on the Package step. |
| `input_hint` | Optional, at most 40. What to paste in. Defaults to the plan's `input_shape` only when that fits. |
| `output_noun` | Optional, at most 24. What one answer is called. `label` for a classifier; otherwise only when typed. |
| `examples` | Up to 5 `{input, output, group?}`. For a run: **passed check-ride rows only** — held out, never the practice set, never the control. For a lesson: its approved pairs. Image rows are left out. |
| `input_stats` | Word and line counts of the held-out inputs, p50 and p95. Numbers only. |
| `check` | `{passed, total}` from the check ride. Absent for a lesson, which is checked by asking, not scored. |
| `revision` | 1, then one more each time the same `id` is packed into a lineup that already has it. |
| `talks_like` | Persona only, at most 60. |

Text that is too long is **refused, never cut**: a cut sentence is a different
sentence. Words typed for the card that do not fit stop the pack and are named;
a default from the plan that does not fit leaves the file without a card, and
the Package step says why. Unknown card keys are skipped by a reader, not
refused. Writer: `manifest.py` (`build_card`, `card_problems`, shared with
`validate.py`). Tests: `test_a_run_packs_a_card_from_its_plan_and_its_passed_check_ride_rows`,
`test_a_lesson_card_uses_its_approved_pairs_and_has_no_check`,
`test_an_over_long_does_is_refused_by_name_and_never_cut`,
`test_a_manifest_without_a_card_still_validates`,
`test_a_bad_card_in_a_file_is_refused_by_the_import_check`.

## 3. Reading identity out of a GGUF header

You will need this in your own language. The header is small and the reader
is ~40 lines: `src-tauri/python/pigeon/gguf.py` (`read_header`,
`base_identity`, `same_base`). The shape:

```
"GGUF"  u32 version  u64 tensor_count  u64 kv_count
then kv_count × ( string key, u32 type, value )
strings are u64 length + bytes; arrays are u32 elem_type + u64 count + values
```

Stop after the KV pairs — nothing else is needed, so a 4 GB file costs one
disk read. Identity is `{arch: general.architecture, name: general.basename
|| general.name, size: general.size_label}`. `same_base` compares `arch`
exactly, `size` after stripping non-alphanumerics, and `name` by prefix
agreement after the same normalization — because one converter writes
`"Qwen2.5"` and another `"Qwen2.5 1.5B Instruct"` for the same model.
Tests: `test_header_identity_is_read_from_the_file`,
`test_same_base_survives_a_different_quantization_and_naming_noise`,
`test_a_different_model_is_not_the_same_base`,
`test_a_different_size_of_the_same_family_is_not_the_same_base`,
`test_not_a_gguf_is_refused_by_name`.

## 4. Opening one — the import check

Reference: `src-tauri/python/pigeon/validate.py` (`validate_pigeon`,
`installed_bases`). Standard library only, so it can be lifted whole. The
steps, in order, with what to show the person:

1. **A ZIP with `pigeon.json` at the root.** Otherwise refuse by name
   (`test_not_a_zip_is_rejected`, `test_a_zip_without_a_manifest_is_rejected`).
2. **The version gate.** `format_version` newer than you read → refuse and
   name both versions and "update the app"
   (`test_a_newer_format_is_refused_and_names_the_version`).
3. **Required fields, by kind.** Name the specific missing field
   (`test_a_missing_required_field_is_named`); `{input}` must be in the
   template (`test_a_template_without_the_input_slot_is_rejected`); the
   per-kind files above. A file that fails here is not worth checking the
   base for — stop.
4. **The base.** This is the whole point of the validator, and it has two
   rules that are easy to get backwards:

   * **Identity refuses; the hash is only a fast path.** If an installed
     model's `file` and `sha256` match, load, no warning
     (`test_the_exact_file_is_the_fast_path_no_warning`). Otherwise compare
     *identity* (§3) against every installed GGUF. Same identity, different
     file — typically the same model at another quantization — **warns and
     loads**: an adapter works at any quantization of the base it was
     trained on (measured, the state record §9). **That measurement was
     made on short outputs and does not hold for long exact ones.** A titling
     adapter (5 tokens) survives a quantization change; an adapter emitting 200
     tokens of exact SVG coordinates returns well-formed nonsense across the
     same gap, because small per-token error compounds (the baseline notes,
     "Packaged as a `.pigeon` — and it does not work on the base it names").
     The rule stands; the warning it prints should be believed. The warning says
     "trained against X; this machine has the same base as Y — loads, but
     that is documented behavior, not the file it was measured on"
     (`test_the_strict_hash_trap_same_base_other_quant_warns_and_loads`,
     `test_the_hash_is_never_the_refusal_rule`).
   * **A different base refuses and names the one needed.** Never load onto
     a different architecture, model or size, and never substitute silently.
     The message names `base_model.file` and the identity triple
     (`test_a_wrong_base_refuses_and_names_the_model_needed`). A machine with
     no models at all is `missing` rather than `wrong` — the same refusal,
     the only difference being that the app should offer the download first
     (`test_no_models_at_all_is_missing_not_wrong`,
     `test_a_missing_base_refuses_to_guess`).

   `base_match` in the report is one of `exact`, `same_base_other_quant`,
   `wrong`, `missing`. Validation without a machine context (`bases=None`)
   still checks the file itself
   (`test_validation_without_a_machine_context_still_checks_the_file`).
5. **What a self-test may claim.** `report.self_test` is `verdict` for
   `classifier` and `router` — their outputs exact-match, so running
   `examples/` and showing pass/fail is honest — and `samples` for
   `freeform`, `persona`, `extractor`, whose output only another model could
   grade. For those, show the examples as samples. A green tick that means
   nothing is worse than no tick
   (`test_self_test_verdict_only_for_label_kinds`).
6. **Then the app's own steps:** copy the file somewhere the app owns
   (Flight School: `~/Library/Application Support/mode.airoplane.flightschool/pigeons/<id>.pigeon`,
   `src-tauri/python/pigeon/store.py`), register it, and at run time render
   `prompt_template` into the base's chat template, load
   `adapter.lora.gguf` as a LoRA on that base, cap generation at
   `output_contract.max_tokens`, and for `label` outputs match against
   `labels.txt`.

What "never load onto a different base" looks like in a UI: a refused file
is not copied anywhere, and the person sees the model's file name and a way
to get it. Flight School's band reads *"needs the base it was trained on —
Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (qwen2, Qwen2.5 1.5B) — and this machine
does not have it. It will not load onto a different base; download that
model first."*

The packer runs this same check on the file it just wrote and deletes the
file if it fails (`src-tauri/src/table.rs::table_send`,
`test_the_packagers_own_gate_accepts_what_it_wrote`).

## 5. `report.json` — the scorecard

Optional, and only Flight School writes one today. `check_ride` carries
`runtime` — **which engine produced the numbers** — plus
`passed`, `total`, `control_passed` (the same test with the adapter's scale
set to 0 — the number that separates "it learned" from "the instructions
got shorter"), `graded_by_model` (rows an LLM judged rather than
exact-matched; show a `~`), and the paired significance line. Numbers were
measured **on the machine that made the file**, in the engine `runtime` names;
an app that shows them should say so. `unrecorded` means the run did not stamp
one, and a score that cannot say where it was measured is not a score — measured:
an adapter that reproduced its own corpus in MLX scored **0/20** through
llama.cpp on the same questions. Flight School's own rides fly the CONVERTED
GGUF adapter in llama.cpp, the engine the sibling apps serve with, which is why
the run order is train → convert → fly. Flight School's library shows the score with a hover
"measured on the machine that made it" and does not re-run it.
Writer: `pack.py::build_report`.

## 6. `examples/`

`NNN.txt` is the input, `NNN.expected.txt` the expected output. A vision
Pigeon's example is `NNN.jpg` (the picture) with `NNN.prompt.txt` (the words
asked with it) in place of `NNN.txt`
(`test_a_vision_pack_carries_the_sample_photos_and_the_projector_hash`).

A reader MAY ask a Pigeon its examples on import and keep the revision it
replaced until they pass — Pigeon does, and puts the old one back on a miss.
When every `NNN.prompt.txt` holds the same words, a reader MAY ask them with a
picture that arrives with none typed: that is the only way the adapter was
trained to be asked, and without them Pigeon measured 4 of 6 breeds right
against 5 of 6 with them.
Picked from the run's **held-out** rows, stratified across its groups, and
they change with a seed — so they are a test the training never saw, which
is what makes step 5's `verdict` honest (`test_examples_are_stratified_and_swap_with_the_seed`).

## 6a. `actions.json` — what a picture Pigeon presses

Written by Flight School for a picture Pigeon made by recording, and read by a
reader that PLAYS it (Pigeon's `docs/PLAY-LOOP.md`). A `labels.txt` answer such
as `KEY(Confirm)` names what a key *does*; this says which key it is:

```json
{ "format_version": 1,
  "keys": { "Confirm": { "code": 7, "name": "X", "hold_ms": 95 },
            "Up":      { "code": 126, "name": "Up", "hold_ms": 260 } },
  "not_pressed": [ { "answer": "KEY(Save)", "key": "Cmd+S", "why": "held with ⌘, Ctrl or Option" } ],
  "frames": { "stack": 1, "gap_ms": 250, "max_side": 768, "crop": [0.25, 0.1, 0.5, 0.8] },
  "question": "What do you do next?" }
```

- `keys` is keyed by the name inside `KEY(…)`. `code` is the macOS key code
  (layout-independent); `name` is the key as recorded; `hold_ms` is the median
  of the maker's own presses, or `null` when none were timed — a reader then
  presses briefly and says so.
- A key recorded with ⌘, Ctrl or Option held is never in `keys`: a reader
  MUST NOT press anything that is not listed there.
- `frames` is how each picture was made: a reader asks with a picture of the
  same shape (the same number of frames side by side, the same gap between
  them, the same size cap), because that is the only shape it learned.
  `max_side` is the size it was *trained* at, which may be smaller than it was
  recorded at (Teach can shrink a table's pictures; it never enlarges them).
  It was always 768 before that choice existed. A reader shrinks each frame so
  its longer side is at most `max_side`, and never enlarges one.
  `crop`, when present, is the part of the window it learned from, as
  fractions `[x, y, w, h]` from the window's top left: a reader cuts every
  frame to that part before anything else, and `max_side` applies to the part
  kept. Absent means the whole window. A reader that cannot crop MUST NOT play
  a Pigeon that has one — it would be shown pictures it never learned.
- `question` is the words every row was asked with.

## 7. The `.nest` — a flock in one file

For moving several Pigeons at once. A store-mode ZIP of `.pigeon` files at
the root plus `nest.json`:

```json
{ "format_version": 1, "name": "my flock", "created": "2026-09-10",
  "pigeons": [ { "file": "note-to-title.pigeon", "id": "note-to-title", "name": "note-to-title",
                 "kind": "freeform", "base_file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                 "bytes": 5302115, "sha256": "f31d…" } ] }
```

**You may ignore `nest.json` entirely** and take the `.pigeon` members;
each one is validated on its own by the rules above, and a nest with one
bad member still yields the good ones with the bad one named. A ZIP of
Pigeons without `nest.json` is refused with "add each .pigeon on its own",
so hand-zipped folders do not masquerade as nests. Reference:
`src-tauri/python/pigeon/nest.py`; tests `bench/test_nest.py`.
macOS UTI for both files: they conform to `public.zip-archive`; Flight
School registers `.pigeon` and `.nest` in `src-tauri/tauri.conf.json`
(`bundle.fileAssociations`) and handles the open-file event in
`src-tauri/src/lib.rs`.

## 8. `grounded` — a Pigeon that answers from material it is handed

A `grounded` Pigeon carries reference material and answers **from what is put in
front of it at ask time**, never from what its weights remember. The adapter
supplies the manner of answering; the material supplies the facts. This is the
only kind whose file contains content rather than only behaviour.

**The app does the retrieving. The Pigeon cannot.** A Pigeon is input → output;
it has no way to search anything. On every question the host app finds the
relevant sections, fills them into the template, and runs the model. Nothing in
this file describes an agent that goes looking.

### 8.1 `material` in the manifest

```json
"kind": "grounded",
"prompt_template": "{material}\n\n{input}",
"material": {
  "retrieval": "whole",
  "sections": 47,
  "files": ["airoplane-manual.md"],
  "est_tokens": 21840
}
```

| field | what it is for |
|---|---|
| `retrieval` | How the app decides what to paste. `whole` — paste all of it, every time; the honest choice when it fits and there is nothing to get wrong. `keyword-overlap-v1` — the deterministic search in `src-tauri/src/chunker.rs` (`best_chunk_for_in`): word overlap against each section, longest words weighted, **with a floor below which the answer is no section at all**. `keyword-overlap-v2` (2026-09-30, `chunker.rs::best_chunk_v2`) — BM25 (k1 1.2, b 0.75) over stemmed words of three letters or more with a fixed stoplist, an ordinal word ("fifteenth") read as the numeral a heading carries ("xv"), a bonus of twice the word's weight when it is in the section's title, a bonus of 2 per two-word phrase the question and section share, floor 3, one section, ties to the earlier section. Measured on the Constitution Reader's 176 questions: v1 found the right section for 76, v2 for 117; the base model's own mean-pooled embeddings for 8. `hybrid-v1` (2026-09-30) — the blend of an embedding search and `keyword-overlap-v2`, the sections' paragraph vectors carried in the file (§8.7). Measured on the same 176 questions: 142 with the packed 16-bit vectors, 164 in the top three. A reader that does not know the named method MUST refuse the file and say so; it must never fall back to a different method, because answering from the wrong sections looks exactly like answering from the right ones. |
| `embedding` | Required for `hybrid-v1`: `{ "model", "file", "sha256", "dims", "units" }` — the embedding model as a **second pinned identity** beside `base_model`. A reader with a different model, or none, refuses the method by name. |
| `sections` | How many sections the material was chopped into. At least one. |
| `files`, `est_tokens` | Provenance and size. `est_tokens` is chars/3.5, the same estimate the chunker uses. |

`material/chunks.json` is the material itself: `{"chunks": [{"ordinal", "file",
"title", "text", "est_tokens"}], "total_est_tokens"}` — the serialised form of
`chunker.rs::ChunkIndex`. The section text lives here, so the source documents
are not packed a second time and the retriever reads exactly the bytes that
shipped. Writer: `pigeon/pack.py`; the manifest block:
`pigeon/manifest.py::read_material`.

### 8.2 The slot and the kind imply each other, both ways

Validated in both directions (`validate.py`), because each failure is silent:

- **`grounded` without `{material}` in the template** has nowhere to put what it
  carries, and answers every question from nothing.
- **`{material}` on any other kind** reaches a reader that does not fill it,
  which sends the literal characters `{material}` to the model. The answer comes
  back fluent and grounded in nothing at all.

### 8.3 Position is part of the contract

`{material}` sits **above** the question, separated by a blank line. That is
where the training rows put it (`lesson.rs::assemble` →
`chunker.rs::material_block`), and an adapter is only correct in the shape it was
trained in. Reproduce the template byte for byte, as with every other kind.

One section renders as:

```
[§4] How much memory a model really needs
The app shows, for every model, a line like "Needs ~3.8 GB free memory."…
```

### 8.4 Citations are checkable, and that is the point

The `[§n]` label is the section's own ordinal, so an answer that writes `[§4]` is
making a claim the app can **verify against what it actually showed the model** —
`chunker.rs::unsupported_citations`, deterministic, no judge required. Flight
School refuses to train a row whose approved answer cites a section it was not
shown.

Why this exists, measured (the baseline notes, "Step 0 of the document Pigeon"): handed
a passage that did not cover the question, the first grounded adapter invented an
answer and cited a source anyway in **11 of 14** cases — and the base model with
no adapter did the same in 10 of 14. An uncited answer is honest; an unverified
citation is decoration.

### 8.5 What a grounded self-test may claim

`samples`, never a verdict. Grounded output is prose and only another model could
grade it. Two numbers a host app CAN compute honestly and show, because both are
deterministic: how many citations in an answer were unsupported, and whether the
retrieval found any section at all.

### 8.7 `hybrid-v1` — the material embedded once, the question embedded at ask time

The file carries `material/embeddings.json`:

```json
{"model": {"name": "nomic-embed-text-v1.5", "file": "nomic-embed-text-v1.5.Q8_0.gguf",
           "sha256": "3e2434…", "dims": 768, "pooling": "mean",
           "document_prefix": "search_document: ", "query_prefix": "search_query: "},
 "chunks_sha256": "…",
 "units": [{"ordinal": 12, "part": 3, "f16": "<base64 of 768 little-endian half floats, unit length>"}]}
```

One unit per **paragraph** of a section (`text` split on a blank line), each
embedded as `document_prefix + title + "\n" + paragraph`, unit-normalised,
stored as 16-bit floats: about 2 KB a unit, 290 KB for the Constitution's 139.
The packer makes them (`pigeon/embed.py`) through llama-server in embedding
mode, 23 ms a unit; a document is never embedded by the reader.

At ask time the reader embeds the question as `query_prefix + question` with
the **same model** (its sha256 is in the manifest and must match the one the
reader carries), then, per section, takes the best paragraph's cosine, and
blends: `0.7 · z(cosine) + 0.3 · z(keyword-overlap-v2 score)`, z-scored across
sections, the best section wins, ties to the earlier one, and **no section
when the best cosine is under 0.58**. Reference: `chunker.rs::hybrid_v1`,
`best_cosines`, `v2_scores`, `unpack_f16`; Pigeon's `material.rs` is the port,
and both run the same stored questions in their tests.

`whole` and the two keyword methods stay legal. A Reader packed without the
embedding model on the Mac says `keyword-overlap-v2` and the pack reply's
`warnings` say why.

### 8.6 Open, and stated as open

- **Refusal is not better than no adapter, and the test was strong enough to
  say so.** Eight adapters over three corpora, 46 paired near-miss cases, exact
  McNemar: best p = 0.30 (the baseline notes, "Step 0c"). Do not ship a claim that a
  grounded Pigeon knows when to say "not in here." Enforce it instead where it
  is deterministic: when the retrieval floor returns no section, the host app
  should refuse **without calling the model**, and it should always show the
  sections it did pass in, so a person can see a near miss for themselves.
- A grounded `classifier` — fixed labels *and* material — is not defined. Nothing
  produces one; `grounded` currently wins over a label set.
- Embedding-based retrieval would make the embedding model a **second pinned
  identity** with the same doctrine as `base_model`: identity refuses, the hash
  is a fast path, never substitute. An index built by one embedder and searched
  by another returns confident nonsense rather than an error. Nothing implements
  this; `keyword-overlap-v1` needs no such dependency, which is why it is first.
  Measured 2026-09-30: lexical search tops out at two thirds of the Constitution
  Reader's questions (`keyword-overlap-v2`), and the base model's own embeddings
  are no substitute (8 of 176). A dedicated embedding model is the next step, and
  it is exactly the second identity this item describes.
- **One section per ask, because that is how the rows trained.** Measured
  2026-09-30 on the Constitution Reader: handed the right section it cited it in
  170 of 176; handed the three best keyword matches it did worse (91) than the
  one best (102). More sections at ask time is not more help until rows train
  that way.
- Everything is stored uncompressed (§1). Correct for a dense adapter, wasteful
  for prose; a grounded Pigeon carrying a book is the first file where that
  choice costs real megabytes.
- **Copyright.** A grounded Pigeon contains a copy of its material. Fine on the
  machine that made it; unresolved for a `.nest` that gets shared.

---

## 9. `skill` — when the answer is a recipe, not the thing

Some jobs are better answered by **naming the parts** than by drawing them. The
adapter returns a small object — `{"eyes":"happy","mouth":"grin","brows":null,
"extras":["halo"]}` — and a **skill** turns it into the artefact.

**Why the format has this at all**, measured (the baseline notes, "T2" and "The logit
test"). Asked for the drawing directly, an adapter trained on 137 hand-drawn
emoji produced **20/20 perfectly conformant SVGs that matched nothing** — 0/20 on
every part — because a 200-token coordinate sequence is a chain of knife-edge
decisions, and a single flipped token changes the whole picture. Asked for the
recipe, the same corpus carried real information, in **19 tokens instead of 165**.
A recipe also has no coordinates, so the cross-runtime instability that §4
warns about cannot arise for it at all.

### 9.1 The manifest

```json
"skill": { "name": "face-grammar", "version": 1 },
"output_contract": { "type": "json",
                     "fields": ["eyes", "mouth", "brows", "extras"],
                     "max_tokens": 48 }
```

`fields` names the recipe's keys so a reader can check the shape without knowing
the skill. `version` is part of the identity: the parts table IS the contract, and
a Pigeon trained against one set of shapes must not be rendered by another.

### 9.2 `skill/skill.json` — data, never code

```json
{ "name": "face-grammar", "version": 1,
  "render": { "method": "concat", "version": 1 },
  "how": "Concatenate `head` and the fragment for each named part, in the order
          eyes, brows, mouth, extras, inside `wrapper`.",
  "wrapper": "<svg …>{parts}</svg>",
  "head": "<circle cx=\"12\" cy=\"12\" r=\"9\"/>",
  "order": ["eyes", "brows", "mouth", "extras"],
  "parts": { "eyes": {"happy": "<path …/>", …}, "mouth": {…}, … },
  "recipe_schema": { … } }
```

**No executable code ships in a `.pigeon`, ever.** One arrives by AirDrop or a
flash drive; an app that ran code out of it would have thrown away the
validate-then-load doctrine this whole document is built on. A grammar does not
need code: a part name maps to a fragment and rendering is **concatenation** —
ten lines in any language, no evaluator. The reference renderer is
`src-tauri/python/facegrammar.py`, and
`test_a_skill_pigeon_names_its_skill_and_carries_it_as_data` renders from the
payload alone, the way a foreign app would, and asserts it matches.

`recipe_schema` is a JSON Schema whose enums are built FROM the parts tables, so
a runtime that constrains generation with it can never refuse an answer the
corpus contains.

**`render` says how the recipe is drawn, and it is what a reader accepts a skill
by.** The one method today is **`concat` v1**:

* `wrapper` contains `{parts}` **exactly once** (absent means `"{parts}"`);
  `head` is a string (absent means empty);
* `order` is a non-empty list of keys, and `parts` has a table for **every** key
  in `order`, mapping each part name to a string;
* the artefact is `head`, then for each key in `order` the fragment of each named
  part (a value may be one name or a list of names; missing, `null`, `false`,
  `""` and `[]` contribute nothing), all put into `wrapper` at `{parts}`.

The artefact is **untrusted markup** from someone else's file. An app shows it the
way an image is shown (`<img src="data:image/svg+xml,…">`), never by inserting it
into its own page.

A payload with no `render` is read as `concat` v1 **only** for the skills packed
before `render` existed — `face-grammar` v1. Every other skill must say how it is
drawn.

### 9.3 What a reader must do

1. **Accept a skill by how it is drawn, not by its name.** Its table ships
   beside it, so a skill no app has heard of opens in every app that performs its
   `render` method — no update needed. **A method or method version the reader
   cannot perform is a refusal**, named — exactly like an unknown `kind` (§4 step
   3), and for a sharper reason: drawing a recipe the wrong way produces a
   **well-formed artefact of the wrong thing**, which no later check catches.
   `skill/skill.json`'s `name` and `version` must equal the manifest's: the parts
   table IS the contract, so face-grammar v2 is not face-grammar v1. Never
   substitute a skill, and never render a recipe with a table that did not ship
   beside it.
2. **`skill/skill.json` must be present.** A recipe naming parts the file does not
   carry cannot be drawn. `validate.py` refuses the file
   (`test_a_skill_pigeon_without_its_payload_is_rejected`).
3. **A part name the payload does not contain is a miss, not something to
   repair.** `loads_recipe` never coerces an unknown name to the nearest known
   one — a renderer that repairs its input cannot be trusted to have drawn what
   the model asked for.

### 9.4 Grading

A recipe is machine-checkable, so a check ride grades it **by rule** rather than
by a model's opinion: `report.json` carries `how: "rules"` on those rows and
`graded_by_model: 0`. See `instructor/grade.py`; the rule is named in the run's
plan and resolved once, and an unresolvable rule raises rather than quietly
falling back to a judge.

**Open, and stated as open:** §4 step 5 still allows a self-test `verdict` only
for label-shaped kinds. Recipe output is exactly checkable and is not a label, so
a skill Pigeon reports `samples` today even though a verdict would be honest.
That is the third case the rule does not cover, and it wants deciding with a
measurement in hand rather than now.

---

## 10. Versioning and what is open

* `format_version` goes up only when a reader of the old version would
  misread the new file. Adding an optional field is not a version bump;
  changing what `prompt_template` means is.
* Readers accept anything ≤ their own version.
* **`card` (§2.1) is not a version bump.** It is optional, a reader that does
  not know it reads the file exactly as before, and a card with a key a reader
  does not know is read without that key.
* **A new `kind` is not a version bump, because step 3 already refuses a kind
  it does not know and names it.** `grounded` (§8) shipped that way: an older
  reader refuses the file and says which kind it cannot read, which is the
  correct outcome — far better than reading it as `freeform`, ignoring the
  material it carries, and answering from nothing. This only holds while every
  reader keeps that refusal; a reader that treats an unknown kind as freeform
  breaks the guarantee for everyone.
* **A new skill is not a version bump either, and needs no reader update**, as
  long as it is drawn by a method readers already perform (§9.2). A new `render`
  method is the one that needs one — and step 1 of §9.3 refuses it by name until
  the reader has it.
* **Seal trust (accepted for v1):** one trusted key may seal any `id` — a key
  is not scoped to the Pigeons it may sign. Key rotation and revocation ship as
  an app update that changes the trusted key list; there is no in-file or
  online revocation.
* Open: whether `router` should be visible in the apps at all (it
  validates today, nothing produces one); the exact `json` output contract
  beyond `fields`; what the reserved `signature` field in `pigeon.json` will
  ever mean (the seal lives in `seal.json`, §12); the grounded open items in §8.6, and the skill verdict question in §9.4.

## 11. The contract, in one sentence

Copy `validate.py` and `gguf.py`, or keep your own reader green against
`bench/test_pigeon.py` — run with

```bash
python3 -m pytest src-tauri/python/bench/test_pigeon.py -q
```

— and a Pigeon that opens in one app opens in all of them.

---

## 12. Seal — `seal.json`, and the `skillpack` kind

A seal says **who packed a file**, never that it is good. It lives in its own
entry, `seal.json`; the `signature: ""` field in `pigeon.json` stays reserved
and empty. Reader: `src-tauri/python/pigeon/seal.py`; tests:
`bench/test_seal.py`. The Rust and Node readers implement the same text.

### 12.1 `seal.json`

A sealed `.pigeon` has `seal.json` **exactly once, as the last zip entry**:

```json
{"scheme":"pigeon-seal-ed25519-v1","key":"<16 lowercase hex>","signature":"<128 lowercase hex>"}
```

Exactly these three keys, written compact in this order
(`json.dumps(obj, separators=(",", ":"))`). A reader parses it as JSON and
requires exactly these keys (a repeated key is refused), the scheme string
exactly, and the hex fields lowercase at exactly those lengths.

* `key` — the first 16 hex characters of sha256 of the 32-byte raw Ed25519
  public key.
* Signed message (bytes): `PIGEON-SEAL-V1\n`, then for **every entry except
  `seal.json`**, sorted by name bytes ascending, one line:
  `<sha256 of the entry's data, 64 lowercase hex>` + two spaces + `<name>` + `\n`.
* `signature` — Ed25519 (RFC 8032, pure) over that message, hex. Verification
  refuses S ≥ L and any public key or R that does not decode canonically, and
  checks the cofactorless equation [S]B = R + [k]A.

### 12.2 Strict zip rules

A reader that checks a seal parses the zip bytes itself and refuses **the
whole file** on any violation, so every reader sees exactly the same entries:

1. Every name matches `^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$`
   and is 1–255 bytes, checked on raw bytes (non-ASCII refused). That excludes
   `.`/`..` segments, directory entries (trailing `/`), backslashes, NUL,
   newlines and a leading `/`.
2. No duplicate names, and no two names equal after ASCII lower-casing.
3. Every entry stored (method 0); general-purpose flags 0 or 0x800 only, in
   both the local header and the central directory.
4. The local header's name bytes, method, compressed size and uncompressed
   size equal the central directory's; compressed size == uncompressed size.
5. Contiguous: the first local header is at offset 0; walking the central
   directory **in its own order**, each entry's local header offset is exactly
   where the previous entry's data ended (30 + name_len + extra_len + size);
   the central directory starts exactly where the last entry's data ends and
   its size matches its records; the end-of-central-directory record starts at
   exactly `len − 22` with comment length 0 and disk numbers 0; no zip64 (no
   0xFFFF / 0xFFFFFFFF sentinels); at most 4096 entries; the central
   directory's record count matches the end record.

CRC-32 is not checked by these rules; the seal's sha256 lines cover the data.

### 12.3 Kinds

* **`skillpack`** — the one kind that ships code, under `code/`. It MUST carry
  a seal that verifies against a key the app trusts; unsealed, or with no
  trusted keys configured, it is refused. Its `pigeon.json` requires
  `format_version`, `name`, `id`, `kind`, `version` (whole number ≥ 1),
  `models` (object) and `code` (object whose `entry` names an existing
  `code/…` entry). It has no `base_model`, `prompt_template` or
  `adapter.lora.gguf`.
* **Any other kind** with an entry whose name starts with `code/` is refused,
  sealed or not.
* Data-only Pigeons (every earlier kind, no `code/`) stay valid unsealed. If
  one does carry `seal.json`, the strict rules apply and the seal must verify.

### 12.4 Keys

The published **test** key is RFC 8032 §7.1 TEST 1 (secret seed
`9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60`, public key
`d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a`, key id
`21fe31dfa154a261`). It is trusted **only in tests**. A real key is made by
`cli.py keygen --out F` (32-byte seed as 64 hex + newline, mode 0600, parent
0700, never inside either repository); `cli.py seal` refuses a key file that
group or other can read. Trust policy is in §10.

