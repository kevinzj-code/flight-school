> **These are the development notes, kept as history.** This was the README
> while the app was built. Parts are out of date, and it cites measurement
> notes and scripts that are not part of the public source. The
> current README is at the root.

# Airoplane Mode — Flight School

Train a small language model to do one job in your format, on your own Mac, and
pack the result as a `.pigeon` file around a ~5–20 MB LoRA adapter. The GGUF
models it plans, drafts and trains with live in Pigeon's folder,
`~/Library/Application Support/mode.airoplane.pigeon/models`, and Flight School
borrows them (`src-tauri/src/pigeonstore.rs`). No terminal, no cloud, no
account: Flight School builds its own sandboxed Python environment the first
time it teaches, from a button on the Teach screen, and everything after that
happens locally.

**What an adapter is, stated up front.** It is prompt compression. It teaches
*form* — a format, a set of labels, a voice — reliably and cheaply. It does not
install facts; a 130-row attempt to teach a product manual failed by memorizing
fragments instead of learning structure. What it offers is "your format,
reliably, on a laptop," not "a custom AI model."

---

## The app today: one table, six stages

The app is `src/index.html`, a handful of ES modules and no framework:

| File | What it is |
|---|---|
| `src/app.js` | Routing (`#/runs`, `#/run/<id>/<stage>`, `#/pigeons`, `#/bench`), the one `render()`, and every event handled once at the root. Focus and caret survive every redraw; every dialog is modal and Escape closes it. |
| `src/state.js` | The one state object `S`, the rules that decide what a run still needs (`issues`, `gate`, `currentStage`), and every change a person can make to a run. No I/O — the tests read it without a model. |
| `src/engine.js` | The only file that talks to Rust. Saving (debounced, atomic, with a `summary.json` beside every table), pictures (`pic://` URLs), recording, comparing (in a Web Worker, `src/compare.js`), teaching, the check ride, packing, and what this Mac has. |
| `src/screens/` | One module per screen: `runs`, `describe`, `collect`, `record`, `review`, `teach`, `ride`, `send`, `pigeons`, `bench`, plus `shell` (the rail), `cells` (the pieces every row is drawn with) and `html` (escaping and icons). |

**A run is one table becoming one Pigeon.** What goes in on the left (text, or
a picture), what should come out on the right (prose, or one answer from a
list). A new run first asks **what kind of Pigeon** — Reader, Player, Sorter,
Looker, Writer, or Something else (`src/screens/kind.js`, design in
the kinds design note). A kind fixes the columns and hides the choices it
made; Something else is Describe with every dial. Then six stages, each
stating its honest condition:

| Stage | What it does |
|---|---|
| **Describe** | Name it; choose text or pictures in, prose or a function out. Every choice shows its cost — minutes and memory — from measured numbers. |
| **Collect** | Fill the table: record yourself using any app (every key and click becomes a row), import a folder, pair two folders by file name, import a Pigeon's play log, or let a model draft. |
| **Review** | What the table will teach, before a run: rows that disagree, repeats, unnamed clicks, the NONE share, the exam. Every suggestion has a default; only real gaps block. |
| **Teach** | Train it. Progress, loss and memory as the trainer reports them. Stop and Teach-again ask first; a finished Pigeon is kept aside while a new round runs. |
| **Check ride** | The exam it has never seen, asked twice — once of the taught Pigeon, once of the same model untrained — with each kind of mistake counted on its own. |
| **Send** | Pack the `.pigeon`, see what it weighs, hand it to Pigeon. |

**Pigeons** is the lineup: every `.pigeon` this Mac can hand to Pigeon, with
add, export the flock as a `.nest`, and remove to the Trash. **Bench** is every
run and every Pigeon in one table, the score with training beside the score
with it removed.

**This Mac.** The rail's foot says whether the Mac can teach — the training
tools, Pigeon and its models, the picture tools — and the one button that
makes it able to. A fresh Mac learns this on the first screen, not at Teach.

**Seeing it without the native window.** A local preview script (not part of
the public source) copies `src/` behind a stub of the Rust bridge that answers from this Mac's
real run folders; serve `scripts/preview-app/` and open it. Query strings
`?env=0`, `?pigeon=0` and `?vision=0` show the setup states.

## Exports: one file, the `.pigeon`

A finished run packages into **one file** — `<name>.pigeon` — carrying
everything the training needs to work somewhere else:

```
pigeon.json         which LLM it needs, the exact way to ask it, what it answers
adapter.lora.gguf   the training itself
labels.txt          the only answers allowed (sorters)
icon.png · README.md
report.json         the scorecard — the training-removed control included
examples/           3–5 held-out inputs with their right answers
```

It is a ZIP with a different extension, store mode, files at the root. The
manifest is written **from the run's own config** — the base's identity read
out of its GGUF header, the sha256 of the file it was trained against, the
prompt shape the check ride used — never typed by hand. The practice set never
leaves the machine that trained it; `report.json` describes it.

**The packager's own gate.** The file it just wrote is re-opened and passed
through the same import check Pigeon will run. A run is marked
complete only if its own importer accepts the file; a rejected file is not
kept. **`docs/PIGEON-FORMAT.md` is the explainer for the Pigeon
developers**, and the app brief is the shorter thing to hand them
first — what changed, what they must implement, and what we measured that will
bite them if they assume otherwise. `src-tauri/python/pigeon/validate.py` is that check, and
`bench/test_pigeon.py` is the cross-app contract the other apps keep in sync
with.

**The lineup and the `.nest`.** Every Pigeon this Mac can fly lives in one
folder — `~/Library/Application Support/mode.airoplane.flightschool/pigeons/`
— as `<id>.pigeon` beside a `<id>.json` sidecar (when and where it came from,
what the importer said). A run that packages drops a copy there; *Add a
Pigeon…* validates a file someone sent and copies it in (refused files are
never copied); double-clicking a `.pigeon` in the Finder does the same.
*Export the flock…* writes a `.nest` — a store-mode ZIP of `.pigeon` files
plus `nest.json` — and adding a `.nest` adds every good Pigeon inside it.
Code: `src-tauri/python/pigeon/{store,nest}.py`, `src-tauri/src/pigeons.rs`.

**Two rules the validator encodes, both corrections to the first draft of the
spec:**

- *An exact hash is a fast path, not the refusal rule.* An adapter works on
  the base it was trained against at any quantization of that same model
  (measured). Same base, different quantization: warns and loads. A genuinely
  different base — architecture, model or size — refuses, and the message
  names the model needed. It never loads onto a different base.
- *A self-test never claims a verdict it cannot grade.* Sorters and routers
  get pass/fail on their sample inputs; everything else shows the samples
  without a verdict.

**A grounded Pigeon carries its own reference material** — `kind: "grounded"`,
a `{material}` slot above the question in the prompt template, and
`material/chunks.json` in the file. The app pastes the material in at ask time;
the Pigeon never goes looking for anything, because it cannot. In Flight School
this is the **Reader** kind (the kinds design note): drop a document in
(`.md`, `.txt`, `.html`, or a `.pdf` read by PDFKit and, for a scanned page,
macOS's own text recognition — `src-tauri/src/material.rs`), a model drafts
questions per section and near-miss refusals, and the check ride counts three
things beside the untrained model without a judge: refused when it should,
answered a covered question without a citation, and followed the passage when
a number in it was changed (`instructor/grounding.py`). Every row trains on one
section above the question, so the file is always `keyword-overlap-v1`.
the grounded-Pigeon brief is what the Pigeon side needs to know. The
behaviour that decides whether such a Pigeon is shippable — saying "not in
here" when the material does not cover the question — is trainable but **not
yet better than no adapter at all** (the baseline notes, "Step 0b").
Read that before promising it.

A voice trained here is packaged as `freeform`, not `persona`: `persona` means
"inject `system_prompt.txt` on every call," and training strips the system
prompt first so the adapter works with none. Shipping a prompt it
never saw would make it worse.

### What a `.pigeon` is, by comparison

The packaging is the same thought as a Claude Project or a custom GPT: one
model specialized for one job, with a name, an icon, a plain-words README, a
declared job and a declared answer shape, handed to someone else as a thing.
The `.nest` is that thought one level up, and the Finder association is the
part Projects do not have — a Pigeon is a document, not a row in a vendor's
database.

The mechanism is the opposite one. **A Project is configuration; a Pigeon is
weights.** A Project's behaviour lives in a prompt evaluated at run time
against a frontier model the recipient does not hold — change the model
underneath and it still loads, differently. A Pigeon's behaviour was fit by
gradient descent into `adapter.lora.gguf` and means something only against one
base asked in one shape, which is why the two things this format spends most
of its rules on — `base_model.arch/name/size` as the identity, and
`prompt_template` reproduced byte for byte — have no equivalent there. A
Project cannot be broken by losing its prompt shape, because its prompt shape
*is* it. A bare adapter can, and that is why this file exists.

Which is also what makes it a file at all: you cannot ship a custom GPT,
because the expensive half is on someone else's servers. A few megabytes of
LoRA plus a manifest genuinely is the whole specialization — the base model is
the part the recipient already has.

One difference worth naming in the other direction. A Project has no notion of
being checked. `report.json` carries the training-removed control, the self
test refuses a verdict it cannot grade, and the held-out rows that produced
both never ship. That is closer to a model card than to a Project, and it is
what makes a lineup of Pigeons comparable to each other in a way a shelf of
custom GPTs is not.

**The fused full-model path is retired** (train → fuse → whole-model GGUF). It
produced a bigger file with no more fidelity than the base, had no working UI,
and its output was never verified to load. The adapter path is the product.

## Bases

Fixed, because these are the ones measured trainable and verified end to end
into the sibling app:

| Base | Used for |
|---|---|
| Qwen2.5-1.5B-Instruct | Short, exact jobs — sorting, titles, recaps, diagram notation |
| Qwen3-4B-Instruct-2507 | The Lesson flow's answering jobs |

Qwen3.5-class hybrid models are **not** trainable here: ~60× slower to
back-propagate in MLX, with most layers off-limits to a convertible adapter.
That is an architectural ceiling, not a to-do. See the Phase A results.

---

## Build

```bash
npm install
npm run dev      # tauri dev — the fastest loop, unsigned
npm run app      # a debug .app signed with the Mac's own development identity, so
                 # macOS keeps its Screen Recording and Input Monitoring grants across rebuilds
npm run build    # release bundle (unsigned until a Developer ID is configured)
```

The bundled `llama-server` is built from llama.cpp b9967 with Metal embedded
and OpenSSL off — system frameworks only, so it runs on any Apple Silicon Mac
without Homebrew. Rebuild it with the script in the sibling app's
`scripts/build-sidecar.sh`, adding `-DLLAMA_OPENSSL=OFF` and
`-ffile-prefix-map`.

## The standing gates

Every change ends with these green. They are the asset; the code is the thing
being renovated around them.

```bash
npm test                                                    # the page's rules and engine, in Node, no model
cd src-tauri && cargo test                                  # no Python or network
python3 -m pytest src-tauri/python/bench/ -q                # measurement, the .pigeon contract, the store, the .nest, the loop, grounded
```

The corpus audit no longer depends on anyone remembering to run it — it also
runs automatically inside every build. The script stays because the pilot corpus has checks of its own
that are specific to routing.

Today's numbers, and what any later change is compared against, are in
the baseline notes (kept outside this repository).

The pilot gate runs five configurations over the same 47 held-out cases and the
point of the design is the **control**: the base model with the trained prompt
shape and no adapter. Without it a score cannot tell "the adapter learned the
job" from "the prompt got shorter."

---

## Things worth knowing before changing this

- **The webview owns nothing but pixels.** `capabilities/default.json` grants
  `core:default` and window dragging — no filesystem, no dialog, no shell, no
  network. Every capability is a named Rust command. The shell plugin is
  registered Rust-side only, to spawn the drafting sidecar. Pictures reach the
  page through the `pic://` scheme, which serves only files a run's table names.
- **Every llama-server takes a random `--api-key`**, so another process on the
  Mac cannot use the model through its port. `/health` stays open.
- **The audit** that shaped the current state, with what is still open, is
  an app audit kept outside this repository.
- **Dialog commands must be `async fn`.** A sync one deadlocks the main thread
  on the native panel's callback — it waits forever for a callback that can
  never fire, because the panel needs the thread that is blocked.
- **There are two Python environments and they must stay separate** — mlx-lm
  needs `transformers>=5`, llama.cpp's converter pins `4.57.6`. A third is built
  on demand for vision.
- **An empty `valid.jsonl` is not the same as a missing one.** mlx-lm returns
  `[]` for a missing file and crashes on an empty one, so small datasets get no
  validation file at all.
- **The vendored Python is where fixes go.** `src-tauri/python/instructor/` is
  the pilot's package minus its CLI/TUI/server layer. The pilot repo is a frozen archive; do not edit it and expect anything.
- **`src-tauri/python/bench/` has exactly one copy of each measurement.** The
  leak threshold (0.82) and the label-ratio limit (1.6:1) are measured values,
  and they previously existed in five files across three repos, each free to
  drift. Rust calls into this package rather than reimplementing it. If you
  find yourself writing a second `too_close`, that is the bug.
- **The no-adapter control is not a setting.** Every check ride asks its
  questions twice — once with the adapter, once with it switched off — and
  reports the paired test, not the difference of two totals. Measured
  run-to-run noise on a 20-case judged ride is four cases (the baseline notes §5b).
- **Do not touch the pilot corpus (v3) or its adapter.** The corpus cannot be
  regenerated.
- **The question text is not the identity of a row.** A vision corpus reuses one
  phrasing across every image on purpose, so anything that asks "is this the
  same row?" must key on `(question, image)` — `dataset._identity` is that key.
  Getting this wrong does not crash; it silently reports the whole corpus as
  empty, or as contradictory. It has now done both.
- **`--steps-per-eval` does nothing on the vision path.** mlx-vlm hardcodes
  `val_dataset=None` in its trainer CLI, so vision runs train with no validation
  signal and cannot early-stop. The check ride is the only verdict there — which
  is what this project's doctrine says anyway.

---

## Where this is going

This README describes what runs today. The finished shape — the screens above, real
adaptive follow-up questions, the user's own examples as seed data, and one
portable `.pigeon` file per run — is specified in the product's spec sheet and
build plan, which are kept outside this repository.
A code-verified state record and the measurement notes the numbers in this
file come from are kept outside this repository.

What remains is Phase 6: a signed, notarized build run on a second Mac, and
the gate the spec defines as the meaning of done — a person new to the app completes describe → follow-ups → examples → trained Pigeon, unaided.
