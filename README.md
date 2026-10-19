# Flight School

**Teach a small model one job, on your own Mac, and hand it to Pigeon.**

Flight School is the macOS app that trains Pigeons. You fill a table with
examples of the job (by recording yourself using an app, importing folders,
or letting a model draft rows), and Flight School trains a small LoRA adapter
on them, gives it an exam it has never seen, and packs the result as one
`.pigeon` file of about 5–20 MB. [Pigeon](#pigeon), the companion app, runs it.

An adapter teaches *form*: a format, a set of labels, a voice. It does not
reliably install facts. Every exam is therefore asked twice, once of the
trained Pigeon and once of the same model untrained, and the scores are shown
side by side so you can see what training actually bought.

## Pigeon

Flight School needs Pigeon installed. It borrows the models Pigeon downloads
(from `~/Library/Application Support/mode.airoplane.pigeon/models`) instead of
keeping a second copy, and its exam runs through Pigeon's own engine,
`pigeon-ride`, so the score on the card is the score Pigeon will get.

To build `pigeon-ride`, clone [Pigeon](https://github.com/kevinzj-code/pigeon)
at the commit named in `src-tauri/python/pigeon/ENGINE_PIN.json` and build it:

```bash
cargo build --release -p pigeon-engine --bin pigeon-ride
```

Then either clone into
`~/Library/Application Support/mode.airoplane.flightschool/engine/pigeon`,
where Flight School looks by default, or point `PIGEON_RIDE` at the binary.

## Privacy

Your tables, recordings and trained Pigeons stay on your Mac, in
`~/Library/Application Support/mode.airoplane.flightschool/`. The first time
you teach, Flight School builds its own Python environment and downloads the
training packages from PyPI and the base model's weights from Hugging Face.
After that, teaching runs offline.

Recording captures the window you choose and the keys and clicks you make in
it, so it needs Screen Recording and Input Monitoring permission. It skips
keystrokes while macOS has secure input on (password fields).

Training on a rented GPU (`src-tauri/python/remote/`) is optional and only
happens when you start it yourself. It copies the training rows to a machine
you rent.

## Requirements

- A Mac with Apple Silicon. 16 GB of RAM is comfortable for the 2B base.
- Python 3.9–3.13 (`python3` on the path; the one with Xcode's tools works).
- Pigeon, with its models downloaded.
- To build from source:
  - Xcode command line tools (`xcode-select --install`). These include the
    Swift compiler that builds the window recorder.
  - Rust (`rustup`) and Node 18 or later.

## Build and run

```bash
npm install
```

```bash
npm run dev
```

`npm run dev` opens the app in a development window. To record, use
`npm run app` instead: it builds a real `.app` signed with the first
code-signing identity on your Mac (or `FS_SIGN_IDENTITY`), which macOS needs
before it will keep Screen Recording and Input Monitoring permission across
rebuilds.

```bash
npm run build
```

`npm run build` makes a release bundle in `src-tauri/target/release/bundle/`.
You need to sign and notarise it yourself.

The bundled `llama-server` (llama.cpp b9967, Metal embedded, system frameworks
only) is committed. To rebuild it from source:

```bash
sh src-tauri/scripts/build-llama-server.sh
```

## Tests

```bash
npm test
```

```bash
cd src-tauri && cargo test
```

```bash
python3 -m pytest src-tauri/python/bench/ -q
```

`npm test` checks the page's rules and engine in Node with no model. `cargo
test` needs no Python or network. The pytest suite covers measurement, the
`.pigeon` format, the seal, the lineup and the `.nest`; it needs `numpy`.
A few tests read runs or Pigeons from your own app folder and skip when there
are none.

## Layout

```
src/                 the app: plain ES modules, no framework
  screens/           one module per screen
src-tauri/           the Rust shell (Tauri 2); the webview gets no filesystem, shell or network
  python/            the trainer, the packer, the exam and the seal (driven by Rust)
    instructor/      training, conversion to GGUF, the check ride
    pigeon/          the .pigeon packer, reader, seal and CLI
    bench/           every measurement, once, and the pytest suite
    remote/          optional training on a rented GPU
    vendor/          llama.cpp's GGUF converter, pinned to b9967
  recorder/          the window recorder (Swift), built by build.rs
  keys/              the public keys this build trusts (docs/SIGNING.md)
  registry/          the sealed list of base models a Pigeon may name
datasets/            the three small sets the tests read
games/seed-run/      a small game to record a Player Pigeon on
test/                the Node tests
docs/                the .pigeon format, the training recipe, signing, development notes
```

[`docs/PIGEON-FORMAT.md`](docs/PIGEON-FORMAT.md) describes the `.pigeon` file.
[`docs/DEVELOPMENT-NOTES.md`](docs/DEVELOPMENT-NOTES.md) is the project's lab
notebook: how each part came to be the way it is, and what was measured.

## Base models

Flight School trains on bases from its sealed registry. Each base's licence
travels with every Pigeon trained on it.

| Base | Licence |
|---|---|
| Qwen3-VL 2B Instruct (the default) | Apache 2.0 |
| Qwen3.5 4B | Apache 2.0 |
| Qwen2.5 1.5B Instruct | Apache 2.0 |

## Licence

Apache 2.0; see [LICENSE](LICENSE). Bundled third-party code and fonts are
listed in [NOTICE](NOTICE).
