# Training a text Pigeon on Qwen3-VL-2B: the recipe as run

Measured on the patch planner, rounds 1 to 3 (2026-10-08 to 10-12), on a 16 GB Apple-silicon Mac.

Train with `mlx_lm lora` on `mlx-community/Qwen3-VL-2B-Instruct-bf16` (mlx-lm drops the vision tower on load,
so a text run holds the 1.72B-parameter language model alone and there is nothing further to skip), with
Pigeon's LoRA shape (`model/planner-data/lora_config.yaml`: rank 16, scale 20), every layer (`--num-layers -1`),
learning rate 2e-5, `--mask-prompt`, `--grad-checkpoint` and `--seed 1`, at **batch 1 with
`--grad-accumulation-steps 4`** (an effective batch of 4) for **two passes** (`--iters` = 2 × the training rows),
with `--max-seq-length` set to the longest row, measured as tokens of the chat-templated row and rounded up to
a multiple of 128, `--steps-per-eval 1000` and `--save-every 500` so every 500-step checkpoint can be benched
afterwards. It runs at **0.44 rows a second with a 6 to 7 GB peak** at rows of up to about 2,000 tokens, so a
run takes 2 × rows ÷ 0.44 seconds (8,950 rows: about 11 hours) with no timing run needed. Batch 2 was measured
slower (0.35 rows a second, 10.2 GB peak on the longest rows), batch 3 went over 12 GB (13.3 GB), and batch 4
ran out of memory at 1,659 tokens: one row at a time already keeps this GPU busy, and batching pads every row
to the longest. Quit Pigeon before training (its engine holds GPU memory, and vision runs ran out of memory
twice beside it). Convert with `instructor.convert.to_gguf(…, vision=False)`, then pack with
`cli.py base-pack` from a table carrying the adapter, which flies the grade rows with the adapter and again
with it switched off as the control. The validation loss mlx-lm prints scores only 25 rows, so swings
between checkpoints of 0.001 to 0.03 are noise; the Pigeon's own bench decides.

## Remote: the default since 2026-10-13

Every run now trains on a rented RunPod GPU, not on this Mac. The recipe above is copied setting for
setting in PyTorch + PEFT on `Qwen/Qwen3-VL-2B-Instruct` (`src-tauri/python/remote/`; its `README.md`
maps each setting to the mlx-lm line it copies). The settings that carry over:
- alpha 320 (mlx-lm's scale 20 × rank 16);
- Adam without bias correction;
- the same loss mask, including mlx-lm's one pad target;
- the same row order and validation rows, from the same numpy seed;
- the step count rule rounded to the nearest row.

The Mac tokenizes, so the pod trains on the same tokens; the pod re-checks them and refuses to train
if one row differs. Run it as:

    src-tauri/python/remote/train-remote.sh <folder> root@<host>:<port>

It tokenizes, syncs, trains, downloads, and converts on the Mac along the same path as above. The
adapter then packs with `base-pack` as before. The procedure:
- create a fresh pod per run wherever stock is (Secure Cloud, RTX 4090 or L40S, 50 GB disk, SSH
  only);
- quote its price first;
- terminate it the same hour.

The pack's `report.json` says where it trained, in `check_ride.trained_where` (D97, spec Draft 11).

**Proof, the asks second pass** (`patch-asks-training-2`, 8,303 steps). The Mac file and the remote file
came from the same folder and the same steps. Pigeon's asks bench, higher is better:

| | Real asks let through, of 308 | No-skill stopped, of 100 | Grade ride, of 558 |
|---|---|---|---|
| Mac | 300 | 97 | 552 |
| Remote | 305 | 95 | 547 |

Kevin passed the remote path on these on 2026-10-13. Real asks differ by 5, against a line of 2, but
in the better direction.

**Speed, measured on an L40S:**
- asks: 8,303 rows in 43.6 min, 3.17 rows a second, 6.7 GB peak; the Mac ran 0.29 to 0.33 rows a
  second;
- focus: 1,665 rows in 9.5 min, 2.93 rows a second.

Each fresh pod also downloads the base model and the pinned packages: about 5 min (estimated). Cost
is a pod's uptime × its hourly price: about $0.93 for asks and $0.22 for focus (estimated, list
prices $1.09/hr).

**The Mac path stays as the fallback**, exactly as written above. Each run still waits for one "GPU
free" from Kevin, because the pack's ride and the bench afterwards run on this Mac's GPU. The training
itself waits on no one.
