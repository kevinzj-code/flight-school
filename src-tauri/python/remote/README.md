# Training on a rented GPU

The Mac recipe (`docs/TRAINING-RECIPE-QWEN3VL.md`, `mlx_lm lora` 0.31.3) in PyTorch + PEFT on a
RunPod RTX 4090. The pod only trains. Tokenizing, converting, packing, the check ride, sealing
and benching all stay on the Mac.

    src-tauri/python/remote/train-remote.sh <folder> root@<ip>:<port> [--name N] [--expect-first-val X]

| File | Runs on | Does |
|---|---|---|
| `rows.py` | both | the rules: tokens and loss offset, sequence limit, step count, row order, validation rows, folder check |
| `prepare_rows.py` | Mac, CPU, training venv | tokenizes with mlx-lm's own loader and `ChatDataset`; writes `plan.json` and `tokens/` |
| `train_remote.py` | pod | trains; writes mlx-lm-shaped adapters every 500 rows, `train.log`, `metrics.jsonl`, `timings.json` |
| `convert_remote.py` | Mac, CPU, converter venv | `instructor.convert.to_gguf` (the same path as a Mac adapter) and the tensor-table check |
| `train-remote.sh` | Mac | everything above, over SSH |
| `requirements.txt` | pod | pinned; torch is the image's |

## Setting by setting

| mlx-lm 0.31.3, as the Mac runs it | Here | Where mlx-lm does it |
|---|---|---|
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`, text only | `Qwen/Qwen3-VL-2B-Instruct` at `89644892`, bf16, `Qwen3VLForConditionalGeneration` fed token IDs only | |
| rank 16, `scale` 20 (`lora_config.yaml`) | `r=16`, **`lora_alpha=320`** | `tuner/lora.py:98`: `return y + (self.scale * z).astype(x.dtype)`; it multiplies by `scale` directly, and PEFT multiplies by `lora_alpha / r`, so alpha = 20 × 16 |
| `--num-layers -1`, no `keys` | q, k, v, o, gate, up, down in all 28 language layers (196 modules); the vision tower and `lm_head` carry none | the Mac adapter's 392 tensors (`test_peft_names_become_the_mac_adapters_names`) |
| LoRA weights fp32, A ~ U(±1/√in), B = 0, dropout 0 | PEFT's fp32 adapter weights, the same distributions (a different random draw) | `tuner/lora.py:87–95` |
| `adam`, lr 2e-5, constant | `MlxAdam`: betas (0.9, 0.999), eps 1e-8, **no bias correction** | `mlx.optimizers.Adam(bias_correction=False)`; `torch.optim.Adam` always corrects, so it is not used |
| `--mask-prompt` | loss from `len(template(messages[:-1], add_generation_prompt=True))` | `tuner/datasets.py:65–74` |
| … including one pad target | the step after the last token (id 0) is scored too | `trainer.py` `default_loss`: `steps <= lengths`, with the row padded by zeros; the Mac log counts 4 trained tokens per asks row (label, `<\|im_end\|>`, `\n`, pad) |
| loss: mean over a row's scored tokens | the same, in fp32 | `default_loss` |
| batch 1, `--grad-accumulation-steps 4` | gradients summed over 4 rows, ÷ 4, then one update | `trainer.py` `step()` |
| `--iters` counted in rows; an update when `it % 4 == 0` | the same; the last `iters % 4` rows never update, as on the Mac | `trainer.py` |
| step count: two thirds of two passes, to the nearest row | `rows.iters_for` (6,227 → 8,303; 1,249 → 1,665) | the asks second-pass `train.sh` |
| sequence limit: the longest templated row, rounded up to 128 | `rows.seq_limit` (asks: 1,081 → 1,152) | the recipe |
| row order: sorted by token length, a `numpy` permutation per pass from one seed | the same draws in the same order (`rows.mlx_order`), so the **same rows in the same order** | `lora.py:320` `np.random.seed`, `trainer.py` `iterate_batches` (`test_the_row_order_and_validation_rows_are_mlx_lms` runs mlx-lm's own) |
| validation: 25 rows at it 1, every 1,000 and the last | the **same 25 rows** each time | `trainer.py` `evaluate` |
| `--grad-checkpoint` | `gradient_checkpointing_enable(use_reentrant=False)` | |
| `--seed 1` | `torch.manual_seed(1)` for the LoRA init | |
| checkpoints every 500 | `0000500_adapters.safetensors` …, `adapters.safetensors`, `adapter_config.json` in mlx-lm's names and shapes (A and B transposed) | |
| chat template | the Mac tokenizes (`prepare_rows.py`, mlx-lm's code); the pod re-tokenizes with its own tokenizer and refuses to train if one row differs | asks second pass: 0 of 6,733 differ (2026-10-13) |

## What is not the same

- **Kernels.** CUDA bf16 against Metal bf16, and a different random LoRA draw. The runs agree
  closely, not bit for bit. The Pigeon's own bench decides.
- **The untrained validation loss is a weak check.** On the asks second pass the pod scored 6.225
  on the same 25 rows, against the Mac's 6.302 (1.2% lower). Almost all of it is mlx-lm's pad
  target: mean loss per scored position on the pod was 0.03 (answer), 0.00 (`<|im_end|>`), 0.00
  (`\n`) and 24.87 (the pad), so the number measures one token about 25 nats down the tail, where
  CUDA and Metal kernels differ. `--expect-first-val` stops a run outside 1%, which that run would
  have failed, so use it only with a tolerance you set before the run, or not at all. The
  tokenizer check, the 4 scored tokens per row and the bench are the real gates.

## Re-running

A plan is named by its `plan.json` hash. A finished plan is fetched from the pod, not retrained.
An unfinished one starts again from nothing. A folder whose files do not match its `counts.json`,
or that is missing a split, is refused before anything is sent.
