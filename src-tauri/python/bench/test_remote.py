"""The remote recipe (src-tauri/python/remote/) against the mlx-lm run it copies.

    python3 -m pytest src-tauri/python/bench/test_remote.py -q

The tests that need mlx run it in the Mac's training venv as a subprocess, on
the CPU, and skip when that venv is missing. The PEFT tests skip without peft
(the pod's pins, remote/requirements.txt, install it).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY))

from remote import rows as R  # noqa: E402
from remote import train_remote as T  # noqa: E402

MLX_PY = Path.home() / "Library/Application Support/mode.airoplane.flightschool/venv/bin/python"
MAC_ADAPTER = (Path.home() / "Library/Application Support/mode.airoplane.flightschool/bakeoff/runs"
               / "patch-asks-trained-qwen3vl/adapters/adapters.safetensors")


def run_mlx(code: str) -> str:
    if not MLX_PY.exists():
        pytest.skip("no Mac training venv")
    pre = "import mlx.core as mx\nmx.set_default_device(mx.cpu)\n"
    return subprocess.run([str(MLX_PY), "-c", pre + textwrap.dedent(code)], capture_output=True, text=True,
                          check=True, timeout=300).stdout


# --- the rules ------------------------------------------------------------------

def test_step_count_and_sequence_limit_match_the_mac_asks_run():
    assert R.iters_for(6227) == 8303          # train.sh: 8303 steps over 6227 rows
    assert R.iters_for(1249) == 1665 and R.iters_for(1299) == 1732  # the focus and OCR train.sh
    assert R.seq_limit([1081, 40]) == 1152    # train.sh: --max-seq-length 1152
    assert R.seq_limit([128]) == 128


def test_the_loss_covers_the_answer_and_mlx_lms_one_pad_target():
    tokens = [10, 11, 12, 13, 14, 15]          # prompt 10..13, answer 14 15
    inputs, targets, mask = R.batch_arrays(tokens, 4, 2048)
    assert len(inputs) == 32 and inputs[:6] == tokens  # padded to 1 + 32, minus one
    scored = [t for t, m in zip(targets, mask) if m]
    assert scored == [14, 15, R.PAD_ID]        # 3 targets: the answer and the first pad


def test_a_truncated_row_scores_no_pad():
    tokens = list(range(1, 41))
    inputs, targets, mask = R.batch_arrays(tokens, 30, 36)
    assert len(inputs) == 35 and targets[-1] == 36
    assert [t for t, m in zip(targets, mask) if m] == list(range(31, 37))  # tokens 30..35


def test_the_mask_matches_mlx_lms_default_loss():
    """The targets default_loss scores, read off mlx-lm's own batches."""
    cases = {1: (7, 3), 2: (40, 30), 3: (33, 32), 4: (90, 80)}   # first token: (length, offset)
    out = run_mlx(f"""
        import json
        from mlx_lm.tuner.trainer import iterate_batches
        rows = [([k] + list(range(100, 99 + n)), o) for k, (n, o) in {cases}.items()]
        res = {{}}
        for _, (batch, lengths) in zip(range(4), iterate_batches(rows, 1, 128)):
            b = batch.tolist()[0]; o, l = lengths.tolist()[0]
            res[b[0]] = [b[:-1], [t for s, t in enumerate(b[1:], 1) if o <= s <= l]]
        print(json.dumps(res))
    """)
    got = json.loads(out)
    assert len(got) == 4
    for k, (n, o) in cases.items():
        inputs, targets, mask = R.batch_arrays([k] + list(range(100, 99 + n)), o, 128)
        assert got[str(k)] == [inputs, [t for t, m in zip(targets, mask) if m]]


def test_the_row_order_and_validation_rows_are_mlx_lms():
    """mlx-lm's own iterate_batches, called in train()'s order, from one numpy seed."""
    train_lens = [5, 9, 3, 9, 12, 7, 3, 20, 11, 6, 9]
    valid_lens = [4, 8, 2, 8, 6, 10, 3]
    iters, every, vb = 40, 7, 4
    out = run_mlx(f"""
        import json, numpy as np, mlx.core as mx
        from mlx_lm.tuner.trainer import iterate_batches
        train = [([i] + [1] * (n - 1), 0) for i, n in enumerate({train_lens})]
        valid = [([i] + [1] * (n - 1), 0) for i, n in enumerate({valid_lens})]
        np.random.seed(1)
        order, evals = [], {{}}
        for it, (b, _) in zip(range(1, {iters} + 1), iterate_batches(train, 1, 64, loop=True, comm_group=mx.distributed.init())):
            order.append(int(b[0, 0].item()))
            if it == 1 or it % {every} == 0 or it == {iters}:
                evals[it] = [int(vb_[0, 0].item()) for _, (vb_, _) in zip(range({vb}), iterate_batches(valid, 1, 64, comm_group=mx.distributed.init()))]
        print(json.dumps([order, evals]))
    """)
    order, evals = json.loads(out)
    mine, mine_evals = R.mlx_order(train_lens, valid_lens, iters, seed=1, steps_per_eval=every, val_batches=vb)
    assert mine == order
    assert {str(k): v for k, v in mine_evals.items()} == evals


def test_the_adam_step_is_mlxs_without_bias_correction():
    torch = pytest.importorskip("torch")
    g = [[0.5, -0.2, 0.0, 3.0], [-1.0, 0.25, 0.1, -0.3]]
    p0 = [0.1, 0.2, -0.3, 0.4]
    out = run_mlx(f"""
        import json, mlx.core as mx, mlx.optimizers as o
        opt = o.Adam(learning_rate=2e-5)
        params = {{"w": mx.array({p0})}}
        for g in {g}:
            params = opt.apply_gradients({{"w": mx.array(g)}}, params)
        print(json.dumps(params["w"].tolist()))
    """)
    p = torch.tensor(p0)
    opt = T.MlxAdam([p], 2e-5)
    for gg in g:
        opt.step([torch.tensor(gg)])
    assert np.allclose(p.numpy(), json.loads(out), rtol=0, atol=1e-7)
    q = torch.tensor(p0)
    torch_adam = torch.optim.Adam([torch.nn.Parameter(q)], lr=2e-5)
    assert torch_adam.defaults["betas"] == (0.9, 0.999)


# --- the adapter's shape ----------------------------------------------------------

def test_peft_names_become_the_mac_adapters_names():
    key, which = T.mlx_key("base_model.model.model.language_model.layers.27.mlp.down_proj.lora_B.default.weight")
    assert (key, which) == ("language_model.model.layers.27.mlp.down_proj.lora_b", "B")
    names = {T.mlx_key(f"base_model.model.model.language_model.layers.{l}.{m}.lora_{w}.default.weight")[0]
             for l in range(T.LAYERS)
             for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj") for w in "AB"}
    assert len(names) == 392
    if MAC_ADAPTER.exists():
        with open(MAC_ADAPTER, "rb") as f:     # the safetensors header: 8-byte length, then JSON
            header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        assert names == set(header) - {"__metadata__"}


def test_lora_lands_on_the_language_layers_only_and_saves_in_mlx_shapes():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    tc = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, vocab_size=512, tie_word_embeddings=True,
              rope_scaling={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"})
    vc = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
              deepstack_visual_indexes=[])
    torch.manual_seed(1)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(text_config=tc, vision_config=vc))
    cfg = peft.LoraConfig(r=16, lora_alpha=320.0, lora_dropout=0.0, target_modules=T.TARGET, bias="none")
    model = peft.get_peft_model(model, cfg)
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(lora) == 2 * 2 * T.PER_LAYER and all("language_model.layers" in n for n, _ in lora)
    layer = model.base_model.model.model.language_model.layers[0].self_attn.q_proj
    assert layer.scaling["default"] == 20.0    # alpha / r == mlx-lm's scale
    state = T.mlx_state(model)
    a = state["language_model.model.layers.0.self_attn.q_proj.lora_a"]
    b = state["language_model.model.layers.0.self_attn.q_proj.lora_b"]
    assert tuple(a.shape) == (64, 16) and tuple(b.shape) == (16, 64) and a.dtype == torch.float32
    assert float(b.abs().sum()) == 0.0         # B starts at zero, as in mlx-lm
    bound = 1 / np.sqrt(64)                    # A ~ U(+-1/sqrt(in)), as in mlx-lm
    assert float(a.abs().max()) <= bound + 1e-6
    # one row trains and its loss counts what default_loss counts
    loss, n = T.row_loss(model, list(range(5, 45)), 36, 64, torch.device("cpu"))
    assert n == 5 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for _, p in lora)


def test_the_adapter_config_is_mlx_lms_shape():
    cfg = T.adapter_config({"batch_size": 1, "grad_accumulation_steps": 4, "iters": 8303, "learning_rate": 2e-5,
                            "rank": 16, "dropout": 0.0, "scale": 20.0, "max_seq_length": 1152, "save_every": 500,
                            "seed": 1, "steps_per_eval": 1000, "steps_per_report": 10, "val_batches": 25})
    assert cfg["lora_parameters"] == {"rank": 16, "dropout": 0.0, "scale": 20.0}
    assert cfg["num_layers"] == -1 and cfg["fine_tune_type"] == "lora" and cfg["mask_prompt"] is True


# --- refusing a partial folder ------------------------------------------------------

def _folder(tmp_path, counts=None, drop=None, bad_row=False):
    chat = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}]}
    for split, n in (("train", 3), ("valid", 2), ("grade", 1)):
        if split == drop:
            continue
        rows = [chat] * n
        if bad_row and split == "train":
            rows = [{"messages": [{"role": "user", "content": "q"}]}] + rows[1:]
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "counts.json").write_text(json.dumps(counts or {"rows": {"train": {"all": 3}, "valid": 2,
                                                                          "grade": {"all": 1}}}))
    return tmp_path


def test_a_whole_folder_passes(tmp_path):
    assert R.check_folder(_folder(tmp_path)) == {"train": 3, "valid": 2, "grade": 1}


def test_a_short_copy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="train.jsonl has 3 rows; counts.json says 4"):
        R.check_folder(_folder(tmp_path, counts={"rows": {"train": 4, "valid": 2, "grade": 1}}))


def test_a_missing_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no grade.jsonl"):
        R.check_folder(_folder(tmp_path, drop="grade"))


def test_a_row_without_its_answer_is_refused(tmp_path):
    with pytest.raises(ValueError, match="row 1 is not a chat"):
        R.check_folder(_folder(tmp_path, bad_row=True))


def test_a_folder_without_counts_is_refused(tmp_path):
    _folder(tmp_path)
    (tmp_path / "counts.json").unlink()
    with pytest.raises(ValueError, match="no counts.json"):
        R.check_folder(tmp_path)


# --- conversion ---------------------------------------------------------------------

def test_tensor_tables_report_every_difference():
    from remote.convert_remote import same_shape
    ref = [("blk.0.attn_q.weight.lora_a", (2048, 16), "F16"), ("blk.0.attn_q.weight.lora_b", (16, 2048), "F16")]
    assert same_shape(ref, ref) == []
    made = [("blk.0.attn_q.weight.lora_a", (2048, 8), "F16")]
    assert same_shape(made, ref) == ["missing blk.0.attn_q.weight.lora_b",
                                     "blk.0.attn_q.weight.lora_a: ((2048, 8), 'F16') against ((2048, 16), 'F16')"]


# --- D97: the report says where the adapter trained -----------------------------------

def test_the_report_carries_trained_where_beside_rows_source():
    from pigeon import basepack
    flight = {"passed": 1, "total": 2, "samples": 0, "off_contract": 0, "runtime": "llama.cpp", "temperature": 0,
              "constrained": True, "control_passed": 0, "same_as_expected": None, "rows": []}
    m = {"base_model": {"sha256": "ab", "file": "base.gguf"}}
    t = {"rows_source": "grade.jsonl", "adapter": "/x/a.lora.gguf", "trained_where": "RunPod RTX 4090, torch + peft"}
    ride = basepack.report(t, m, flight, [])["check_ride"]
    assert ride["trained_where"] == "RunPod RTX 4090, torch + peft" and ride["rows_source"] == "grade.jsonl"
    assert "trained_on" not in ride, "D97 is trained_where; the manifest's trained_on is a date"
    old = dict(t, trained_on=t.pop("trained_where"))
    assert basepack.report(old, m, flight, [])["check_ride"]["trained_where"] == "RunPod RTX 4090, torch + peft", \
        "a table written before the rename still packs"
    del t["adapter"]
    assert "trained_where" not in basepack.report(t, m, flight, [])["check_ride"]
