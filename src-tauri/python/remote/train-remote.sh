#!/bin/zsh
# Train a Flight School folder on a rented GPU, from the Mac.
#
#   train-remote.sh <folder> <user@host:port> [--name N (default: the folder without "-training")] [--iters N] [--expect-first-val X]
#
# 1. checks the folder is whole and tokenizes it as mlx-lm does (prepare_rows.py, CPU);
# 2. syncs the folder, the tokens and this directory to /workspace/fs on the pod;
# 3. installs requirements.txt into /workspace/venv if it changed (torch from the image);
# 4. trains (train_remote.py; the base model is cached in /workspace/hf), unless this
#    exact plan already finished there; with --expect-first-val it stops the run when
#    the untrained model's validation loss is more than 1% from the Mac's;
# 5. syncs back the adapters, the checkpoints, the log and timings.json;
# 6. converts to <name>.lora.gguf on the Mac (convert_remote.py, CPU) and checks its
#    tensors against an MLX-made adapter's.
#
# The network volume refuses chown, so files are synced without owner, group or mode (-rlt).
# Safe to re-run: a finished plan is fetched, not retrained; an unfinished one is
# started again from nothing. Any failure stops the script with its reason.
set -euo pipefail

usage() { echo "usage: train-remote.sh <folder> <user@host:port> [--name N] [--iters N] [--expect-first-val X]" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
FOLDER="${1:A}"; ADDR="$2"; shift 2
NAME="${${FOLDER:t}/-training/}"; ITERS=""; EXPECT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --expect-first-val) EXPECT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$ADDR" == *@*:* ]] || { echo "the address is user@host:port, e.g. root@203.0.113.7:22042" >&2; exit 2; }
HOSTPART="${ADDR%:*}"; PORT="${ADDR##*:}"

HERE="${0:A:h}"
APP="$HOME/Library/Application Support/mode.airoplane.flightschool"
RUN="$APP/bakeoff/runs/$NAME-remote-qwen3vl"
TRAIN_PY="$APP/venv/bin/python"
CONVERT_PY="$APP/venv-convert/bin/python3"
CONFIG="${CONFIG:-$APP/bakeoff/runs/patch-asks-trained-2-qwen3vl/config.toml}"
REFERENCE="${REFERENCE:-$APP/bakeoff/runs/patch-asks-trained-qwen3vl/export/patch-asks.lora.gguf}"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o BatchMode=yes -p "$PORT" "$HOSTPART")
RSYNC=(rsync -rlt --checksum -e "ssh -i '$KEY' -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p $PORT")

say() { print -r -- "== $*"; }
die() { print -r -- "train-remote: $*" >&2; exit 1; }
for f in "$CONFIG" "$REFERENCE" "$TRAIN_PY" "$CONVERT_PY"; do [[ -e "$f" ]] || die "missing $f"; done

say "1/6 checking and tokenizing $FOLDER (CPU)"
mkdir -p "$RUN"
rm -rf "$RUN/prep"
HF_HUB_OFFLINE=1 HF_HOME="$HOME/Library/Application Support/mode.airoplane.pigeon/models/training" \
  "$TRAIN_PY" "$HERE/prepare_rows.py" "$FOLDER" "$RUN/prep" ${ITERS:+--iters $ITERS} || die "the folder failed its checks"
PLAN_SHA=$(shasum -a 256 "$RUN/prep/plan.json" | cut -c1-12)
RD="/workspace/fs/runs/$NAME-$PLAN_SHA"
say "plan $PLAN_SHA -> $RD"

say "2/6 syncing to the pod"
"${SSH[@]}" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)
  mkdir -p /workspace/fs/code '$RD/data' /workspace/hf && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
"${RSYNC[@]}" --delete --exclude __pycache__ "$HERE/" "$HOSTPART:/workspace/fs/code/remote/"
"${RSYNC[@]}" "$FOLDER/train.jsonl" "$FOLDER/valid.jsonl" "$FOLDER/grade.jsonl" "$FOLDER/counts.json" "$HOSTPART:$RD/data/"
"${RSYNC[@]}" "$RUN/prep/plan.json" "$RUN/prep/tokens" "$HOSTPART:$RD/"

say "3/6 the pod's environment"
"${SSH[@]}" "set -e; REQ=\$(sha256sum /workspace/fs/code/remote/requirements.txt | cut -c1-16)
  if [ \"\$(cat /workspace/venv/.req 2>/dev/null)\" != \"\$REQ\" ]; then
    python3 -m venv --system-site-packages /workspace/venv
    /workspace/venv/bin/pip install -q -r /workspace/fs/code/remote/requirements.txt
    echo \$REQ > /workspace/venv/.req
  fi
  /workspace/venv/bin/python -c 'import torch,transformers,peft;assert torch.cuda.is_available();print(\"torch\",torch.__version__,\"transformers\",transformers.__version__,\"peft\",peft.__version__)'"

say "4/6 training"
STATE=$("${SSH[@]}" "cd '$RD'; if [ -f DONE ] && grep -q \$(sha256sum plan.json | cut -d' ' -f1) DONE; then echo done;
  elif [ -f pid ] && kill -0 \$(cat pid) 2>/dev/null; then echo running; else echo new; fi")
if [[ "$STATE" == new ]]; then
  # Only the nohup line goes to the background: a backgrounded `a && b &` list keeps the
  # SSH channel open until training ends.
  "${SSH[@]}" "cd '$RD' && rm -rf adapters train.log metrics.jsonl timings.json DONE pid run.out || exit 1
    HF_HOME=/workspace/hf nohup /workspace/venv/bin/python /workspace/fs/code/remote/train_remote.py '$RD' > run.out 2>&1 < /dev/null &
    echo \$! > '$RD/pid'"
  say "started"
else
  say "this plan is $STATE on the pod"
fi
CHECKED=""
while :; do
  S=$("${SSH[@]}" "cd '$RD'; if [ -f DONE ]; then echo DONE; elif [ -f pid ] && kill -0 \$(cat pid) 2>/dev/null; then echo RUN; else echo DEAD; fi; tail -n 1 train.log 2>/dev/null || true")
  print -r -- "${S#*$'\n'}"
  if [[ -n "$EXPECT" && -z "$CHECKED" ]]; then
    FIRST=$("${SSH[@]}" "grep -m1 -o 'Iter 1: Val loss [0-9.]*' '$RD/train.log' 2>/dev/null | awk '{print \$5}'" || true)
    if [[ -n "$FIRST" ]]; then
      CHECKED=1
      if ! python3 -c "import sys; a,b=float('$FIRST'),float('$EXPECT'); sys.exit(abs(a-b)>0.01*b)"; then
        "${SSH[@]}" "kill \$(cat '$RD/pid') 2>/dev/null || true"
        die "the untrained validation loss is $FIRST against the Mac's $EXPECT (more than 1% apart); stopped before training"
      fi
      say "untrained validation loss $FIRST against the Mac's $EXPECT: within 1%"
    fi
  fi
  case "${S%%$'\n'*}" in
    DONE) break ;;
    DEAD) "${RSYNC[@]}" "$HOSTPART:$RD/run.out" "$HOSTPART:$RD/train.log" "$RUN/" 2>/dev/null || true
          tail -n 20 "$RUN/run.out" >&2 || true; die "training stopped without finishing (run.out above)" ;;
  esac
  sleep 30
done

say "5/6 fetching"
"${RSYNC[@]}" "$HOSTPART:$RD/adapters" "$HOSTPART:$RD/train.log" "$HOSTPART:$RD/metrics.jsonl" \
  "$HOSTPART:$RD/timings.json" "$HOSTPART:$RD/run.out" "$RUN/"
[[ -f "$RUN/adapters/adapters.safetensors" ]] || die "no adapters.safetensors came back"
cp "$RUN/prep/plan.json" "$RUN/plan.json"

say "6/6 converting on the Mac (CPU)"
(cd "$HERE/.." && "$CONVERT_PY" remote/convert_remote.py "$RUN" "$NAME" --config "$CONFIG" --reference "$REFERENCE") \
  | tee "$RUN/convert.json" || die "conversion or the tensor check failed (convert.json)"

say "done: $RUN"
python3 -c "import json; t=json.load(open('$RUN/timings.json')); print(f\"{t['iters']} rows at {t['rows_per_second']} rows a second, peak {t['peak_gb']} GB, wall {t['wall_seconds']/60:.1f} min on {t['gpu']}\")"
