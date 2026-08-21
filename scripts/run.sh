#!/usr/bin/env bash
# Launch Qwen3.8-27B + DFlash2 (validated config, single DGX Spark / GB10).
#
# Prereqs (see README): built image (scripts/build.sh), target NVFP4 model
# dir, DFlash2 draft in a local HF cache. All paths overridable via env.
set -euo pipefail

IMAGE="${IMAGE:-qwen38-27b-dflash2:local}"
TARGET_MODEL="${TARGET_MODEL:-/home/argo/models/Qwen3.8-27B-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/DFlash2/.cache/huggingface}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
DRAFT_REPO="${DRAFT_REPO:-z-lab/Qwen3.8-27B-DFlash2}"
DRAFT_REV="${DRAFT_REV:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
NAME="${NAME:-qwen38-dflash2}"
CONTAINER_MEM="${CONTAINER_MEM:-100g}"

# sanity
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not built - run ./scripts/build.sh first" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "target model dir $TARGET_MODEL missing" >&2; exit 1; }
[[ -d "$HF_CACHE" ]] || { echo "HF cache $HF_CACHE missing - download the draft first (see README)" >&2; exit 1; }
# mem-fraction 0.95 hard-reboots GB10 during draft-graph capture - refuse it
if awk "BEGIN{exit !($MEM_FRACTION >= 0.95)}"; then
  echo "mem-fraction >= 0.95 is NOT safe on GB10 (hard reboot at graph capture). Use 0.90." >&2; exit 1
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --gpus all --ipc host --network host \
  --memory "$CONTAINER_MEM" --memory-swap "$CONTAINER_MEM" \
  -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  --mount type=bind,source="$TARGET_MODEL",target=/model \
  --mount type=bind,source="$HF_CACHE",target=/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name qwen38-27b --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
    --context-length 262144 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_REPO" \
    --speculative-draft-model-revision "$DRAFT_REV" \
    --speculative-num-draft-tokens 16 \
    --max-prefill-tokens 16384 --mamba-full-memory-ratio 4.21 --max-mamba-cache-size 40 --max-running-requests 10 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 --port "$PORT"

echo "launched $NAME on port $PORT - wait for 'ready to roll' in: docker logs -f $NAME"
