#!/bin/bash
# ============================================================================
# nvfp4-guard.sh — NVFP4 enforcement harness for Qwen3.8-27B DFlash2 testing
#
# Makes it impossible to accidentally benchmark/serve a non-NVFP4 checkpoint
# and report the numbers as NVFP4. Four gates:
#   1 FORMAT  config.json must say nvfp4-pack-quantized, 4-bit
#   2 BOOT    container must be up AND serving that exact model dir
#   3 QUALITY junk detectors (JSON empty-fields, math anchor, repetition loop)
#   4 CLOCK   SM clock sampled mid-load; flags throttled boxes
#
# Usage:
#   ./nvfp4-guard.sh check <model_dir>
#   NAME=clean PORT=30001 ./nvfp4-guard.sh full <model_dir> [port]
# Env: NAME PORT MEM_FRACTION IMAGE HFCACHE
# ============================================================================
set -u
RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; RST=$'\e[0m'
fail() { echo "${RED}FAIL-GUARD:${RST} $*" >&2; exit 1; }
pass() { echo "${GRN}PASS${RST} $*"; }
warn() { echo "${YLW}WARN${RST} $*"; }
PORT=${PORT:-30001}

gate_format() {
  local d="$1"
  [ -d "$d" ] || fail "model dir not found: $d"
  [ -f "$d/config.json" ] || fail "no config.json under $d"
  local cfg; cfg=$(readlink -f "$d/config.json")
  local out rc
  out=$(python3 - "$cfg" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
q = d.get('quantization_config', {}) or {}
g = (q.get('config_groups') or {}).get('group_0', {}) if isinstance(q.get('config_groups'), dict) else {}
fmt = g.get('format') or q.get('format')
bits = g.get('input_activations', {}).get('num_bits')
print(f"format={fmt} bits={bits}")
sys.exit(0 if (fmt == 'nvfp4-pack-quantized' and str(bits) == '4') else 1)
PY
  ) && rc=0 || rc=$?
  echo "  checkpoint says: $out"
  [ $rc -eq 0 ] || fail "NOT true NVFP4 -> would produce gibberish + invalid numbers. Aborting."
  pass "GATE 1 (format): true NVFP4 (nvfp4-pack-quantized W4A4)"
}

gate_boot() {
  local name="$1" mdir="$2" p="${3:-$PORT}"
  docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -qx "$name" || fail "container '$name' not running"
  local base; base=$(basename "$(readlink -f "$mdir")")
  for i in $(seq 1 60); do
    sleep 15
    if curl -s -m 5 "http://127.0.0.1:$p/get_model_info" | grep -q "$base"; then
      pass "GATE 2 (boot): serving exactly $base on :$p"; return 0
    fi
    if docker logs "$name" 2>&1 | grep -q Traceback; then
      docker logs "$name" 2>&1 | grep -B2 -A10 Traceback | tail -14
      fail "server crashed during boot"
    fi
  done
  fail "boot timeout on :$p"
}

gate_quality() {
  local p="${1:-$PORT}" tmp; tmp=$(mktemp -d)
  q() { curl -s -m 120 "http://127.0.0.1:$p/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen38-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; }
  q "Return a JSON object with keys name, age, and city filled with realistic values. Only the JSON." 200 > "$tmp/j.json"
  if python3 - "$tmp/j.json" <<'PY'
import json, sys, re
c = json.load(open(sys.argv[1]))["choices"][0]["message"].get("content", "") or ""
m = re.search(r"\{.*\}", c, re.S)
j = json.loads(m.group(0)) if m else {}
assert j.get("name") and j.get("age") and j.get("city"), f"empty-field junk: {c[:120]!r}"
PY
  then pass "GATE 3a (JSON): fields populated"; else fail "JSON junk (empty fields)"; fi
  A=$(q "What is 17 times 23? Answer with only the number." 100 | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message'].get('content',''))")
  echo "$A" | grep -q "391" && pass "GATE 3b (math): 391 correct" || fail "math broken: '$A'"
  q "Write a detailed story about a robot learning to paint." 400 > "$tmp/s.json"
  if python3 - "$tmp/s.json" <<'PY'
import json, sys
from collections import Counter
c = json.load(open(sys.argv[1]))["choices"][0]["message"].get("content", "") or ""
grams = Counter(c[i:i+48] for i in range(0, max(len(c)-48, 1), 8))
worst = grams.most_common(1)[0][1] if grams else 0
assert worst < 4, f"repetition loop ({worst}x same 48-char span)"
PY
  then pass "GATE 3c (loops): none"; else fail "repetition loop detected"; fi
  rm -rf "$tmp"
  pass "GATE 3 (quality): CLEAN"
}

gate_clock() {
  local p="${1:-$PORT}"
  curl -s -m 90 "http://127.0.0.1:$p/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"Write a long story about a robot learning to paint."}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' &
  local cpid=$!
  sleep 8
  local clk; clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
  wait $cpid
  if [ "${clk:-0}" -lt 2000 ]; then
    warn "GATE 4 (clock): ${clk} MHz under load - THROTTLED. Speed NOT comparable (~2400MHz normal). Quality still valid."
    echo "CLOCK_FLAG=${clk}MHz THROTTLED"
  else
    pass "GATE 4 (clock): ${clk} MHz under load - speed valid"
    echo "CLOCK_FLAG=${clk}MHz OK"
  fi
}

case "${1:-}" in
  check) gate_format "$2" ;;
  launch)
    gate_format "$2"
    NAME=${NAME:-sweep}; PORT=${PORT:-30001}; MEM_FRACTION=${MEM_FRACTION:-0.90}
    IMAGE=${IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2-minoverlay}
    HFCACHE=${HFCACHE:-$HOME/DFlash2/.cache/huggingface}
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all --ipc host --network host \
      --memory 100g --memory-swap 100g \
      -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
      --mount type=bind,source="$2",target=/model \
      --mount type=bind,source="$HFCACHE",target=/root/.cache/huggingface \
      "$IMAGE" python3 -m sglang.launch_server \
      --model-path /model --served-model-name qwen38-27b --trust-remote-code \
      --mem-fraction-static "$MEM_FRACTION" \
      --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph \
      --kv-cache-dtype fp8_e4m3 \
      --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
      --context-length 262144 \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
      --speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
      --speculative-num-draft-tokens 8 \
      --max-prefill-tokens 16384 --mamba-full-memory-ratio 4.21 --max-mamba-cache-size 40 --max-running-requests 10 \
      --reasoning-parser qwen3 \
      --host 0.0.0.0 --port "$PORT" >/dev/null \
      && pass "launched '$NAME' (gated NVFP4-only)" || fail "docker run failed" ;;
  verify-boot) gate_boot "$2" "$3" "${4:-$PORT}" ;;
  quality)     gate_quality "${2:-$PORT}" ;;
  clock)       gate_clock "${2:-$PORT}" ;;
  full)
    gate_format "$2"
    NAME=${NAME:-sweep}
    "$0" launch "$2" && "$0" verify-boot "$NAME" "$2" "${3:-$PORT}" \
      && "$0" quality "${3:-$PORT}" && "$0" clock "${3:-$PORT}" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
