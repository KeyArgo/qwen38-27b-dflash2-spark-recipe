#!/usr/bin/env bash
# ============================================================================
#  spark-bench.sh — Fleet readiness check + endpoint quality bench (v1)
#
#  Part 1  Fleet check: is every Spark up, what is it serving, GPU clocks,
#          VRAM in use. Verifies "all sparks are loaded".
#  Part 2  Quality bench: runs a 10-probe battery (factual, code, JSON, math,
#          language, instruction-following, long coherence, stability x2)
#          against ANY OpenAI-compatible endpoint (default: a2best on A).
#
#  Usage:
#    ./spark-bench.sh                # fleet check + bench a2best (A:30003)
#    ./spark-bench.sh --bench-only   # skip fleet check
#    ./spark-bench.sh --fleet-only   # skip the quality bench
#    BENCH_URL=http://host:port BENCH_MODEL=name ./spark-bench.sh
#
#  Needs: ssh (argo@<spark>), curl, python3. No other deps.
#  NOTE: the default bench target is the a2best-judge endpoint on Spark A.
#  Point BENCH_URL/BENCH_MODEL at any OpenAI-compatible endpoint to bench it instead.
# ============================================================================
set -uo pipefail

# ---- config ----------------------------------------------------------------
BENCH_URL="${BENCH_URL:-http://10.0.0.50:30003}"
BENCH_MODEL="${BENCH_MODEL:-a2-best-abliterated}"
BENCH_HOST="$(echo "$BENCH_URL" | sed -E 's|https?://([^:/]+).*|\1|')"
BENCH_PORT="$(echo "$BENCH_URL" | sed -E 's|https?://[^:/]+:?([0-9]*).*|\1|')"
BENCH_PORT="${BENCH_PORT:-80}"
SSH_OPTS="-o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new"

# spark topology: name:primary_ip:alt_ip
SPARKS=(
  "A:10.0.0.50:"
  "B:10.0.0.51:"
  "E:10.0.0.52:"
  "C:192.168.50.20:"
  "D:192.168.50.21:"
)

DO_FLEET=1; DO_BENCH=1
for a in "$@"; do
  case "$a" in
    --fleet-only) DO_BENCH=0 ;;
    --bench-only) DO_FLEET=0 ;;
  esac
done

# ---- helpers ---------------------------------------------------------------
BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; CYAN='\033[36m'; NC='\033[0m'
ok()   { printf "${GREEN}%-6s${NC} %s\n" "PASS" "$1"; }
bad()  { printf "${RED}%-6s${NC} %s\n" "FAIL" "$1"; }
warn() { printf "${YELLOW}%-6s${NC} %s\n" "WARN" "$1"; }
info() { printf "${CYAN}%s${NC}\n" "$1"; }

# get_api_key — extract a2best api key from A's docker config (never printed)
get_api_key() {
  ssh $SSH_OPTS argo@10.0.0.50 \
    "docker inspect a2best-judge --format '{{range .Config.Cmd}}{{.}} {{end}}'" 2>/dev/null \
    | grep -oP -- '--api-key \K\S+' 2>/dev/null
}

# curl_completion — POST a chat completion, print raw JSON
curl_completion() {  # $1=prompt  $2=max_tokens
  local prompt="$1" maxtok="${2:-200}"
  python3 - "$BENCH_URL" "$BENCH_MODEL" "$prompt" "$maxtok" <<'PYEOF'
import json, sys, urllib.request, os
url, model, prompt, maxtok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
key = os.environ.get("BENCH_API_KEY", "")
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": maxtok, "temperature": 0}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", data=body,
      headers={"Content-Type": "application/json",
               **({"Authorization": "Bearer " + key} if key else {})})
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        print(r.read().decode())
except Exception as e:
    print(json.dumps({"error": str(e)}))
PYEOF
}

# extract_content — pull the content field from a completion JSON
extract_content() {  # stdin: JSON
  python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d["choices"][0]["message"].get("content") or "")
except Exception:
    print("")'
}

# ---- Part 1: Fleet check ----------------------------------------------------
if [ "$DO_FLEET" = 1 ]; then
  info ""
  info "======================================================================"
  info "  PART 1 — FLEET READINESS  (all 5 sparks)"
  info "======================================================================"
  for entry in "${SPARKS[@]}"; do
    name="${entry%%:*}"; rest="${entry#*:}"; ip="${rest%%:*}"; alt="${rest#*:}"
    host="$ip"
    if ! ping -c1 -W2 "$ip" >/dev/null 2>&1 && [ -n "$alt" ]; then host="$alt"; fi
    printf "\n${BOLD}Spark %s${NC}  (%s)\n" "$name" "$host"
    if ! ssh $SSH_OPTS argo@"$host" true 2>/dev/null; then
      bad "ssh unreachable"
      continue
    fi
    # what's running: containers, GPU, listening model ports
    state=$(ssh $SSH_OPTS argo@"$host" '
      echo "CTRS:"; docker ps --format "  {{.Names}} | {{.Image}} | {{.Status}}" 2>/dev/null | head -6
      echo "GPU:"; nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null
      echo "APPS:"; nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | head -4
      echo "PORTS:"; ss -tln 2>/dev/null | grep -oE ":(30000|30003|8096|8099|11434|4000) " | sort -u | tr "\n" " "
    ' 2>/dev/null)
    if [ -z "$state" ]; then bad "no state returned"; continue; fi
    echo "$state" | sed 's/^/  /'
    # serving check on any found model port
    served=""
    for p in 30000 30003 8096 8099; do
      if [ "$name" = "A" ] && [ "$p" = "30003" ]; then
        k=$(get_api_key)
        m=$(curl -s -m 4 -H "Authorization: Bearer $k" "http://$host:$p/v1/models" 2>/dev/null | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print(",".join(x["id"] for x in d.get("data",[])))
except Exception: print("")' 2>/dev/null)
      else
        m=$(curl -s -m 4 "http://$host:$p/v1/models" 2>/dev/null | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print(",".join(x["id"] for x in d.get("data",[])))
except Exception: print("")' 2>/dev/null)
      fi
      [ -n "$m" ] && served="$served port $p: $m;"
    done
    if [ -n "$served" ]; then ok "serving:${served}"; else warn "no OpenAI model listener found"; fi
  done
fi

# ---- Part 2: Quality bench ---------------------------------------------------
if [ "$DO_BENCH" = 1 ]; then
  info ""
  info "======================================================================"
  info "  PART 2 — ENDPOINT QUALITY BENCH"
  info "  URL:   $BENCH_URL   model: $BENCH_MODEL"
  info "======================================================================"
  # auth: only for the default a2best endpoint on A
  BENCH_API_KEY=""
  if [ "$BENCH_HOST" = "10.0.0.50" ] && [ "$BENCH_PORT" = "30003" ]; then
    BENCH_API_KEY="$(get_api_key)"
    [ -n "$BENCH_API_KEY" ] && info "  auth: key extracted from A (len ${#BENCH_API_KEY})" \
                             || warn "  auth: no key found; endpoint may 401"
  fi
  export BENCH_API_KEY

  # probe list: name|prompt|max_tokens|expected-marker
  probes=(
    "factual|What is the capital of France? Answer in one sentence.|120|Paris"
    "code|Write a python function that reverses a string.|200|def"
    "instruction|Respond ONLY with the word banana.|40|banana"
    "language-es|Explica en espanol que es la gravedad en una frase.|150|"
    "json|Return a JSON object with keys name, age, and city. Only the JSON.|120|"
    "math|What is 17 times 23? Answer with only the number, no explanation.|40|391"
    "summary|Summarize in 2 sentences: The water cycle describes how water evaporates from oceans, condenses into clouds, falls as precipitation, and returns to oceans, driven by solar energy.|150|"
    "coherence|Write a 3-paragraph essay on why the sky is blue.|400|"
  )

  printf "\n${BOLD}%-16s %-8s %-10s %s${NC}\n" "PROBE" "STATUS" "TOKENS" "NOTES"
  printf "%-16s %-8s %-10s %s\n" "----------------" "------" "----------" "--------------------------------"
  pass=0; fail=0
  for p in "${probes[@]}"; do
    IFS='|' read -r name prompt maxtok marker <<< "$p"
    raw=$(curl_completion "$prompt" "$maxtok")
    content=$(echo "$raw" | extract_content)
    ntokens=$(echo "$raw" | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens","?"))
except Exception: print("?")' 2>/dev/null)
    clen=${#content}
    if [ -z "$content" ] || [ "$clen" -lt 2 ]; then
      bad "$name"; echo "    (empty or missing content)"; fail=$((fail+1)); continue
    fi
    note=""
    if [ -n "$marker" ]; then
      if echo "$content" | grep -qi "$marker"; then ok "$name"; pass=$((pass+1));
      else bad "$name"; note="missing marker '$marker'"; fail=$((fail+1)); fi
    elif [ "$name" = "json" ]; then
      if echo "$content" | python3 -c 'import sys,json
try: json.loads(sys.stdin.read()); print(1)
except Exception: print(0)' 2>/dev/null | grep -q 1; then ok "$name"; pass=$((pass+1));
      else bad "$name"; note="not valid JSON"; fail=$((fail+1)); fi
    elif [ "$name" = "coherence" ]; then
      if [ "$clen" -gt 300 ]; then ok "$name"; pass=$((pass+1));
      else warn "$name"; note="short (${clen}B)"; fi
    else
      ok "$name"; pass=$((pass+1))
    fi
    printf "    %-40.40s…\n" "$content"
    [ -n "$note" ] && printf "    ${YELLOW}%s${NC}\n" "$note"
  done

  # ---- stability: same prompt twice at temp 0 -------------------------------
  printf "\n${BOLD}STABILITY (same prompt, temperature 0, run twice)${NC}\n"
  for q in "What is the capital of France? Answer in one sentence." \
           "Write a python function that reverses a string."; do
    r1=$(curl_completion "$q" 120 | extract_content)
    r2=$(curl_completion "$q" 120 | extract_content)
    if [ -n "$r1" ] && [ "$r1" = "$r2" ]; then ok "identical output (${#r1}B)"
    elif [ -n "$r1" ] && [ -n "$r2" ]; then warn "outputs differ (${#r1}B vs ${#r2}B)"
    else bad "empty output in stability run"; fi
    printf "    run1: %.60s…\n    run2: %.60s…\n" "$r1" "$r2"
  done

  # ---- summary ----------------------------------------------------------------
  printf "\n======================================================================\n"
  if [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]; then
    printf "  ${GREEN}VERDICT: %d/%d probes passed — endpoint returns good responses${NC}\n" "$pass" "$((pass+fail))"
  else
    printf "  ${RED}VERDICT: %d passed, %d failed — inspect failures above${NC}\n" "$pass" "$fail"
  fi
  printf "======================================================================\n"
fi
echo ""
