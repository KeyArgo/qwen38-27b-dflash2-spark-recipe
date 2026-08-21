# Qwen3.8-27B · DFlash2 recipe (DGX Spark / GB10)

A self-contained, reproducible **recipe** that adds **DFlash2** (decoupled
speculative decoding) to the official SGLang `qwen38-27b` runtime image, tuned
and validated for a single **NVIDIA DGX Spark (GB10)**.

DFlash2 is a speculative-decoding draft that proposes several tokens ahead
with a small parallel draft head, then verifies them against the target model
in one pass. On the target model this raises sustained decode throughput well
above non-speculative serving. This repo only adds the five Python modules the
runtime is missing; every other layer comes from the upstream image unchanged,
so `fork` kernels, quantization (NVFP4) and CUDA-graph handling are preserved.

## What is in this repo

| Path | Purpose |
|------|---------|
| `Dockerfile` | `FROM lmsysorg/sglang:qwen38-27b`, overlays the 5 modules |
| `overlay/`   | The 5 DFlash2 Python modules (the adapted source) |
| `scripts/build.sh` | Portable local build driver (Docker only) |
| `scripts/run.sh`   | Validated one-command launcher (all flags + safety checks) |
| `scripts/nvfp4-guard.sh` | Checkpoint-format / boot / quality / clock gates (run before quoting any result) |
| `.github/workflows/build.yml` | CI build on any GitHub runner |
| `overlay/MANIFEST.sha256` | Pin of the overlay bytes, verified at build |

## Why the base image needs this

Upstream SGLang merged DFlash2 after the `qwen38-27b` runtime was tagged, so
the pinned image does not ship it. This repo supplies the missing modules and
adapts them to the runtime's API surface. No released-image changes or forks
are required — just these files.

## 1. Build

Requires Docker (BuildKit auto-enabled by `scripts/build.sh`).

```bash
./scripts/build.sh
# IMAGE_TAG=my/image:tag ./scripts/build.sh   # custom tag
# DOCKER=podman ./scripts/build.sh            # non-Docker driver
```

The build verifies `overlay/` against `MANIFEST.sha256` and refuses to build
on mismatch. The base image (`lmsysorg/sglang:qwen38-27b`, ~38.6GB) is pulled
over your own internet on first build — allow a few minutes.

> **On a box where `github.com` is blocked** (e.g. a restricted-subnet fleet
> box): fetch each file individually from `raw.githubusercontent.com` instead
> of `git clone` — the repo has no build-time network dependency beyond the
> base image, so a plain file fetch + `./scripts/build.sh` works identically.

## 2. Get the models

Two pieces, both public on Hugging Face:

- **Target weights** — `sakamakismile/Qwen3.8-27B-MTP-NVFP4` (~20GB), a
  **verified true-NVFP4** (`nvfp4-pack-quantized`, W4A4) quantization of the
  original `Qwen/Qwen3.8-27B` with the native MTP head preserved in bf16.
  Either a local model dir or an HF cache entry works.
  (Uncensored alternative with measured-identical scores:
  `sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4`.)

> **Warning — fake-NVFP4 checkpoints.** `RadixArk/Qwen3.8-27B-NVFP4` and
> `unsloth/Qwen3.8-27B-NVFP4` are actually **FP8** (`float-quantized`, 8-bit)
> checkpoints mislabeled as NVFP4 (and byte-identical re-uploads of each other).
> Served with this recipe they produce deterministic gibberish: empty-field
> JSON loops (`{"name": "", "", ""}`), repeated fragments, stray tags.
> Always verify a downloaded target before use:
>
> ```bash
> ./scripts/nvfp4-guard.sh check /path/to/model_dir
> # must PASS: format=nvfp4-pack-quantized bits=4
> ```
- **DFlash2 draft** — `z-lab/Qwen3.8-27B-DFlash2` (~2.6GB), revision-pinned:
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Download into a local HF cache:

```bash
mkdir -p ~/DFlash2/.cache/huggingface/hub
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir ~/DFlash2/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

(`hf` = `pip install -U "huggingface_hub[cli]"`. Unauthenticated HF downloads
are rate-limited; set `HF_TOKEN` for the download process only if it stalls.)

## 3. Run

```bash
TARGET_MODEL=/path/to/Qwen3.8-27B-NVFP4 \
HF_CACHE=$HOME/DFlash2/.cache/huggingface \
./scripts/run.sh
```

`run.sh` bakes in the validated DFlash2 config: block size 16, `mem 0.90`,
`extra_buffer` mamba radix cache, 262144 context, fp8 KV, and a
`--memory 100g` container cap (the GB10 freeze trap). It refuses
`mem-fraction >= 0.95` (that value hard-reboots GB10 during draft-graph
capture) and sanity-checks that image + models exist before launching.

All knobs are env-overridable: `IMAGE`, `TARGET_MODEL`, `HF_CACHE`, `PORT`,
`MEM_FRACTION`, `DRAFT_REPO`, `DRAFT_REV`, `NAME`. Serves on
`http://<host>:30000` with served name `qwen38-27b`.

Wait for `The server is fired up and ready to roll!` in
`docker logs -f qwen38-dflash2` — cold start is ~3-4 minutes on GB10
(weight load ~160s + CUDA-graph capture ~30s).

## 4. Benchmark

```bash
BASE_URL=http://127.0.0.1:30000 MODEL=qwen38-27b ./bench-matrix.sh
```

Before quoting any number: gate quality and clock first —

```bash
./scripts/nvfp4-guard.sh quality 30000   # junk detectors must PASS
./scripts/nvfp4-guard.sh clock 30000     # >=2000 MHz under load, else speeds invalid
```

(`bench-matrix.sh` is the frozen engine-agnostic battery from
`hasso5703/dgx-spark-qwen38`: 8 workloads × EN/FR/DE, greedy, two-call delta
net of prefill. Results are comparable across engines and boxes.)

## Throughput

Measured with **true-NVFP4 checkpoints, quality-gated** (populated JSON fields,
correct arithmetic anchor, no repetition loops) and **SM clock sampled under
load** so throttled boxes can't pollute numbers.

| Config | Clock | math (EN) | code (EN) | code (DE) | tech (FR) | reason (FR) | prose (EN) | prose (FR) | prose (DE) |
|---|---|---|---|---|---|---|---|---|---|
| **DFlash2 + true NVFP4** (AEON), box A, two runs avg | 2496 MHz | **45.6** | **37.4** | **23.4** | **29.8** | **42.9** | **20.3** | **19.9** | **16.3** |

All values tok/s, decode net of prefill, greedy, frozen battery. Two independent
runs reproduced within 0.3 tok/s. Graded answer quality is identical to
non-speculative serving — DFlash2 changes speed, not outputs (same answers,
right and wrong, byte-for-byte, on a 10-question graded battery).

Raw evidence: `results/results-A-aeon-nvfp4-clock2496.json`.

### Historical numbers (invalid — kept for the record)

The previously published table (math ~117 / code ~83 / prose ~51, boxes A+B at
~2400 MHz) was measured against `RadixArk/Qwen3.8-27B-NVFP4`, which later
proved to be an **FP8 checkpoint mislabeled as NVFP4** that produces
deterministic gibberish with this recipe (see the warning above). Those
throughputs are real measurements of a broken model — not achievable by any
usable configuration and not comparable to the gated numbers above.

## License

Apache-2.0. Files under `overlay/` are derived from the SGLang project
(https://github.com/sgl-project/sglang); see `NOTICE`. The base runtime image
is governed by the NVIDIA Deep Learning Container License.
