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

- **Target weights** — `RadixArk/Qwen3.8-27B-NVFP4` (~22GB), the NVFP4
  checkpoint this recipe is tuned for. Either a local model dir or an HF
  cache entry works.
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

(`bench-matrix.sh` is the frozen engine-agnostic battery from
`hasso5703/dgx-spark-qwen38`: 8 workloads × EN/FR/DE, greedy, two-call delta
net of prefill. Results are comparable across engines and boxes.)

## Throughput

Measured on two DGX Spark (GB10) boxes, same image, same flags, GPU clocks
verified under load (A 2411 MHz, B 2398 MHz). Each box: two warm runs of the
frozen battery, averaged. Cold-boot first runs are excluded (radix-cache warmup
inflates short-workload numbers; e.g. tech-FR read 180 on both boxes cold vs
73 warm). All values tok/s, decode net of prefill.

| Box | math (EN) | code (EN) | code (DE) | tech (FR) | reason (FR) | prose (EN) | prose (FR) |
|---|---|---|---|---|---|---|---|
| A | 117.0 | 85.0 | 103.4 | 73.3 | 86.5 | 50.8 | 56.1 |
| B | 116.3 | 81.3 | 102.6 | 73.1 | 86.2 | 50.6 | 55.9 |
| **avg** | **116.7** | **83.2** | **103.0** | **73.2** | **86.3** | **50.7** | **56.0** |

prose (DE) is consistently skipped by the battery's guard (short-answer delta
artifact, Δtok≈0) — same on both boxes; not a failure.

Raw per-run JSONs (bench-matrix-dflash2-a-warm.json, bench-matrix-dflash2-b-final.json)
live on the boxes.

## License

Apache-2.0. Files under `overlay/` are derived from the SGLang project
(https://github.com/sgl-project/sglang); see `NOTICE`. The base runtime image
is governed by the NVIDIA Deep Learning Container License.
