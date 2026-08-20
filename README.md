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
| `patches/`   | Unified diffs of `overlay/` against the upstream base, for audit |
| `scripts/build.sh` | Portable local build driver (Docker only) |
| `.github/workflows/build.yml` | CI build on any GitHub runner |
| `overlay/MANIFEST.sha256` | Pin of the overlay bytes, verified at build |

## Why the base image needs this

Upstream SGLang merged DFlash2 after the `qwen38-27b` runtime was tagged, so
the pinned image does not ship it. This repo supplies the missing modules and
adapts them to the runtime's API surface. No released-image changes or forks
are required — just these files.

## Build

Requires Docker (BuildKit auto-enabled by `scripts/build.sh`).

```bash
./scripts/build.sh
# IMAGE_TAG=my/image:tag ./scripts/build.sh   # custom tag
# DOCKER=podman ./scripts/build.sh            # non-Docker driver
```

BuildKit and GHA cache are wired into the workflow, so CI builds are fast.

## Run

All normal SGLang speculative args apply. Pass the draft as a repo id so the
hub snapshot resolver loads it, then enable DFLASH for block size:

```bash
docker run --gpus all --ipc host --network host \
  -v /path/to/target-model:/model \
  qwen38-27b-dflash2:local \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name qwen38-27b \
    --host 0.0.0.0 --port 30000 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path <draft-repo-id> \
    --speculative-draft-model-revision <rev> \
    --speculative-num-draft-tokens 16
```

> Pass the draft model as a **repo id + revision**, not a bare local path —
> the config resolver needs the hub snapshot route. If the draft repo is
> already in the local HF cache, set `HF_HUB_CACHE` so no network is needed.

## Throughput

This recipe was validated end-to-end on a single NVIDIA DGX Spark (GB10) with
block size 16 and `mem 0.90`. Sustained decode throughput is materially higher
than non-speculative serving on the same box. Exact figures vary with hardware,
batch, and load, so they are intentionally not published here.

## License

Apache-2.0. Files under `overlay/` are derived from the SGLang project
(https://github.com/sgl-project/sglang); see `NOTICE`. The base runtime image
is governed by the NVIDIA Deep Learning Container License.
