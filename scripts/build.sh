#!/usr/bin/env bash
# Build the Qwen3.8-27B DFlash2 image locally.
#
# Portable: needs only a working Docker (or buildx). Produces the same image
# from the pinned base + the overlay/ tree in this repo. No network fetch of
# SGLang source, no git checkout at build time.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_image="${BASE_IMAGE:-lmsysorg/sglang:qwen38-27b}"
tag="${IMAGE_TAG:-qwen38-27b-dflash2:local}"
docker_bin="${DOCKER:-docker}"

cd "$repo_dir"

# sanity: overlay tree must be complete before we build
required=(
  overlay/sglang/kernels/ops/speculative/dflash.py
  overlay/sglang/srt/models/dflash.py
  overlay/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
  overlay/sglang/srt/speculative/dflash_utils.py
  overlay/sglang/srt/speculative/dflash_worker_v2.py
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f - overlay tree incomplete" >&2; exit 1; }
done
# verify the overlay against the audited manifest if present
if [[ -f overlay/MANIFEST.sha256 ]]; then
  ( cd overlay && sha256sum -c MANIFEST.sha256 ) >/dev/null || {
    echo "overlay checksum mismatch - refusing to build" >&2; exit 1; }
fi

export DOCKER_BUILDKIT=1
"$docker_bin" build -t "$tag" -f Dockerfile "$repo_dir"
echo "built $tag"
