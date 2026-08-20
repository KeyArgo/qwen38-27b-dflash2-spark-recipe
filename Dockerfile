# Build Qwen3.8-27B with DFlash2 (decoupled speculative decoding) on the
# official SGLang qwen38-27b runtime image.
#
# This image only overrides the five Python modules needed for DFlash2 that
# are not yet present in the pinned runtime. Everything else comes from the
# upstream runtime layer unchanged.
FROM lmsysorg/sglang:qwen38-27b

COPY overlay/sglang/kernels/ops/speculative/dflash.py \
     /sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py
COPY overlay/sglang/srt/models/dflash.py \
     /sgl-workspace/sglang/python/sglang/srt/models/dflash.py
COPY overlay/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py \
     /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
COPY overlay/sglang/srt/speculative/dflash_utils.py \
     /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
COPY overlay/sglang/srt/speculative/dflash_worker_v2.py \
     /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py
