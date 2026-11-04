# Third-party source

`qwen3_vl_embedding.py` is an unmodified copy from the official Qwen repository:

- Project: https://github.com/QwenLM/Qwen3-VL-Embedding
- Commit: `393e2978d27852b0d0230d6994f37f9c15bed73c`
- Source file: `src/models/qwen3_vl_embedding.py`
- License: Apache-2.0; original text in `LICENSE`.
- SHA256: `0e447b6a80e05527f3fd437ab4f1617a6e7d0df102b8164348bbce787a41b267`

Project-owned adaptations are in `src/multimodal_ecommerce/embedding.py`:
strict preprocessing errors instead of upstream's NULL fallback, local image
inputs only, BF16/SDPA, one input at a time, pixel/token limits, and diagnostics.
The original vendored file is excluded from project formatting and lint commands
so its pinned checksum remains reproducible.
