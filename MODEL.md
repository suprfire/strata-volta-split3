# The model mod — NVFP4 → Volta-friendly split GGUF

The served pack (`split40/qwen38fn-split-q4.gguf`, 127,864,594,176 bytes) is **not** an upstream artifact: it is produced by a converter written for this box, `scripts/split_pack.py`, from a Strata NVFP4 pack.

## Why a model mod is needed at all

NVFP4 weights need Blackwell tensor cores. Volta (`sm_70`) has neither NVFP4 nor FP8 hardware paths, and the all-cards split engine consumes ordinary GGML tensors. So the model is **dequantized from NVFP4/FP8 and re-quantized to Q4_0/Q8_0** — quantizations Volta runs well — in one pass, layer by layer.

## Pipeline

```
NVIDIA Qwen3.8-Flash-Next checkpoint (BF16 + NVFP4-GPTQ release)
        │  upstream Strata tools (v0.2.14 tree)
        ▼
pack/  ── dense.bin · experts.bin · index.txt · native_experts.txt · tokenizer/
        │  scripts/split_pack.py                ← the mod
        ▼
split40/qwen38fn-split-q4.gguf                  ── served by strata-split-3.exe
```

- **Upstream stage.** `tools/iq_pack.py` builds the pack index (`pack index v3`, 1079 dense tensors, align 64) from `qwen-nvfp4-gptq-dense.gguf`; `native_experts.txt` records the expert blob table (512 experts/layer, 67,948,118,016 B total; experts were GPTQ-requantized from the BF16 checkpoint upstream via `tools/requant.py`). The NVFP4-Local (QAD) conversion path upstream is `tools/nvfp4_local_pack.py` (`requirements-nvfp4-local.txt`: torch 2.9.1 CPU / transformers 4.57.6 / safetensors 0.7.0). A ready-made Strata-compatible pack for this model also exists on Hugging Face ([Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)) — that is the directory we consume. Weight chain per its model card: `Qwen/Qwen3.8-Flash-Next` (BF16) → GPTQ NVFP4; the PLE table is Qwen's own FP8 file ([Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)).
- **Helper artifacts kept beside the pack** (used by the vision engine and as converter inputs): `qwen-nvfp4-gptq-dense.gguf` (6.0 GB — the vision `model` path), `ple-fp8.gguf` (51.2 GB, FP8 E4M3 PLE tensor), `token-embd-bf16.gguf` (1.27 GB, duplicate of the dense embedding).

## Type mapping applied by `split_pack.py`

| Source | Destination (split GGUF) |
|---|---|
| MoE experts, NVFP4 blobs | **Q4_0** (decode → re-encode per expert) |
| Dense BF16 matrices | **Q8_0** |
| indexer / hyper-connection (`hc_*`) | stays **BF16** |
| other BF16/F16 | **F32** |
| PLE FP8 (E4M3, one 320M-element tensor) | **Q8_0** (flat 1M-element chunking — row-chunking would need ~200 GB RAM) |
| token_embd BF16 | **Q8_0** (taken from the dense stream; the emb file is a duplicate) |

Emitted tensor names follow what the split loader expects (`blk.N.ffn_{gate,up,down}_exps.weight`, dims innermost-first).

## NVFP4 blob decode semantics (the part we got wrong first)

Per expert blob: `gate rows | up rows | down rows | tail {s_gate, s_up, s_down, 0}` (fp32).
Row bytes: `cols/64` blocks of **36 bytes = 4 UE4M3 scales FIRST, then 32 nibble bytes**; each 8-byte sub-block packs **low nibbles = values 0..7, high nibbles = 8..15** into the E2M1 sign/magnitude table. Scales are **UE4M3 unsigned** — bit 7 ignored, `0x7F` decodes to `0`.

The first conversion attempt decoded with a Q4_0-style nibble interleave and signed E4M3; `scripts/rewrite_experts.py` rewrites the 144 expert tensors (48 layers × 3 projections) in place with the corrected decode. New packs should run `split_pack.py` (it already has the fixed decode) and skip the rewrite.

## Run it

```
python scripts/split_pack.py <packdir> <outdir>
```

Notes:
- Run on the serving box (needs ~free RAM for 67 GB of expert mmap; numpy only otherwise).
- It imports the llama.cpp-style `gguf` python writer/reader module; `rewrite_experts.py` has a `sys.path` line pointing at our tree — adjust to wherever your forked `gguf` module lives.
- `split_pack.py` runs a `selftest()` before the main conversion; if it prints `SPLITPACK-DONE <path> <bytes>`, the pack is complete and the byte count should match what you expect (~128 GB for this model).

## What we did not check

- Numerical quality impact of NVFP4→Q4_0 (we did not run evals against the NVFP4 original).
- Whether `tools/requant.py` (upstream) can replace the custom converter for split-engine packs — it is absent from our v0.2.14 tree listing, its replacement may change the pipeline.
