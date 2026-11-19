# Strata Volta Split3 — 262k context on three 32GB Volta cards

A Volta-GPU deployment of [Strata](https://github.com/Niko1221/Strata) using its **all-cards split engine**, plus small serve-side patches that make VRAM-fit and telemetry behave on a multi-Volta machine.

This repo is **not a fork of the Strata source**. It documents exactly what differs from a stock `Niko1221/Strata` checkout, ships the serve config + launcher that do the work, and gives the CMake flags to build the Volta split engine binary yourself.

**Scope: this is a Windows deployment.** Everything here was built, tuned, and verified on Windows 11 — the engine is an `.exe` compiled with MSVC (`vcvars64`) + CUDA on Windows, the launcher is a `.bat`, and the config uses Windows paths and CUDA DLL `lib_dirs`. The pieces are portable in principle (the CMake flags work with any `nvcc`; `split_pack.py` is pure numpy/python and the serve patches are platform-neutral Python), but Linux has **not** been tested with this combination — treat a Linux port as unsupported territory, not a checkbox.

| | |
|---|---|
| Upstream base | `Niko1221/Strata` @ tag `v0.2.14` (commit `09b603886`) |
| Hardware | 2× Tesla V100-PCIE-32GB + 1× Tesla PG500-216 — all three report compute capability **7.0 (Volta)**, 96 GB VRAM total |
| Model served | Qwen3.8-Flash-Next, NVFP4-GPTQ origin → split GGUF with Q4_0 experts (`split40` pack) |
| Context | 262 144 tokens, batch 4, spec 4, MTP drafter, vision on a dedicated card |
| OS / toolchain | Windows 11, Python 3.12, CUDA Toolkit 12.9, VS 2022 BuildTools + Ninja |

## What you get

- One OpenAI/Anthropic-compatible endpoint (`:8080`, `0.0.0.0`) backed by **all three GPUs working as one engine**.
- 262k-token context that no single 32GB card could hold — the split engine shards the model across the three Volta cards with peer-to-peer transfers.
- Vision (image input) pinned to its own card so the reader doesn't fight the engine's ring.
- VRAM/telemetry visibility for all three cards in the Strata monitor and CLI.

## What this repo contains

```
README.md                 ← you are here
MODIFICATIONS.md          ← serve-side diffs vs stock v0.2.14, with rationale
ENGINE.md                 ← how to build strata-split-3.exe for sm_70 (Volta)
MODEL.md                  ← the model mod: NVFP4 pack → Q4_0/Q8_0 split GGUF (Volta can't run NVFP4)
configs/
  strata-nvfp4-split3-262k-4096p4.json   ← the live serve config
scripts/
  launch-live.bat         ← the launcher (env vars + serve invocation)
  split_pack.py           ← the NVFP4→split-GGUF converter (run on the box)
  rewrite_experts.py      ← one-time in-place expert-decode fixer (history; split_pack.py already contains the fix)
patches/
  volta-serve.patch       ← the serve-side patch, apply to a stock v0.2.14 tree
LICENSE                   ← upstream MIT (Niko1221 and the Strata contributors)
```

## Reproduce it

> **Path placeholders:** `<ROOT>` = the directory where you keep the Strata tree and model data; `<CUDA_PATH>` = your CUDA Toolkit install (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`). Every path in `configs/` and the docs is written against these — substitute your own.

1. **Check your cards are Volta:**
   ```
   nvidia-smi --query-gpu=index,name,compute_cap --format=csv
   ```
   Every card must report `7.0`. Note: the NVIDIA 615+ driver branch drops Volta support — use the 580/582-era driver branch.

2. **Check out stock Strata at the base tag and apply the patches:**
   ```
   git clone https://github.com/Niko1221/Strata
   cd Strata
   git checkout v0.2.14
   git apply /path/to/patches/volta-serve.patch
   ```
   Then install (Python 3.12): create a venv and `pip install -r requirements.txt`.

3. **Build the Volta split engine** — see [ENGINE.md](ENGINE.md). Produces `strata-split-3.exe` compiled for `sm_70`.

4. **Convert the model** — Volta cannot run NVFP4 weights; requantize the pack with `scripts/split_pack.py` — full pipeline, type mapping, and gotchas in [MODEL.md](MODEL.md).

5. **Point the config at your paths.** Edit `configs/strata-nvfp4-split3-262k-4096p4.json`:
   - `exe` → your `strata-split-3.exe`
   - `args[1]` → your split GGUF pack
   - `--mtp` → the MTP draft head (ships in the model repo's `mtp/` directory)
   - `tokenizer`, `vision.*` → your tokenizer / vision engine / mmproj
   - `lib_dirs` → your CUDA `bin` dir

6. **Launch:**
   ```
   scripts\launch-live.bat
   ```
   Watch the log path set in the config until the engine reports ready, then hit `http://<host>:8080`. The model id is `qwen3.8-flash-next-split3`:
   ```
   curl http://<host>:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-flash-next-split3\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32}"
   ```

## The modifications at a glance

The mod stands on three legs — none of them a fork of Strata's engine code:

1. **Model** — Volta has no NVFP4 or FP8 hardware (NVFP4 needs Blackwell), so the pack must be requantized before it can run here. `scripts/split_pack.py` converts the Strata NVFP4 pack to a split GGUF: MoE experts NVFP4→**Q4_0**, dense BF16→**Q8_0**, hyper-connection/indexer stay BF16, PLE FP8→Q8_0. Getting the NVFP4 blob decode exactly right (UE4M3 scales, nibble order) was the hard part. Details: [MODEL.md](MODEL.md).
2. **Engine** — upstream Strata's split engine, compiled for Volta: `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_BUILD_SPLIT=ON -DSTRATA_SPLIT_ARCH=70`. Details: [ENGINE.md](ENGINE.md).
3. **Serve-side code** — four small patches (17 insertions, 4 deletions across 3 files) that make stock Strata launch and monitor a split engine:

| File | Change | Why |
|---|---|---|
| `serve/server.py` | skip `vram_fit.fit_start()` when the exe is a split engine | `strata-split-3.exe` sizes its own pools and **rejects the `--kv-resident` flag**, so stock VRAM-fit breaks split-engine startup |
| `serve/server.py` | honor `vision.cuda_device` | pins the vision reader process to one card instead of inheriting the engine's whole GPU ring |
| `serve/telemetry.py` | enumerate NVML cards 1–7 when config names a single GPU | split engines pick all cards themselves; the monitor can then show total VRAM across the machine |
| `strata_cli/client.py` | telemetry `devices` falls back to `safe_devices` | CLI monitor showed no device rows when the raw field was null |

Full diffs with context: [MODIFICATIONS.md](MODIFICATIONS.md).

## Tuning the split engine (env vars)

All set by `scripts/launch-live.bat`; verified values for 3× 32GB Volta:

| Variable | Value | Effect |
|---|---|---|
| `STRATA_SM70_TABLE` | `1` | use the sm_70 (Volta) weight-table path |
| `FNX_PF_ROWS` | `4096` | prefetch rows per pass |
| `FNX_PF_PASS` | `***` | prefetch passes |
| `FNX_PF_A2A_ROWS` | `1024` | all-to-all prefetch rows |
| `FNX_PF_PEER` | `1` | peer-aware prefetch |
| `FNX_PEER` | `1` | GPU-to-GPU peer transfers |
| `FNX_ARENA` | `0` | disable the VRAM arena |
| `FNX_EXPERT_RAM_GIB` | `off` | disable expert RAM pinning |
| `CUDA_VISIBLE_DEVICES` | `2,1,0` | card order for the engine ring; `vision.cuda_device: 0` then pins the vision reader to the ring's first entry (physical card 2, the PG500-216 here) — pick whichever card should host vision |

## Caveats

- **The endpoint is unauthenticated.** The live config binds `0.0.0.0` with no API key — fine behind a trusted LAN, dangerous anywhere else. Put it behind a reverse proxy or add auth before exposing it.
- **Volta on modern drivers:** NVIDIA dropped Volta from the 615+ driver branch. Stay on the 580/582-era branch (data-center 582.78 binds Tesla cards cleanly; on mixed GeForce+Tesla boxes install the data-center package first, then GeForce on top).
- **Suspend/resume wedges Tesla cards** (GSP-style wedge, `nvidia-smi` → "Unknown Error"). Disable OS sleep on the serving box; the live config sets `"prevent_sleep": true` as a second line of defense.
- **The split engine source** in our build was a PR-branch working tree, not the v0.2.14 tagged tree. Build flags in ENGINE.md are exact; for the source, use upstream's split-engine branch current with your needs.
- Throughput numbers depend on your cards and pack — measure with the upstream bench harness (`bench/`); ours are specific to 3× 32GB Volta + this pack.

## Sources & credit

None of this exists without other people's work. What we built on top of:

| Project | What it gave us |
|---|---|
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | the entire base: engine, all-cards split engine, serve stack, `tools/iq_pack.py` / `tools/nvfp4_local_pack.py`, vision engine — this repo is patches + config around it, MIT |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | the model itself (BF16 origin) |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Qwen's FP8 PLE table (our `ple-fp8.gguf` is it, E4M3) |
| [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) | the NVFP4-GPTQ Strata pack our converter consumes — downloaded as published (`pack/`, dense/PLE/embedding GGUFs) |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF container + Q4_0/Q8_0 block formats that `split_pack.py` targets; the `gguf` python writer/reader module |
| [johnfink8/sfnx](https://github.com/johnfink8/sfnx) | the sibling all-cards engine we also serve through Strata — the telemetry patch was written for this "sfnx-style" setup |
| NVIDIA | CUDA Toolkit 12.9; the NVFP4 formats (E2M1 nibbles, UE4M3 scales) and E4M3 FP8 that the converter decodes; ModelOpt quantization lineage behind NVIDIA's FP4-Local release |
| [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | the UD-Q4_K_XL pack this box ran before the split3 setup (tokenizer path in some configs still points at it) |

The upstream NVFP4-Local conversion additionally pins `torch` (BSD-3), `transformers` (Apache-2.0) and `safetensors` (Apache-2.0) — permissive licenses, nothing of theirs is shipped here.

## License

Same as upstream Strata — MIT, © Niko1221 and the Strata contributors. See [LICENSE](LICENSE).
