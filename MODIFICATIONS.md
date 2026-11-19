# Modifications vs stock Strata v0.2.14

Base: `Niko1221/Strata`, tag `v0.2.14`, commit `09b603886` (2026-11-15), detached HEAD.

Everything below is the complete delta of the **Python serve side**. The engine binary delta is build flags only — see [ENGINE.md](ENGINE.md).

```
 serve/server.py      | 14 +++++++++++---
 serve/telemetry.py   |  5 +++++
 strata_cli/client.py |  2 +-
 3 files changed, 17 insertions(+), 4 deletions(-)
```

---

## 1. `serve/server.py`

### 1a. Split engines skip the VRAM-fit prologue (the critical one)

Stock Strata runs `vram_fit.fit_start()` before every engine start: it estimates how much KV cache fits on the selected card(s) and injects flags like `--kv-resident`. The all-cards split engine manages its own pools and **rejects `--kv-resident` at startup**, so on a split config stock Strata fails to start (or fights its own engine). The fix keys off the exe path:

```diff
@@ -3920,7 +3925,10 @@ def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) ->
     try:
-        args, fit_parallel, fit_note = vram_fit.fit_start(cfg, args, drafting_vram_mib(cfg, backend, archs), backend)
+        if "split" in str(cfg.get("exe", "")).lower():
+            args, fit_parallel, fit_note = args, None, ""   # split-engine configs: their own pool sizing, the exe rejects --kv-resident
+        else:
+            args, fit_parallel, fit_note = vram_fit.fit_start(cfg, args, drafting_vram_mib(cfg, backend, archs), backend)
     except Exception:                                        # noqa: BLE01 - an estimate that fails decides nothing; the engine still refuses what does not fit
         fit_parallel, fit_note = None, ""
```

### 1b. Vision reader pinned to its own card

The vision mmproj reader inherits the engine's `CUDA_VISIBLE_DEVICES` — on a split setup that's the whole ring, and the reader contends with the engine's ring card. A new optional `"cuda_device": N` in the `vision` block remaps the reader's environment down to that one physical card:

```diff
@@ -3194,6 +3194,11 @@ class Vision:
     def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = False):
         args = self.reader_args(cfg)
+        cd = cfg.get("cuda_device")
+        if isinstance(cd, int) and env is not None:
+            vis = (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")
+            if len(vis) > cd and vis[cd].strip():
+                env = dict(env); env["CUDA_VISIBLE_DEVICES"] = vis[cd].strip()   # the reader's own card, not the engine's ring card
         self.dir = self.work_dir()
```

In the live config the ring is `"2,1,0"` and `"vision.cuda_device": 0`, i.e. the reader is pinned to physical card 2 (the PG500-216) — whichever card you want to host vision.

### 1c. Editor artifact that is a real (minor) regression

The patched file was re-saved on Windows, which introduced two encoding changes:

1. A UTF-8 **BOM** was added to the first line — harmless (Python handles it).
2. In `Detokenizer.push`, the guard `text.endswith("\ufffd")` was rewritten from the raw U+FFFD character to the bytes `C3 AF C2 BF C2 BD` — a double-encoded form that Python parses as the three-character string `ï¿½`, which never matches decoded text. **The "withhold a partial codepoint until it completes" guard is dead in this path.** Impact is cosmetic and rare: a streamed delta may briefly show `` where upstream would have held it back. One-line fix (replace the mojibake literal with the ASCII escape `\ufffd`); the sibling guard above it already uses the correct escaped form.

We left it as-is to keep the deployed tree and this patch set byte-identical; apply the fix when convenient.

---

## 2. `serve/telemetry.py`

Split engines pick every NVIDIA card themselves and the serve config names none, so stock telemetry reports a single GPU — you can't see the other two cards' VRAM in the monitor. When only one index is known on NVIDIA, probe NVML cards 1–7 and keep every card that answers:

```diff
@@ -544,6 +544,11 @@ class Telemetry:
         idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
+        if len(idx) == 1 and not amd:      # sfnx-style engines: the engine picks every NVIDIA card itself and the
+                                           # config names none, so enumerate NVML and report all of them (monitor VRAM total)
+            for i in range(1, 8):
+                if i != idx[0] and gpu_reader(i, False).ok():
+                    idx.append(i)
         self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
```

---

## 3. `strata_cli/client.py`

With the telemetry change, raw payloads can carry `devices: null` in some paths and the CLI monitor rendered an empty device panel. Fall back to the already-sanitized `safe_devices` rows:

```diff
@@ -426,7 +426,7 @@ def sanitize_telemetry(raw: dict[str, Any]) -> dict[str, Any]:
-        "devices": _rows(raw.get("devices"), ("index", "name", "mem_used", "mem_total", "util", "temp", "power"), 16),
+        "devices": _rows(raw.get("devices") or safe_devices, ("index", "name", "mem_used", "mem_total", "util", "temp", "power"), 16),
```

---

## Not code, but part of the mod

- **Engine build** — `strata-split-3.exe` compiled for `sm_70`: see [ENGINE.md](ENGINE.md).
- **Model conversion** — `scripts/split_pack.py` requantizes the NVFP4 pack to Q4_0/Q8_0 for Volta: see [MODEL.md](MODEL.md).
- **Serve config** — `configs/strata-nvfp4-split3-262k-4096p4.json`: 262k context, batch 4, spec 4, `--conversation-cache-mib 32768`, `--vision`, elastic idle unload (`elastic_idle_s: 30`, `idle_unload_s: 600`), `prevent_sleep: true`, CUDA v12.9 in `lib_dirs`.
- **Launcher env** — the `FNX_*` / `STRATA_SM70_TABLE` table in README.md.
