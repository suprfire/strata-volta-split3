# Building the Volta split engine (`strata-split-3.exe`)

The split engine is upstream Strata's all-cards engine. For Volta you compile it targeting **`sm_70`**. This is the exact configuration in production here (Windows).

## Prerequisites (Windows)

| Component | Version used |
|---|---|
| Visual Studio 2022 Build Tools | any recent (provides `vcvars64.bat`) |
| CUDA Toolkit | 12.9 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`) |
| Generator | Ninja |
| Python | 3.12 (only needed for the serve side) |

Linux is the same CMake flags with your toolkit path; the `vcvars64.bat` line is the only Windows-ism.

## Build script

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
cd /d D:\path\to\strata-engine-src
cmake -S . -B build-split70 -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" ^
  -DCMAKE_CUDA_ARCHITECTURES=70 ^
  -DSTRATA_EXPERIMENTAL_SM60=ON ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DSTRATA_BUILD_SPLIT=ON ^
  -DSTRATA_SPLIT_CARDS="2;3" ^
  -DSTRATA_SPLIT_ARCH=70 ^
  -DSTRATA_BUILD_TESTS=OFF ^
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build-split70 --target strata-split-2 strata-split-3
```

Output: `build-split70\src\split\strata-split-3.exe` (and `strata-split-2.exe`).

## The flags that matter for Volta

| Flag | Meaning |
|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES=70` | compile CUDA kernels for Volta only (sm_70) |
| `-DSTRATA_BUILD_SPLIT=ON` | build the all-cards split engine targets |
| `-DSTRATA_SPLIT_ARCH=70` | split-engine code paths tuned for the Volta generation |
| `-DSTRATA_SPLIT_CARDS="2;3"` | card-class tiers the pack is split across (2 + 3 = the tier layout our three 32GB cards report) |
| `-DSTRATA_EXPERIMENTAL_SM60=ON` | enables the older-arch fallback tables alongside sm_70 |
| `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` | link against shared CUDA runtime — match this with the `lib_dirs` entry in the serve config pointing at the CUDA `bin` directory |

## Source tree caveat

Our production binary was built from a PR-branch working tree of Strata carrying the split-engine work, not from the `v0.2.14` tag (the working copy on our box is no longer under git, so the exact PR ref is lost). The CMake interface above is stable; build from the latest upstream tree that contains `src/split` and the `STRATA_BUILD_SPLIT` options.

## Runtime pairing

The serve config must set, for the split engine to behave on Volta:

- `STRATA_SM70_TABLE=1` — weight-table path for sm_70
- `FNX_ARENA=0`, `FNX_EXPERT_RAM_GIB=off` — the engine's pool managers that don't fit the 32GB Volta budget alongside a 262k cache
- `FNX_PEER=1`, `FNX_PF_PEER=1` — peer-to-peer transfers/prefetch between the cards (all three here are on the same NUMA root; verify P2P with `nvidia-smi topo -m` before enabling on other layouts)
- `FNX_PF_ROWS=4096`, `FNX_PF_PASS=*** `FNX_PF_A2A_ROWS=1024` — prefetch depth; tuned for 3×32GB, halve rows if VRAM is tighter

See `scripts/launch-live.bat` for how these are set at launch.
