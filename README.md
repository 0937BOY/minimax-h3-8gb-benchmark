# MiniMax-H3 on 8 GB VRAM — Resolution Benchmark

Measured end-to-end generation times for **MiniMax-H3 Ref2VA** (pruned INT8 + 4-step Turbo LoRA) on an **RTX 3050 8 GB** laptop-class GPU. Three resolution presets, 4–6 runs each, same prompt / same frame count / same step count.

The point: MiniMax-H3's official guidance only covers **480P** and **720P**. This machine cannot reach 720P. Everything in between is undocumented — these numbers fill that gap.

---

## TL;DR

| Preset | Resolution | Pixels | n | **Mean total** | Mean sampling | s/step | Range |
|---|---|---|---|---|---|---|---|
| **0.5** | 960 × 544 | 0.522 MP | 6 | **326.3 s** (5:26) | 287.0 s | 47.88 | 319.9 – 350.1 |
| **0.6** | 1056 × 608 | 0.642 MP | 5 | **418.9 s** (6:59) | 371.2 s | 61.97 | 412.9 – 432.8 |
| **0.7** | 1152 × 640 | 0.737 MP | 4 | **510.6 s** (8:31) | 456.8 s | 76.21 | 505.0 – 517.3 |

All runs: **158 frames / 24 fps / 6.583 s**, **6 sampling steps**, 1 reference image, batch size 1.

**Cost per megapixel degrades with resolution**: 625 / 652 / 693 s per MP. Going 0.5 → 0.7 costs 56.5% more time for 41.2% more pixels.

---

## Three findings worth reading

### 1. VRAM peak is independent of resolution

| Preset | Peak VRAM | Headroom |
|---|---|---|
| 0.5 | 7868 MiB | 324 MiB |
| 0.6 | 7865 MiB | 327 MiB |
| 0.7 | 7864 MiB | 328 MiB |

Essentially identical across all three. The peak occurs during **VAE decode**, not during sampling — so sampling resolution barely moves it. **0.7 MP did not OOM and was never close to it.**

This means on an 8 GB card the limit is *time*, not memory. If you were extrapolating VRAM from pixel count (as is common), note that it does not hold here.

### 2. Long consecutive runs degrade badly

The 6th consecutive 0.7 MP run took **1018 s of sampling** versus 456 s for the first four — **2.2× slower**. The log shows tqdm at `169.75 s/it` instead of `76.2 s/it`.

Observed symptoms:
- Log stalls completely for ~4 minutes after the final sampling step
- ComfyUI process CPU time does not advance at all (verified via `Get-Process .CPU` over a 6 s window)
- GPU sits at 3% / 29 W

Root cause is **host RAM exhaustion, not VRAM**: physical free memory was down to **2.13 GB** with the ComfyUI working set at **21.52 GB**, and the pagefile (64 GB, on a separate SSD) already carrying 3.8 GB.

**Recommendation: restart ComfyUI every 4–5 runs at 0.7 MP.** At 0.5 MP this did not appear across 6 consecutive runs.

### 3. Model staging on resolution change costs ~50 s

Changing `megapixels` triggers re-staging of `MiniMaxH3TEModel_` (14.9 GB) plus the main model (20 GB). Measured **48–51 s**. The same-resolution re-runs pay only ~0.3 s.

That is why the first run of each preset is excluded from the table above. Once staged, consecutive same-resolution runs are warm.

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3050, 8 GB, compute capability 8.6 (Ampere) |
| Driver | 621.14 |
| CPU / RAM | Intel i5-10400F / 32 GB DDR4 |
| OS | Windows 11 Pro for Workstations, 22631 |
| ComfyUI | 0.44.0 |
| Python / PyTorch | 3.13.3 / 2.11.0+cu130 |
| Pagefile | 64 GB, fixed, on a separate SATA SSD |

**Custom node pack required: [`ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes).**

It supplies `MiniMaxLowVRAMAttention` and `MiniMaxChunkFeedForward` — the two patches the model chain passes through. These are not optional on an 8 GB card; they are the reason it fits at all.

Everything else in the workflow (`ResolutionSelector`, `ComfyMathExpression`, `CreateVideo`, `SaveVideo`) ships with ComfyUI 0.44 core.

---

## Models

All weights come from **Comfy-Org's official repack**: [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) (mirrors: [ModelScope](https://www.modelscope.ai/models/Comfy-Org/MiniMax-H3), gitcode).

| Role | File | Size | Folder |
|---|---|---|---|
| DiT (reference-to-video) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GiB | `models/diffusion_models/` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GiB | `models/text_encoders/` |
| Turbo LoRA (4-step) | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.8 GiB | `models/loras/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.6 GiB | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577 MiB | `models/vae/` |

**Total ≈ 39.8 GB on disk.** No weights are redistributed in this repository — follow the links above.

Notes on the specific files:
- `int8_convrot` is Comfy-Org's INT8 + ConvRot quantisation of the pruned (rank-8 AdaLN) DiT. Comfy-Org's recommendation is to use `int8_convrot` when running PyTorch with cu130, and `fp8_scaled` only as a fallback. This machine uses cu130, hence this choice.
- `nvfp4_awq` for the text encoder is *not* limited to Blackwell GPUs — Comfy-Org states explicitly that this quantised encoder works without Blackwell. It is the lightest of the three encoder options (bf16 / int8_convrot / nvfp4_awq).
- The Turbo LoRA originates from [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), converted to ComfyUI format by Kijai.

---

## Reproducing

1. Install ComfyUI ≥ 0.30.0 (H3 native support landed there; 0.44.0 used here) with PyTorch built against **cu130**.
2. Place the five files above into the folders listed.
3. Load `workflow/minimax_h3_ref2v_8gb.json`.
4. Put your own image into `ComfyUI/input/` and name it `example.png` — the workflow has a single `LoadImage` node pointing at that filename. A placeholder ships in `assets/example.png`; copy it there if you just want to see the graph run.
5. Set the seed, press Run.

The workflow is deliberately a **minimal, ungrouped, single-reference build**: 23 nodes, 28 links, straight-line layout, no parameter bus, no group bypasser. It is the same compute graph that produced the numbers above, stripped of the author's panel layout. If you want six reference images, add five more `LoadImage` nodes and wire them to `ref_image_1` … `ref_image_5` on the `MiniMaxH3ReferenceToVideo` node — those inputs exist but are left unconnected here.

### What the workflow exposes

- **`秒数` (seconds)** — an integer. Do not expect exactly this number. Frames snap to the `17k+5` grid required by the H3 video VAE, and the mapping rounds *up*: input `6` → `round(6×24)=144` → `144 % 17 = 8` → padded to **158 frames** = **6.583 s**. Any input in the 6.0–6.583 s range produces the same 158 frames. The next step down is **141 frames** (5.875 s), saving ~11% of sampling time.
- **`megapixels` (ResolutionSelector)** — the only resolution control. Internally `total = mp × 1024²`, then sides are rounded to multiples of 32. Hence the actual pixel counts run 4–7% above the nominal value: 0.5 → 0.522 MP, 0.6 → 0.642 MP, 0.7 → 0.737 MP.
- **Steps** — 6 here, paired with the 4-step Turbo LoRA. Upstream defaults to 20 steps without the LoRA.

> **Do not read dimensions off the `MiniMaxH3ReferenceToVideo` node.** Its `width`/`height`/`length` widgets show stale values (`1344 × 768 / 124`) that are overridden by `GetNode` connections. The effective values come from the selector and the expression node.

### On the 0.98 MP ceiling

Upstream guidance notes that the **1.0 MP preset resolves to 1376 × 768, which exceeds H3-Base's pixel-area limit**, and recommends **0.98 MP** as the practical maximum. This benchmark stops at 0.7 MP, comfortably inside that ceiling — 0.7 MP was chosen as the last preset this 8 GB card could run without hitting the time ceiling, not because of a memory limit.

---

## How the numbers were measured

- **Total time** — ComfyUI's own `Prompt executed in X seconds` line. Not wall-clock, not inferred.
- **Sampling time** — the cumulative time in the final tqdm line, e.g. `6/6 [07:36<00:00, 76.16s/it]` → 456 s.
- **VRAM / temp / power** — `nvidia-smi` polled every 1.5 s. Peak is the max observed, not the max reported by PyTorch.
- **Run identification** — every run was separated by its `Prompt executed` anchor. Note `got prompt` is *not* a valid anchor: it fires when the HTTP request arrives, which can be long before execution starts when several jobs are queued.
- **Staging detection** — a run is classified as a "resolution switch" only if `MiniMaxH3TEModel_` is re-staged during it. `MiniMaxH3`, `MiniMaxH3VideoVAE` and `MiniMaxH3AudioVAE` also print `prepared for dynamic VRAM loading` on *every* run and must not be used as the signal. Getting this wrong marks every run as a switch.

Excluded from the results: 1 cache-hit run (0.02 s, duplicate seed), 2 resolution-switch runs, 1 degraded run (the 1018 s outlier described above).

The measurement script is in [`tools/monitor.py`](tools/monitor.py). Raw per-run data is in [`data/benchmark.csv`](data/benchmark.csv).

---

## Scaling beyond this card

The RTX 3050 8 GB is close to the **bottom edge** of what runs H3 at all. It is an entry-level Ampere part: 2560 CUDA cores, 8 GB of GDDR6 on a 128-bit bus, **224 GB/s**. That bandwidth, not the 8 GB capacity, is what sets the pace here — 625–693 seconds per megapixel is simply what this tier costs.

Moving to a 40- or 50-series card buys two independent things:

1. **Speed** — bandwidth and compute scale roughly together on this workload.
2. **Ceiling** — more VRAM means higher presets, more steps, and eventually dropping the low-VRAM patches (`MiniMaxLowVRAMAttention`, `MiniMaxChunkFeedForward`) that this 8 GB run depends on.

### Order-of-magnitude reference

> These are **bandwidth-ratio estimates, not measurements**. Real speedups come in below the bandwidth multiple, because attention and sampler overhead do not scale with it. Treat the VRAM column as the reliable part.

| GPU | VRAM | Bandwidth | vs 3050 | Practical preset range |
|---|---|---|---|---|
| RTX 3050 8 GB *(this run)* | 8 GB | 224 GB/s | 1.0× | 0.5 – 0.7 MP, 6 steps |
| RTX 4060 Ti 16 GB | 16 GB | 288 GB/s | 1.3× | up to ~0.98 MP |
| RTX 5070 | 12 GB | 672 GB/s | 3.0× | up to ~0.98 MP |
| RTX 4070 Ti Super | 16 GB | 672 GB/s | 3.0× | ~0.98 MP, higher step count |
| RTX 5070 Ti | 16 GB | 896 GB/s | 4.0× | ~0.98 MP, 20-step quality mode |
| RTX 5090 | 32 GB | 1792 GB/s | 8.0× | full 1344×768, no low-VRAM patches |

The 8 GB → 16 GB jump is a **threshold**, not a slope: it is the difference between 0.7 MP at 6 steps and 0.98 MP at 20 steps. Below 16 GB you are mostly buying linear speed; at 16 GB and above you unlock presets this card cannot address at all.

### Benchmark your own card

The script in `tools/monitor.py` is hardware-agnostic. To produce a row comparable to the table at the top:

1. Use the same workflow, same prompt, same 158 frames, same step count.
2. Sweep `megapixels` across the presets your card can actually finish. **Skip the first run of each preset** — it pays a ~50 s model-staging cost that same-preset runs do not.
3. Run at least 3–4 per preset. Discard cache hits (they finish in under a second) and any run that re-staged `MiniMaxH3TEModel`.
4. Record total time from `Prompt executed in X seconds`, sampling time from the final tqdm line, VRAM peak from `nvidia-smi`.

Two ratios are what make results comparable across cards:

- **seconds per megapixel** — 625 / 652 / 693 here. Compare yours against this.
- **whether VRAM peak moves with resolution** — here it does not (7864–7868 MiB across all three presets), which is unusual and worth confirming on your hardware.

If you run this on other hardware, a PR adding your row to `data/benchmark.csv` is welcome.

---

## Caveats

- **n is 4–6 per preset, single machine, single session.** No confidence intervals. Run-to-run spread was ±1–4% within a preset, except the outlier noted above (which was 120%).
- Sampling speed drifts even with identical parameters: 46.8–52.1 s/step at 0.5 MP across six runs, an 11% spread. Treat the means as ±10 s, not exact.
- One 0.5 MP run (350.1 s) is a mild outlier spanning a monitoring-script restart; excluding it lowers the 0.5 MP mean to 321.6 s.
- Temperatures were 66–68 °C throughout; power sat at the ~130 W board limit. No thermal throttling was involved.
- This is a consumer quantised setup. Absolute times will not transfer to other GPUs, but the *shape* — flat VRAM, degrading s/MP, staging cost — should.

---

## 中文摘要

在 RTX 3050 8 GB 上实测 MiniMax-H3 Ref2VA（pruned INT8 + 4 步 Turbo LoRA）三档分辨率的端到端出片耗时，每档 4–6 条，统一 158 帧 / 24fps / 6.583 秒 / 6 步采样：

| 档位 | 分辨率 | 平均总耗时 | 平均采样 | 秒/步 |
|---|---|---|---|---|
| 0.5 MP | 960×544 | 326.3 s | 287.0 s | 47.88 |
| 0.6 MP | 1056×608 | 418.9 s | 371.2 s | 61.97 |
| 0.7 MP | 1152×640 | 510.6 s | 456.8 s | 76.21 |

三个结论：**① 显存峰值与分辨率无关**（三档均为 7864–7868 MiB，峰值出现在 VAE 解码阶段），0.7 MP 不会 OOM，真正的瓶颈是时间；**② 0.7 MP 连跑第 6 条会严重退化**（采样 456 s → 1018 s），根因是主机内存耗尽触发换页，建议每 4–5 条重启 ComfyUI；**③ 改分辨率要额外付约 50 秒的模型重新 staging**，同分辨率连跑只需 0.3 秒。

模型全部来自 Comfy-Org 官方 repack（`Comfy-Org/MiniMax-H3`），Turbo LoRA 来自 `lightx2v/Minimax-h3-Turbo`。本仓库不包含任何权重文件。

**关于换卡**：3050 8GB 是能跑 H3 的下限档位，主要瓶颈是 224 GB/s 的显存带宽而非容量。换 40/50 系同时带来「更快」和「能开更高档」两件事，其中 **8GB → 16GB 是门槛级差异**（0.7MP/6 步 → 0.98MP/20 步），而同容量档内主要是线性提速。README 英文部分给了带宽倍数参考表和自测方法，但那部分是量级估算，不是实测——**建议按同样流程在自己的卡上跑一遍再对比**，两个关键比值是「每百万像素秒数」（本机 625/652/693）和「显存峰值是否随分辨率变化」（本机不变）。

---

## License

- Text, tables and measurement data: **CC BY 4.0**
- `tools/monitor.py` and the workflow JSON: **MIT**

Model weights are governed by the [MiniMax-H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3) and are not distributed here.
