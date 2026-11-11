# Deployment — running MiniMax-H3 on 8 GB of VRAM

**[English](DEPLOY.md) | [简体中文](DEPLOY.zh-CN.md)**

> License: [Non-Commercial Personal-Study License v1.0](LICENSE) — free for personal study and research, **commercial use not permitted**.

This guide has two tracks. **Read the difference before you start:**

| Track | Goal | Guarantee |
|---|---|---|
| **A. Minimum viable** | Get the workflow to produce a first video on your 8 GB card | **Follow this and it runs.** No extra components needed |
| **B. Acceleration (optional)** | Make it run faster | **Not guaranteed.** It works on the author's machine; on yours it may not install, may not help, or may error out |

**Do A first, then consider B.** Track A leaves you with a system that produces video. Track B is a bonus — if it does not install, nothing is broken.

> **About the numbers**: the timings in the README were measured with **track B acceleration enabled** on the author's machine. If you only do track A, you will be slower than the table. **Track A promises "it runs", not "it runs in 326 s".** How much slower depends on your card and your attention backend; this document does not estimate it.

If you only want the measurements, go back to the [README](README.md).

---

## 0. Requirements

| Item | Requirement | Note |
|---|---|---|
| GPU | NVIDIA, **≥ 8 GB VRAM**, Ampere or newer | RTX 3050 8 GB reaches 0.7 MP. Below 8 GB is untested |
| System RAM | **≥ 32 GB** | A harder constraint than VRAM. The 19.5 GiB DiT and the 14.6 GiB text encoder stay resident together; on 32 GB only 1–2 GB stays free |
| Disk | **≥ 70 GB free** | 39.8 GB of weights + ~15 GB for ComfyUI and the Python env + output |
| Driver | One that supports cu130 | 621.14 here |
| OS | Windows 10/11 or Linux | Measured on Windows 11 22631 |

> **What insufficient RAM looks like**: not an error. Sampling degrades from 456 s to 1018 s (observed on the 6th consecutive 0.7 MP run). The log goes silent for several minutes, GPU utilisation drops to single digits, then it recovers on its own. If you see this, check memory before suspecting the model.

---

## 1. System level: pagefile (mandatory on Windows)

32 GB of physical RAM cannot hold 34 GB of resident weights, so the pagefile is load bearing. **Windows' automatic management is not enough** — set a fixed size manually.

**GUI**: `System Properties → Advanced → Performance Settings → Advanced → Virtual Memory → Change`
- Uncheck "Automatically manage paging file size for all drives"
- Pick a **non-system partition** (avoid competing with the OS for IO), choose "Custom size"
- Initial size = Maximum size = **65536 MB (64 GB)**
- Reboot for it to take effect

**Command line** (admin PowerShell — this changes a machine-wide setting):

```powershell
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false}
}
# assuming D: is the target
$pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='D:\\\\pagefile.sys'"
if ($pf) {
    Set-CimInstance -InputObject $pf -Property @{InitialSize=65536; MaximumSize=65536}
} else {
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='D:\\\\pagefile.sys'; InitialSize=65536; MaximumSize=65536}
}
```

After rebooting, verify with `Get-CimInstance Win32_PageFileUsage` — `AllocatedBaseSize` should read 65536.

> Keep the pagefile on an SSD. On a spinning disk the paging slowdown goes from "4 minutes late" to "40 minutes late".

---

## 2. ComfyUI and PyTorch

**Version**: ComfyUI **≥ 0.30.0** (native H3 support landed there); **0.44.0** was used here.

**PyTorch must be a cu130 build.** This is not a preference — the quantisation format you pick is tied to the CUDA version (see section 4).

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# expected: 2.11.0+cu130 (8, 6)
```

`(8, 6)` is sm86. **Remember that value** — every later decision about fp8 and triton follows from it.

---

## 3. Custom nodes

### 3.1 Required: ComfyUI-KJNodes

```bash
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
```

It supplies two nodes that are the **reason** this fits in 8 GB at all. They are not optional:

| Node | Role |
|---|---|
| `MiniMaxLowVRAMAttention` | Chunked attention; avoids materialising the full attention matrix at once |
| `MiniMaxChunkFeedForward` | Chunked FFN; cuts peak activation memory |

**Remove either one and sampling OOMs.** Not "gets slower" — it does not run.

### 3.2 Not needed

Every other node in the workflow ships with ComfyUI 0.44 core: `ResolutionSelector`, `ComfyMathExpression`, `CreateVideo`, `SaveVideo`.

You do **not** need cg-use-everywhere, rgthree-comfy, or VideoHelperSuite. The workflow in this repo is a rebuilt straight-line graph — no Set/Get bus, no Fast Groups Bypasser.

At this point track A is complete. Jump to section 4.

---

## 4. Models: five files, 39.8 GB

All from Comfy-Org's official repack: **[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)**

Inside mainland China, download from the **ModelScope mirror** (`modelscope.cn/models/Comfy-Org/MiniMax-H3`) or the gitcode mirror. Pulling 40 GB from the HuggingFace origin will usually stall.

| Role | File | Size | Destination |
|---|---|---|---|
| DiT (reference-to-video) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GiB | `models/diffusion_models/` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GiB | `models/text_encoders/` |
| Turbo LoRA (4-step) | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.8 GiB | `models/loras/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.6 GiB | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577 MiB | `models/vae/` |

### Why these particular quantisations

**`int8_convrot` matches cu130.** Comfy-Org's guidance: use `int8_convrot` when PyTorch is built for cu130, and fall back to `fp8_scaled` only on other CUDA versions. Choosing wrong does not error out — it just gets slower or produces bad output.

**`nvfp4_awq` does not require Blackwell.** This is the most common misconception. Comfy-Org states explicitly that the quantised encoder works without Blackwell, and it is the lightest of the three encoder options (bf16 / int8_convrot / nvfp4_awq). It is the correct pick for a 3050.

**Turbo LoRA origin**: [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), converted to ComfyUI format by Kijai. **Without it you are running 20 steps**, i.e. 3.3× the wall clock.

> No weights are redistributed in this repository. MiniMax-H3 weights fall under the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3).

---

## 5. Launching (the configuration track A guarantees)

On an 8 GB card running a 14B model, one wrong flag can mean OOM or black frames. **The minimum viable set is two flags:**

```bat
.venv\Scripts\python.exe main.py ^
  --disable-cuda-graphs ^
  --preview-method none
```

| Flag | Effect | Category |
|---|---|---|
| `--disable-cuda-graphs` | Frequent paging while running a large model on 8 GB makes CUDA graph capture prone to OOM | **Required for stability** |
| `--preview-method none` | Disables sampling previews. A preview costs an extra VAE decode, and on 8 GB that alone can hit the ceiling | **Required for VRAM** |

Adding `--auto-launch` opens the browser on startup. Convenience only, no effect on results.

### Flags deliberately left out

| Flag | Why not |
|---|---|
| `--highvram` / `--gpu-only` | Does not fit in 8 GB; guaranteed failure |
| `--fp8_*` | The 3050 is **sm8.6 with no fp8 hardware**, so fp8 paths are slower, not faster |
| `--fast` | Requires triton, not installed |
| `--reserve-vram` | Do not reserve VRAM on 8 GB — let DynamicVRAM use the whole dedicated pool |

That last row is a common mistake: people reserve 1–2 GB out of habit from 24 GB cards, which on 8 GB means voluntarily giving up 25% of usable VRAM.

---

## 6. Track B: acceleration (optional, no guarantee)

**This section is not a prerequisite for reproducing anything.** The author's machine had it installed and enabled, which is why the published timings include it — but whether it installs on your machine and whether it helps **depends on your card and CUDA environment. Nothing here is promised.**

```bash
.venv\Scripts\pip install comfy-kitchen
```

Then add one flag to your launch command:

```bat
  --use-ck-attention
```

That is comfy-kitchen's INT8 attention kernel. The author ran **0.2.41**; upstream documentation claims 3.5–4.2× on sm86, and it is mutually exclusive with native PyTorch attention.

### Three honest caveats

1. **Installing it does not mean it takes effect.** A version, CUDA, driver or architecture mismatch can make it fall back silently to the native path with no visible sign. To check, remove `--use-ck-attention` and compare sampling time.
2. **Do not install it first.** Get one video out via track A, confirm the whole chain works, then come back. Otherwise a failure leaves you unable to tell whether the model, the nodes, or the acceleration kernel broke.
3. **If it feels wrong, just drop the flag.** Removing `--use-ck-attention` is enough; no need to uninstall. You can add it back any time.

### Do not install sage attention

Many 8 GB tutorials recommend `--use-sage-attention`. **With MiniMax-H3 it produces fully black video** — the Qwen encoder is incompatible with it. This machine deliberately does not have the library installed.

---

## 7. First-run verification

1. Drop any image into `ComfyUI/input/` and **name it `example.png`** (the repo ships `assets/example.png`; copy it there if you just want to see it run)
2. Load `workflow/minimax_h3_ref2v_8gb.json`
3. Confirm three parameters: `秒数` = 6, `megapixels` = 0.5, steps = 6
4. Press Run

**Cold start is 6–10 minutes** (34 GB of weights to load); warm start 5–8 minutes. The range is this wide precisely because acceleration and your specific card both move the number.

Verify the output: **960×544, 24 fps, 158 frames, 6.583 s**. If it differs, the parameters did not take effect.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OutOfMemoryError` during sampling | The two low-VRAM KJNodes are not wired in, or more than one reference image | Check the chain `UNETLoader → LoRA → MiniMaxLowVRAMAttention → MiniMaxChunkFeedForward → sampler` is intact |
| Output is **entirely black** | `--use-sage-attention` is enabled (or the package is installed) | Remove the flag; this machine has no such library |
| **Startup error** after adding `--use-ck-attention` | comfy-kitchen does not match your CUDA/driver | Remove the flag and return to track A — video generation is unaffected |
| Added `--use-ck-attention`, **no speed change** | The kernel fell back silently to the native path | Compare sampling time with the flag removed; if there is no gain, drop the package — functionality is unchanged |
| **Red nodes** after importing the workflow | KJNodes missing | Install KJNodes and restart |
| Sampling jumps from 456 s to 1018 s | **Host RAM exhausted, paging** | Restart ComfyUI to release fragmentation; confirm the 64 GB pagefile; restart every 4–5 runs at 0.7 MP |
| Result returns in 0.x s | Execution cache hit (identical seed and parameters) | Change the seed, or set the seed control to `increment` |
| First run after changing resolution is much slower | Model re-staging, ~50 s | Expected; the second run at the same resolution is back to normal |
| Output dimensions differ from what you set | You are reading the stale widgets on `MiniMaxH3ReferenceToVideo` | Effective size comes from `ResolutionSelector`; ignore the `1344×768` shown on the node |
| Video is longer than the seconds you entered | Frame count snaps to a `17k+5` grid, rounding **up** | 6 s → 158 frames → 6.583 s. To shorten, enter 5.875 (141 frames) |

---

## 9. Going faster or higher

- **Acceleration**: see section 6. It is the only speed-up that does not require new hardware, and it is not guaranteed.
- **Higher preset**: 0.7 MP is this card's measured boundary. Upstream's stated ceiling is **0.98 MP** (the 1.0 MP preset resolves to 1376×768, exceeding H3-Base's pixel-area limit).
- **Different card**: 8 GB → 16 GB is a **threshold**, not a slope. See `Scaling beyond this card` in the README — those figures are bandwidth-ratio estimates, not measurements. **Run `tools/monitor.py` on your own card** rather than trusting the table.
