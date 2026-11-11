# 部署方案 — 8 GB 显卡跑 MiniMax-H3

**[English](DEPLOY.md) | 简体中文**

> 许可：[非商业个人学习许可 v1.0](LICENSE) —— 可用于个人学习与研究，**不允许商用**。

这份文档分两条路线，**请先看清区别再动手**：

| 路线 | 目标 | 承诺 |
|---|---|---|
| **A. 最小可跑** | 让工作流在你的 8 GB 卡上跑出第一个视频 | **按此做完就能跑起来**，不需要额外组件 |
| **B. 加速（可选）** | 让它跑得更快 | **不保证**。作者的机器上有效，你的机器上可能无效、可能没装上、也可能报错 |

**先走 A，跑通了再考虑 B。** A 路线走完，你得到的是一个能出片的系统；B 路线是锦上添花，装不上不影响出片。

> 性能数字的说明：README 里那张表的耗时，是在**作者机器开启了加速（B 路线）**的条件下测得的。你如果只走 A，速度会慢于表里的数字。**A 路线的目标是「能跑」，不是「跑出 326 秒」**。具体慢多少取决于你的卡和 attention 实现，本文不做估计。

如果你只想看数据，回到 [README](README.zh-CN.md)。

---

## 0. 前提与门槛

| 项 | 要求 | 说明 |
|---|---|---|
| 显卡 | NVIDIA，**≥ 8 GB 显存**，Ampere 及以上 | 实测 RTX 3050 8 GB 可跑 0.7 MP。低于 8 GB 未验证 |
| 系统内存 | **≥ 32 GB** | 这一条比显存更硬。主模型 19.5 GiB + 文本编码器 14.6 GiB 会同时驻留，32 GB 物理内存只剩 1~2 GB 空闲 |
| 硬盘 | **≥ 70 GB 可用** | 权重 39.8 GB + ComfyUI 本体与 Python 环境约 15 GB + 输出 |
| 驱动 | 支持 cu130 的版本 | 本机 621.14 |
| 系统 | Windows 10/11 或 Linux | 本机为 Windows 11 22631 |

> **内存不足会怎样**：不会报错，而是让采样从 456 s 退化到 1018 s（实测第 6 条 0.7 MP）。表现是日志静止几分钟、GPU 利用率掉到个位数，然后自己恢复。看到这个现象，先查内存，别怀疑模型。

---

## 1. 系统层：页面文件（Windows 必做）

32 GB 物理内存装不下 34 GB 的常驻模型，必须靠页面文件兜底。**默认的「系统自动管理」不够用**，要手动设成固定大小。

**图形界面**：`系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存 → 更改`
- 取消勾选「自动管理所有驱动器的分页文件大小」
- 选一个**非系统盘**的分区（避免和系统争 IO），设「自定义大小」
- 初始大小 = 最大值 = **65536 MB（64 GB）**
- 确定后**重启生效**

**命令行（管理员 PowerShell，慎用，会改全局设置）**：

```powershell
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false}
}
# 假设要设在 D 盘
$pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='D:\\\\pagefile.sys'"
if ($pf) {
    Set-CimInstance -InputObject $pf -Property @{InitialSize=65536; MaximumSize=65536}
} else {
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='D:\\\\pagefile.sys'; InitialSize=65536; MaximumSize=65536}
}
```

重启后确认：`Get-CimInstance Win32_PageFileUsage` 应显示 `AllocatedBaseSize` 为 65536。

> 页面文件放在 SSD 上。放机械盘会让换页退化从「慢 4 分钟」变成「慢 40 分钟」。

---

## 2. ComfyUI 与 PyTorch

**版本要求**：ComfyUI **≥ 0.30.0**（H3 原生支持从这版开始），本机用 **0.44.0**。

**PyTorch 必须用 cu130 构建**。这不是偏好问题——量化格式的选择直接绑定 CUDA 版本（见第 4 节）。

Windows 推荐用官方 portable 包，或手动建 venv：

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# 期望：2.11.0+cu130 (8, 6)
```

`(8, 6)` 就是 sm86。**记住这个值**——后面所有「能不能开 fp8」「要不要装 triton」的判断都由它决定。

---

## 3. 自定义节点

### 3.1 必装：ComfyUI-KJNodes

```bash
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
```

它提供两个节点，是 8 GB 能**跑起来**的**根本原因**，不是可选项：

| 节点 | 作用 |
|---|---|
| `MiniMaxLowVRAMAttention` | 分块计算注意力，避免一次性物化完整 attention 矩阵 |
| `MiniMaxChunkFeedForward` | 分块跑 FFN，把峰值激活值压下来 |

**去掉这两个节点，采样阶段必然 OOM。** 这不是「慢一点」的问题，是跑不出来。

### 3.2 不需要装

工作流里其余节点全部是 ComfyUI 0.44 核心自带：`ResolutionSelector`、`ComfyMathExpression`、`CreateVideo`、`SaveVideo`。

**不需要** cg-use-everywhere、rgthree-comfy、VideoHelperSuite。仓库里的工作流是重新搭的直线版，没有 Set/Get 总线，也没有 Fast Groups Bypasser。

到这里 A 路线（最小可跑）的依赖就齐了，可以直接跳到第 4 节。

---

## 4. 模型：5 个文件，39.8 GB

全部来自 Comfy-Org 的官方 repack：**[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)**

国内下载建议走 **ModelScope 镜像**（`modelscope.cn/models/Comfy-Org/MiniMax-H3`），或 gitcode 镜像。40 GB 走 HuggingFace 原站大概率断流。

| 角色 | 文件 | 体积 | 放到 |
|---|---|---|---|
| DiT（参考转视频） | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GiB | `models/diffusion_models/` |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GiB | `models/text_encoders/` |
| Turbo LoRA（4 步） | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.8 GiB | `models/loras/` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.6 GiB | `models/vae/` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577 MiB | `models/vae/` |

### 为什么选这几个量化版

**`int8_convrot` 对应 cu130。** Comfy-Org 的建议是：PyTorch 用 cu130 就选 `int8_convrot`，只有在别的 CUDA 版本下才退到 `fp8_scaled`。选错不会报错，只会变慢或出异常结果。

**`nvfp4_awq` 不要求 Blackwell。** 这是最常见的误解。Comfy-Org 明确说明这个量化编码器在没有 Blackwell 的卡上也能用，它是三档编码器（bf16 / int8_convrot / nvfp4_awq）里最轻的。3050 用它是正确的。

**Turbo LoRA 出处**：[`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo)，由 Kijai 转换成 ComfyUI 格式。**没有它就要跑 20 步**，耗时直接乘以 3.3。

> 本仓库不包含任何权重文件。MiniMax-H3 权重适用 [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3)。

---

## 5. 启动（A 路线的保证配置）

8 GB 卡跑 14B 模型，启动参数错一个就可能 OOM 或黑图。**最小可跑只需要这两个参数**：

```bat
.venv\Scripts\python.exe main.py ^
  --disable-cuda-graphs ^
  --preview-method none
```

| 参数 | 作用 | 属于 |
|---|---|---|
| `--disable-cuda-graphs` | 8 GB 跑大模型要频繁换页，CUDA graph 捕获时容易 OOM | **运行必需**（稳定性） |
| `--preview-method none` | 关掉采样过程预览。预览要额外跑一次 VAE 解码，在 8 GB 上这一下就可能顶到上限 | **运行必需**（省显存） |

加 `--auto-launch` 可以让它启动后自动打开浏览器，纯方便，不影响结果。

### 刻意不加的参数（重要）

| 参数 | 为什么不加 |
|---|---|
| `--highvram` / `--gpu-only` | 8 GB 装不下，必炸 |
| `--fp8_*` | 3050 是 **sm8.6，没有 fp8 硬件单元**，走 fp8 反而更慢 |
| `--fast` | 依赖 triton，未安装 |
| `--reserve-vram` 预留显存 | 8 GB 不预留，让 DynamicVRAM 吃满专用显存 |

最后一行是常见错误：很多人按 24 GB 卡的习惯预留 1~2 GB，在 8 GB 上等于主动放弃 25% 的可用显存。

---

## 6. B 路线：加速（可选，不保证有效）

**这一段不是复现前提。** 作者的机器装了并开启了它，所以本文的性能数字里包含了这部分加速；但它在你的机器上能不能装、有没有效果，**取决于你的卡和 CUDA 环境，本文不做任何承诺**。

```bash
.venv\Scripts\pip install comfy-kitchen
```

装好之后在启动参数里加一行：

```bat
  --use-ck-attention
```

这是 comfy-kitchen 提供的 INT8 注意力内核。作者机器上的版本是 **0.2.41**，官方文档称在 sm86 上有 3.5~4.2 倍加速，且与 PyTorch 原生 attention 互斥。

### 三条实话

1. **装了不一定生效。** 版本、CUDA、驱动、显卡架构任何一个不匹配，它可能静默回退到原生实现，你完全察觉不到。想确认就把 `--use-ck-attention` 去掉跑一次，比对采样耗时。
2. **不建议先装。** 先用 A 路线跑出第一个视频，确认整条链路没问题，再回头装加速。否则一旦出问题，你分不清是模型、节点还是加速内核的问题。
3. **装着不舒服就去掉。** 去掉 `--use-ck-attention` 参数即可，不需要卸载包，随时可以加回来。

### 特别提醒：不要装 sage attention

网上很多 8 GB 教程推荐 `--use-sage-attention`。**在 MiniMax-H3 上它会出全黑视频**——Qwen 系编码器与它不兼容。本机刻意没装这个库。

---

## 7. 首次运行验证

1. 把任意图片放进 `ComfyUI/input/`，**命名为 `example.png`**（仓库自带 `assets/example.png` 占位图，可以直接复制过去）
2. 界面里加载 `workflow/minimax_h3_ref2v_8gb.json`
3. 确认三个参数：`秒数` = 6、`megapixels` = 0.5、步数 = 6
4. 点运行

**冷启动预期 6~10 分钟**（首次要加载 34 GB 权重），热启动 5~8 分钟。区间留这么宽，就是因为有没有加速、你的卡具体是哪张，都会影响这个数。

出片后核对规格：**960×544、24 fps、158 帧、6.583 秒**。对不上说明参数没生效。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `OutOfMemoryError` 在采样阶段 | KJNodes 的两个低显存节点没接上，或参考图超过 1 张 | 检查链路 `UNETLoader → LoRA → MiniMaxLowVRAMAttention → MiniMaxChunkFeedForward → 采样` 是否完整 |
| 能出片但是**全黑** | 用了 `--use-sage-attention`（或装了这个包） | 去掉这个参数；本机未装该库 |
| 加了 `--use-ck-attention` 后**启动报错** | comfy-kitchen 与你的 CUDA/驱动不匹配 | 去掉这个参数，回到 A 路线即可正常跑，不影响出片 |
| 加了 `--use-ck-attention` 但**速度没变** | 内核静默回退到原生实现 | 去掉参数对比采样耗时；确认无收益就删掉包，不影响功能 |
| 导入工作流后**红节点** | 缺 KJNodes | 装 KJNodes 并重启 |
| 采样突然从 456 s 变 1018 s | **主机内存耗尽触发换页** | 重启 ComfyUI 释放碎片；确认页面文件 64 GB；0.7 MP 每跑 4~5 条重启一次 |
| 点运行后**秒出结果**（0.x 秒） | 命中执行缓存（种子与参数完全相同） | 换种子，或把种子的控制方式改成 `increment` |
| 改分辨率后第一条特别慢 | 模型重新 staging，约 50 s | 正常现象，同分辨率连跑第二条就恢复 |
| 画面尺寸和设定不符 | 看的是 `MiniMaxH3ReferenceToVideo` 节点上的残留显示值 | 实际尺寸由 `ResolutionSelector` 决定，忽略节点上显示的 `1344×768` |
| 视频时长比填的秒数长 | 帧数要对齐 `17k+5` 网格，只向上取整 | 填 6 秒 → 158 帧 → 6.583 秒。想缩短就填 5.875（141 帧） |

---

## 9. 想更快 / 更高

- **加速**：见第 6 节。这是唯一不换硬件就能提速的手段，但不保证有效。
- **换更高档位**：8 GB 的边界是本机测出来的 0.7 MP。上游给的上限是 **0.98 MP**（1.0 MP 预设会解析到 1376×768，超过 H3-Base 的像素面积限制）。
- **换卡**：8 GB → 16 GB 是**门槛级**变化，不是线性提速。详见 README 的 `Scaling beyond this card` 一节——那部分是带宽倍数的量级估算，不是实测，**建议用仓库里的 `tools/monitor.py` 在自己的卡上跑一遍**。
