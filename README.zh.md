<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/interlock/readme.png" width="480" alt="interlock">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/interlock/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/interlock/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
  <a href="https://mcp-tool-shop-org.github.io/interlock/"><img src="https://img.shields.io/badge/docs-landing%20page-blue" alt="Landing page"></a>
  <a href="https://pypi.org/project/mcptoolshop-interlock/"><img src="https://img.shields.io/pypi/v/mcptoolshop-interlock" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/interlock"><img src="https://codecov.io/gh/mcp-tool-shop-org/interlock/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/interlock"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/interlock/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/15467"><img src="https://www.bestpractices.dev/projects/15467/badge" alt="OpenSSF Best Practices"></a>
</p>

# interlock：本地 AI 设备路由器

> **版本 1.0.9：一种研究工具。** 从 1.0 开始，版本遵循语义版本控制：如果某个更改破坏了 CLI、收据格式或实验规范格式，则会发布一个全新的主要版本。互锁仅提供建议：它会建议一个模型、引擎和设备，并且它绝不会处理实际流量。其中的每个本地编号都曾在同一测试平台上进行测量（Intel Core Ultra 9 285K 及其 NPU 和 iGPU，以及一张 RTX 5090），而远程 `colab:<card>` 行则位于各自独立的证据池中，并具有完整的设备标识（`provider`、`card`、`vram_mib`、`driver_version`、`cuda_version`、`pool`）。将它的路由视为针对特定硬件池的测量证据，而不是通用的基准测试。 **1.0.6 版本中的修正：** 在 1.0.5 及更早版本中，Linux CUDA 生成的“首次生成 token 的时间”以及“高于 1 的并发吞吐量”这两个指标并不准确（每个经过计时测试的提示都来自 Ollama 的提示缓存，并发请求通过单个通道排队）；interlock 已经不再使用这些指标，并且版本 3 的数据行取代了它们。

铁路互锁系统可以防止列车被引导到冲突的轨道上。此工具对设备执行相同的操作：它将每个任务发送到模型、引擎和设备。

您提供一个任务（嵌入、重新排序、自然语言推理、验证、生成）以及重要的约束条件（质量、延迟、上下文、推理级别）。它会返回一个模型、一个引擎、一个设备和设置，并显示每个数值背后的收据。一旦您选择了模型，OpenVINO 的 AUTO 和 HETERO 插件就可以选择一个设备。该堆栈中的任何内容都不会选择模型。这个差距就是该产品的核心。

## 安装

从克隆版本：

```
uv sync
uv run interlock --help
```

从 PyPI：

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

发行包的名称是`mcptoolshop-interlock`。导入和命令是`interlock`。Python 3.13、3.14 和 3.15（`requires-python`是`>=3.13`；3.13 是 Colab 默认镜像运行的版本）。CI 在所有三个版本上，在 Ubuntu 上运行测试，其中 3.15 是主要版本。在 3.15 上，PyYAML 会从源代码安装，直到它发布 cp315 轮子（当 libyaml 不存在时，它会构建一个纯 Python 轮子）。`intel` 附加组件需要 Python 3.13 或 3.14：OpenVINO 2026.4.1 尚未提供 cp315 轮子，因此在 3.15 上，附加组件会安装`transformers`，但不会安装`openvino*`（一个环境标记）；核心组件在 3.15 上运行。针对测试平台的实时测量运行，是为用于测量此表格的 Windows 测试平台编写的。Linux CUDA 路径（1.0.0 版本中的新功能）在远程 Colab 虚拟机上运行。

此程序允许访问的内容在 [威胁模型](#威胁模型) 中。

收据模式、路由表和测试框架是最先创建的。`interlock route` 读取该表。它不会消耗设备时间。

## 1.0.0 版本中的新内容

1.0.0 遵循 0.2.0 版本。没有单独的 0.3.0 版本；其更改包含在 1.0.0 版本中。

- **远程 Colab 设备。** `colab:A100-40`、`colab:A100-80`、`colab:L4`、`colab:T4` 和 `colab:G4` 行包含设备的标识（`provider`、`card`、`vram_mib`、驱动程序和 CUDA 版本，以及一个区域 `pool`），并且位于各自的池中，与测试平台上的行分开。请参阅 [设备](#设备)。
- **适用于这些显卡的 Linux CUDA 测试路径**（`interlock run`，具有 `platform: linux-cuda` 规格），包含三个预先注册的规格：`experiments/c2-fill-card/`、`experiments/c2-generate/` 和 `experiments/c2-throughput/`。失败的形状（`oom`、`load_refused`、`timeout`、`ctx_exceeded`）会作为能力行保留。未进行测试的单元（`generation_aborted`，一个已中止的 Ollama 生成过程，或 `unmeasured`，一个未识别的错误或无法达到其长度的提示）也会保留，但无法说明该显卡可以执行哪些操作。每个生成阶梯行都会记录 `settings.ladder_version` 以及模型实际评估的提示长度。
- **每个 C2 模型都指定了一个经过检查的来源：**一个带有其注册表清单摘要的 Ollama 标签，或者一个带有其 Hugging Face 仓库、提交信息和 sha256 的 GGUF 文件。Gemma 4 31B 使用 Ollama 库的 Q4_K_M、Q8_0 和 BF16 构建；它没有发布的 Q6_K 构建，因此该单元预先注册为 `load_refused`。`interlock c2-sources <spec>` 打印每个条目的来源和拉取命令，并且 `--verify-live` 检查每个来源、摘要和量化标签是否与 `registry.ollama.ai` 和 `huggingface.co` 匹配。
- **吞吐量输入是精确的令牌长度。**每个吞吐量模型都会获得固定的 128 令牌和 510 令牌输入，这些输入使用其自身的令牌器在固定的修订版本中构建（510 令牌留出空间用于 512 令牌限制内的特殊令牌），并在 `experiments/c2-throughput/inputs.json` 中提交。重新排序和自然语言推理不在 C2 的范围内：默认的 Ollama 不支持这两者，因此这些单元预先注册为 `load_refused`，并注明了原因。
- **可选的远程路由。** `interlock route` 除非请求通过 `--allow-remote` 或 `--device colab:<card>`，否则会忽略远程行。远程获胜者会被标记为 `remote: true`，并带有其 `remote_device`。请参阅 [路由](#路由)。
- **导入远程窗口：** `interlock import --remote-window <path>` 保留每个重复插槽，并且 `interlock table --device colab:<card>`（或 `colab:*`）列出远程行。

该表中目前没有远程行：尚未导入任何远程窗口。请参阅 [CHANGELOG.md](CHANGELOG.md) 以获取完整列表。

## 表格中显示的内容

路由表中包含 509 行来自已提交收据的数据，所有数据都在测试平台上进行测量。其中一些答案：

- **小型模型在 Intel 侧表现良好。** Phi-3.5-mini 在 CPU 和 iGPU 上解码速度约为 20-22 个令牌/秒，在 NPU 上解码速度约为 15 个令牌/秒，提示词长度为 1024 个令牌，且不进行推理。这使得 5090 可以用于处理更大的任务。
- **8B 模型不适合在 Intel 侧运行。** 在相同的 1024 个令牌提示词长度且不进行推理的情况下，Qwen3-8B 在 CPU、iGPU 和 NPU 上的解码速度约为 10-12 个令牌/秒。在 5090 上，它在不进行推理的情况下解码速度约为 208 个令牌/秒，在进行推理的情况下解码速度约为 211 个令牌/秒。如果要求在 5090 繁忙的情况下达到 15 个令牌/秒，路由器会拒绝并说明导致差距的原因：`interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`。
- **某些形状根本无法运行。** NPU 在 512 个令牌输入下拒绝 Qwen3-Embedding，无论是逐个处理还是以 8 个批次的形式处理。这些行作为能力发现保留，并且不会被视为速度较慢的候选对象。

这些数字是记录在 E3 和 e3-ref 收据中的解码速率。运行说明位于 [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md) 中。`interlock route` 按最差重复次数进行排名，而不是按中位数进行排名，因此它的选择比这些摘要更谨慎。

## 它不会做什么

interlock 不处理实际流量，也不充当代理。它不训练模型。它不编辑 offrig、npu-serve 或 rnd。这些内容由其所有者负责。服务层需要其自身的简要说明。

## 设备

每个设备都有一个名称。

| 名称 | 它是什么 |
|---|---|
| `NPU` | Core Ultra 9 285K 上的 Intel AI Boost |
| `iGPU` | Intel GPU。实时枚举必须显示供应商 `0x8086` 和设备类型 `INTEGRATED`。记录完整的名称，而不是决策。 |
| `CPU` | 同一处理器的 CPU |
| `5090` | RTX 5090，仅通过 Ollama 在 `127.0.0.1:11434` 处访问 |
| `colab:<card>` | 远程 Colab CUDA 设备（`A100-40`、`A100-80`、`L4`、`T4`、`G4`），通过 NVML 或 `nvidia-smi` 在 Linux 上进行测量。 |

`GPU.1` 是在此配置中 OpenVINO 看到的 5090。即使名称或供应商发生更改，解析器也会拒绝该索引，拒绝一个简单的 `GPU`，拒绝 `AUTO`、`HETERO` 和 `MULTI`，并拒绝不是供应商 `0x8086` 和 `INTEGRATED` 的 GPU。即使设备属性列表中没有，也仍然需要完整名称中包含“Intel”。缺少设备是一个错误。互锁机制不会降级到较弱的设备。

远程行包含身份信息（`remote_device`、`provider: "colab"`、`card`、`vram_mib`、`driver_version`、`cuda_version`，以及一个区域字符串 `pool`——绝不会是账户 ID）。远程行与本地设备行分开，并且绝不会合并到设备池中。`interlock table --device colab:<card>`（或`colab:*`）列出远程行，并且只有在启用时（`--allow-remote`或`--device colab:<card>`），`interlock route`才会考虑远程卡。失败的远程形状（`oom`、`load_refused`、`timeout`、`ctx_exceeded`）会被记录为能力行，并记录其测量的原因，而不是跳过。`interlock route`仅在最新的`ladder_version`中生成阶梯行，并且其注释说明它没有比较多少个较旧的行。

NPU 是独占的。5090 步骤仅在发布者授权内运行。运行器记录该授权。CPU 密集型任务等待发布者宣布的时间窗口。

## 收据

一个路由行命名任务、模型（仓库或标签、修订版或摘要、量化、许可证）、引擎及其版本、设备、设置，以及它测量的质量和延迟。质量指标携带自己的 `n`、间隔和数据集哈希。延迟区分冷启动和热启动。单个运行被标记为临时。该行指向源收据以及导入它的提交，或者指向一个实验规范和一个运行 ID。

一个运行还可以记录一个能力行，而不是速度。`device_lost` 和 `hang` 不携带任何时间信息。`native_binary_rejected`——设备编译了模型，并在执行时拒绝了本机二进制文件——仅保留编译时间作为证据，并且范围限定为模型、设备和静态形状：只有具有相同密钥的配置才会标记为未运行，并且该设备的其它形状仍然会运行。这三者都是临时的，并且绝不会被计为速度。一个在计划中间被中止的配置，其收据中会保留 `aborted: true` 标记，并记录实际完成的插槽；路由表拒绝中止的行，并且 `table/gaps.md` 列出中止的运行。一个监视线程从主线程外部保护每个运行：如果本机编译运行时间超过 `compile_ceiling_s`（默认 900 秒），则会被记录为 `compile_timeout` 能力行，并且如果系统 RAM 达到 88%，则会中止运行；两者都会先清除收据，然后使用退出代码 75（编译超时）或 76（RAM 停止）结束进程。每个运行在触及设备之前还会写入 `receipts/<run_id>.run.json`（git 忽略，不是收据），因此即使从外部终止运行，仍然会留下一个记录，表明它已开始。要从外部干净地停止一个运行，请创建 `receipts/<run_id>.stop`：监视线程会清除收据并使用代码 77 退出。

```
interlock check
```

这会验证 `receipts/` 和 `table/routing.jsonl` 下的每个收据。一个空的树是可以的。收据中的一个主目录路径是不可以的。

## 运行生命周期

一个运行以两种状态之一结束。一个完成的运行会留下 `receipts/<run-id>.jsonl`，以及 E4b 的系列。一个在途中停止的运行会先写入该收据，其中包含完成的调用和 `aborted: true`，然后将收据、系列、日志和运行标记移动到 `aborted/<run-id>-<sha>/`，并添加一个 `NOTE.json`（运行 ID、harness SHA、开始和结束时间、异常以及文件）。`aborted/` 会被 git 忽略。它不是一个结果：`interlock import` 不会读取它，并且这些运行永远不会到达路由表。

`receipts/` 不应包含一个没有收据的 `*-series/` 目录。下一个运行在存在一个目录时会拒绝启动，并命名 `interlock runs quarantine <run-id>`。

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` 打印每个运行 ID 和一个状态：`complete`、`aborted` 或 `stray`。`quarantine` 将一个孤立的配置移动到 `aborted/`，并附带一个注释，说明它是手动设置的。这两个命令都不会启动 Ollama 或设备。`--quarantine` 选择设置好的目录；默认情况下，它位于运行根目录下的 `aborted/`。

如果没有 `--run-id`，E4b 将使用 `e4b-block` 或 `e4b-pilot`，然后是 `<base>-2`、`<base>-3`，依此类推。所选 ID 是运行的第一行，也是收据中的一个字段。如果一个名称已经被一个收据、一个系列目录、一个日志、一个编译缓存或一个隔离的运行使用，则会被拒绝。

## 威胁模型

interlock 触及本地设备和本地 HTTP，并且仅通过网络下载一个实验规范已经命名的模型，从该模型的发布者处下载，或者在被要求时检查 C2 来源。

- 收据和路由表是本地文件。
- 当实际启动一个运行实例时，Ollama 调用会连接到 `127.0.0.1:11434`（5090 端口）或 `127.0.0.1:11490`（CPU）。拒绝使用云模型标签。
- 当实际启动一个运行实例时，OpenVINO 运行实例会使用 `NPU`、`CPU` 或 Intel iGPU（供应商 `0x8086` 和 `INTEGRATED`）。它们绝不会选择 `GPU.1`。
- 如果没有 `--grant`，5090 步将不会启动。如果没有 `--npu-session`，NPU 或 iGPU 步将不会启动。
- 5090 路径会检查 VRAM 监控心跳是否在 60 秒以内，并且没有加载任何模型。心跳文件不会被打开。只会读取其修改时间。
- 这里没有任何进程监听端口。
- interlock 不会发送遥测数据。
- Linux CUDA 路径（`platform: linux-cuda`，用于远程 `colab:<card>` 窗口）在远程 VM 上运行，而不是在设备上运行。它会在一个专用的环回端口（默认 `127.0.0.1:11441`）上启动自己的 Ollama，拒绝共享端口 `11434` 和 `11490`，并且拒绝任何不是环回的 Ollama URL：当 `localhost` 仅解析到环回时，主机必须是 `127.0.0.1:<port>`、`[::1]:<port>` 或 `localhost:<port>`，端口必须是 1 到 65535 之间的 ASCII 数字。它通过 NVML 或 `nvidia-smi` 读取显卡。它不会下载模型，也不会触及 Windows 计数器。
- `interlock c2-sources --verify-live` 是唯一一个用于 C2 的连接到互联网的命令。它向 `registry.ollama.ai`（清单和配置数据块）和 `huggingface.co`（文件树和 `tokenizer.json`）发送只读 HTTPS 请求，除此之外不做任何操作。它不会下载任何模型权重，也不会发送任何凭据。如果没有 `--verify-live`，`c2-sources` 仅读取本地文件。
- 远程窗口的结果仅以一个收据文件形式传递，该文件由 `interlock import --remote-window` 读取。interlock 不会连接到 Colab。一个字段的名称与一个帐户相关（`account`、`user`、`owner`，它们的 `_id` 形式、`ggufRepo`、`gguf_repo` 或 `gguf_source`）必须包含 `sha256:<64-hex>`。远程行也会拒绝带有标签的帐户令牌，例如 `account:<id>` 或 `user=<id>`，这些令牌出现在任何字符串值或键中，并且 `remote_device` 会拒绝其自身字段中的帐户 ID、电子邮件、路径和仓库句柄。普通的词语，例如“面向用户”，可以出现在值或键中。
- interlock 不会租用 GPU、设置支出上限或打开 offrig 的数据库。远程价格仅以一个带有日期的快照文件形式出现。

## 布局

`devices` 是允许列表。`guards` 是授权、会话、心跳和空卡。`receipt` 是模式，`interlock check` 也是。`importers` 从已固定的收据中重建 `table/routing.jsonl`，`importers/remote.py` 读取远程窗口的收据。`interlock table` 打印该表，并按任务或设备进行过滤。`table/gaps.md` 是实验积压。该框架运行一个已提交的规范；`harness/linux_cuda.py` 是用于远程卡的 Linux CUDA 路径。`interlock route` 读取该表，并且不会触及任何设备。

构建简报以及它命名的四个反馈说明都以快照的形式保存在 `docs/handoff/` 中。存储在工作室内存中的副本始终是权威版本。

## 路由

`interlock route` 从 `table/routing.jsonl` 接收响应。它不会启动设备、读取账本或调用 offrig。

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` 匹配仓库或标签的子字符串。连字符和冒号将被忽略，因此 `Qwen3-8B` 匹配 `qwen3:8b`。一个短的子字符串可能会匹配到一个非预期的仓库。匹配的模型将被打印。`--thinking off` 仅匹配 `off`。`--thinking on` 匹配任何记录的思考级别，并且该决策会保留该级别，因此一个 OpenVINO `low` 臂将保持 `low`。当请求明确指定一个时，`low`、`medium` 和 `high` 才会完全匹配。`--min-tok-s` 是最小重复槽解码速率的下限。一个拒绝会指定该下限为 `min_tok_s`。`--budget-left-usd` 是一个输入，例如 `--busy`。一个最坏情况超过该值的远程行是一个备选。如果没有标志，远程建议将被标记为预算未检查。`--allow-remote` 允许远程 `colab:<card>` 行进入候选池（默认关闭），并且 `--device` 按确切的设备（`colab:L4`、`5090`）或提供程序通配符（`colab:*`）进行过滤。`--as-of` 默认为今天。策略本身不会读取时钟。

按最低重复槽解码 tok/s 生成排名。所有其他任务按最高的重复槽延迟或按最大声明时间进行排名，如果该行没有重复槽。边际是这些保守边界之间的差距，而不是中位数。平局会使 5090 保持空闲。超过 30 天的行将被视为过时。标记为临时或重复次数少于两次的行将被视为临时。两者都仍然符合条件，并且会被标记。

`native_binary_rejected`、`compile_rejected`、`compile_timeout`、`hang`、`device_lost`、`oom`、`load_refused`、`timeout`、`ctx_exceeded`，未测量的类型（`sampler_error`、`generation_aborted`、`unmeasured`）以及包含`parity_valid`个“false”（`parity_invalid`）的行，都列在排除列表中，并附带相应的代码。它们不属于候选对象。在结果循环之前，远程行会被过滤掉，除非请求明确选择（`--allow-remote`或`--device colab:...`），因此它们永远不会出现在仅针对特定配置的请求的排除列表或拒绝理由中。当某一行同时不满足奇偶校验要求时，会显示相应的代码，这即为能力结果。当没有行满足约束条件时，命令会退出并返回 1，同时指出导致集合为空的约束条件。它不会回退。`--json`会打印一份与`src/interlock/schema/route-decision.v1.json`相符的文档（用`remote: true`和`remote_device`标记远程获胜者）。`--max-ttft-ms`会拒绝，除非每行比较的行都记录了相同的`settings.prefix_caching`（`on`、`off`或`default`）。“未设置”表示未知。

当远程建议由 RunPod 提案支持时，它会包含一个 `offrig_plan` 对象，或者当由 `colab:<card>` 行支持时，它会包含 `remote: true` 和 `remote_device`。interlock 不会调用 `offrig_plan` 或 `offrig_launch`，也不会进行任何支出。请参见 `docs/offrig-integration.md`。

## 运行实验

每个实验都预先注册：其规范在 `experiments/<id>/spec.yaml` 中提交，包括问题、它提供的决策、各个分支、数据集、指标和停止规则，所有这些都发生在任何设备时间之前。`interlock run` 拒绝在提交后被编辑的规范。在运行之前（`require_unmodified`），会检查规范的 blob 是否与该提交匹配，并且由 `--dry-run` 打印。它不会写入收据。一个 E4b 收据会记录框架 `git_sha` 和 `clean_tree`。一个填充卡收据会记录 `experiment_spec` 和 `run_id`，并且在预检运行时，它还会记录 `git_sha` 和 `clean_tree`。任何其他共享运行器收据仅记录 `experiment_spec` 和 `run_id`。根据惯例，规则不会在看到结果后进行更改；一个带有日期的修订将在结果计入的运行之前进行，并且被设置为无效的运行将被记录为无效。

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

一个 `platform: linux-cuda` 规范（`experiments/c2-*/` 中的 C2 规范）在一个预订的窗口内，在远程 Colab VM 上运行，并针对其自身的 Ollama 在一个专用的环回端口上运行。

`--dry-run` 会打印计划，并列出实际运行所需的各项设置，但不会启动任何操作。实际运行需要其设备所要求的标志：`--grant` 用于 5090，`--npu-session` 用于 NPU 或 iGPU，`--cpu-window` 用于 CPU 密集型环节。它会在 `receipts/` 下写入一个记录。`interlock import --remote-window <path>` 然后将远程 Colab 窗口的记录添加到 `table/`，`interlock check` 验证这些记录。从已固定的源代码仓库进行完全重建需要 `--source`（请参见下文）。

## 路径和看门狗

代码中没有记录任何用户的设备配置路径。 以下每个位置首先是一个环境变量，然后是工作目录中的 `interlock.toml` 中的 `[paths]`，并且使用您提供的值：

- `INTERLOCK_SCRATCH_ROOT` / `scratch_root`：offrig 将其打开的临时目录作为其项目目录，用于 `offrig verify calibrate`。默认：系统临时目录下的 `interlock-scratch`。
- `INTERLOCK_WEIGHTS_HOT` / `weights_hot`： 实验分支中，如果设置了 `hf_cache: hot`，则表示该分支会读取热缓存中的权重。 缺少副本即为拒绝。 默认： 位于 `HF_HOME` 目录下的 `hot/` 文件夹，或者位于用户的 Hugging Face 缓存目录中。
- `WATCHDOG_HEARTBEAT` / `heartbeat`：VRAM 监控程序的“心跳”文件，心跳守护程序会在每次 5090 GPU 运行步骤之前读取该文件。如果没有设置默认值，则运行会失败，并显示错误消息 `WATCHDOG_HEARTBEAT`。如果您不运行任何监控程序，请传递 `--no-watchdog`；收据的 `notes` 会记录这一点。

```toml
[paths]
scratch_root = "scratch"
weights_hot = "hf-hot"
heartbeat = "watchdog/HEARTBEAT"
```

## 导入记录

大多数用户只会进行合并。`interlock import --remote-window <file>` 将 Colab 窗口收据中的行（即 `interlock run` 写入的路由行 JSONL，或旧版本的 `{run_id, date, remote_device, records}` JSON）添加到现有的 `table/routing.jsonl` 中。它不需要其他存储库，并且会拒绝表中已存在的行。

完全重建（`interlock import`，不带 `--remote-window`，或者带 `--offers`）会从三个源仓库重新生成已固定的行，每个仓库都对应一个已固定的提交：`rnd`（`08ca419f647a3ed69ee645610eaa1c7f35f40f91`）、`npu`（`275e51101f7557bf50898608812cd61d25d5ff17`，位于 rnd 的 `npu-buildout` 分支上，并标记为 `data/npu-buildout-275e511`，因此它是 rnd 的克隆）和 `aspire`（`3b05eab928894311993ffa3af2075fef39cccce6`）。使用 `--source name=path`（可重复使用）说明每个克隆的位置，或者在 `interlock.toml` 中，紧邻表格处说明一次：

```toml
[import.sources]
rnd = "../rnd"
npu = "../rnd"
aspire = "../aspire-si"
```

该标志的优先级高于文件，并且没有默认路径。提交不可配置：不包含该提交的克隆会被拒绝。导入操作会将每个源文件的名称、标识符和路径（如您提供的那样）写入到`table/gaps.md`的末尾，绝不会写入任何行。重建操作只会替换它生成的行；表格中的所有其他行都保持不变，并且如果重建操作会删除它拥有的行，则会被拒绝，除非您传递`--allow-shrink`。

## 当 Ollama 挂起时

`interlock ollama-health` 向 `GET /api/version`（在 `127.0.0.1:11434` 上，即 5090 服务器）和 `127.0.0.1:11490`（仅 CPU 服务器）发送请求。当该答案在 5 秒内到达时，端口是健康的。该命令会为每个端口打印一行，如果其中任何一个端口出现故障，则退出并返回 1。

实际的 `interlock run` 仅在 `127.0.0.1:11434` 上重启 Ollama，仅在步骤之间，并且仅在步骤因传输错误、实际挂起或服务器出现故障时的 5xx 错误而失败后重启。出现故障意味着三次健康检查失败，间隔为两秒。如果请求在其自身截止日期内完成，则仅当健康检查随后失败，或者 `/api/ps` 不显示模型正在进行时，才会被视为挂起。一个长时间的提示，其版本仍然响应，并且其模型仍然列出，不被视为挂起。一个 118k 令牌的提示不会因为速度慢而被终止。

端口 `11490` 是 offrig 的 CPU 索引器。健康状态仍然会报告它。出现故障的 `11490` 会停止运行。interlock 永远不会停止该进程，并且没有授权涵盖其重启。

在停止任何操作之前，重启会重新检查 `--grant`、看门狗心跳和 `/api/ps`。如果列出的模型来自另一个会话，则会被拒绝。不可读取的 `/api/ps` 本身不会阻止已经出现故障的服务器的重启。在挂起期间，健康状态仍然可以响应。然后，不可读取的 `/api/ps` 会拒绝重启，因为另一个会话的模型将被丢弃，而不会被看到。

要停止的进程是正在监听 `11434` 的进程。没有任何内容会被按名称终止。除非它是该监听器，否则托盘应用程序将保持运行。`ollama serve` 将从为本次运行记录的起始行中重新播放：argv、每个 `OLLAMA_*` 和 `LLAMA_ARG_*` 变量以及日志文件。Stdout 和 stderr 将附加到该日志中。interlock 不会登录。没有 `OLLAMA_CUDA_BACKEND` 变量。如果必须强制后端，则变量为 `OLLAMA_LLM_LIBRARY`，并且它应该位于记录的起始行中。

记录是一个名为 `INTERLOCK_OLLAMA_SERVE` 的 JSON 文件。`SWITCHYARD_OLLAMA_SERVE`、`SWITCHYARD_OLLAMA_LOG` 和 `SWITCHYARD_OLLAMA_EXE` 仍然适用于此版本，并会发出警告。字段为 `argv`、`env`、`log_path`、`library_path`、`ggml_cuda_sha256` 和 `backend_line`。在此配置中，一个记录的启动是 `cmd /c ollama serve`，带有 `OLLAMA_HOST=127.0.0.1:11434` 和 `OLLAMA_MODELS=E:/AI-Models/Ollama`，并记录到 `E:/AI/ollama-cu134/system-11434.log`。加载的库是 `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`。发布者的 sha256 该文件的操作会启动 `6d3128e946bc9c95`。一个运行会记录它收到的起始行，并重新播放该行。缺少记录会拒绝重启，然后再停止任何操作。

在运行开始时，记录的完整 `OLLAMA_*` 和 `LLAMA_ARG_*` 设置会与活动进程环境进行比较。该比较是精确的。当无法读取进程环境时，服务器的 `server config` 日志行将作为后备，并且该比较也是精确的：仅是已记录默认值一部分的记录会禁用重启。不匹配意味着本次运行不会进行任何重启。原因会作为 `ollama_restart_disabled` 写入运行标记中。没有任何内容会被停止。`OLLAMA_HOST` 与或不带有 `http://` 或 `https://` 方案相匹配。

服务器恢复后，程序会从 `library_path` 读取库 sha256。程序会读取闪存注意力机制、KV 缓存类型、`load_backend` 行以及完整的 `OLLAMA_*` 和 `LLAMA_ARG_*` 集合，这些数据都来自启动后写入的日志字节。在偏移量之前的字节属于之前的服务器，并且那里的某一行不匹配。如果无法读取日志的大小，则该偏移量未知，程序将停止。缺失的日志为空，其偏移量为零。在第一次模型加载之后、但在保留该样本之前，会检查尚未写入的 `load_backend` 行。如果缺少或不同的行，程序将停止。新的分段的服务器配置行必须涵盖程序运行期间所测量的所有变量。进程未设置的默认值也可能会被记录。如果服务器在不同的库或不同的环境下恢复，程序将停止。来自 `nvidia-smi` 的驱动程序 CUDA 版本不是检查对象：重启无法更改它。闪存注意力机制和日志中未回显的 KV 缓存类型会导致匹配失败。这些数据不会从记录中复制并被视为读取操作。

如果服务器在 60 秒内没有恢复正常，或者显卡没有恢复到其基准状态，程序将停止并提示需要有人启动 Ollama。每次运行最多重启两次。一次失败的重启会减少重启次数上限。

每次重启都会写入运行标记和 `settings.ollama_restarts`：时间、端口、原因（`down` 或 `hung`）、旧 PID 和新 PID、恢复正常所需的时间以及版本。正在执行的步骤结果不确定，永远不会通过。只有在规范设置 `retry` 时（默认值为无）才会再次尝试。保留的重试会设置 `settings.retried`。第一次重启后保留的样本会设置 `settings.cold_after_restart`，并且不会包含在热百分位数中。在重启期间进行的测量不会被保留。

`--dry-run` 打印此策略，并且不会联系任何端口。

## 开发

Python 3.15（`.python-version`）；3.13 和 3.14 也可以使用。`uv sync --frozen` 从 lockfile 文件创建项目环境。它不会共享 NPU 环境。OpenVINO 2026.4.1 是额外的组件（`intel`）。测试不需要它。

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## 贡献

错误报告请提交到 GitHub Issues。安全报告请遵循 [SECURITY.md](SECURITY.md)，并且不作为公开问题。更改以拉取请求的形式提交到 `main`。标准和首先要运行的命令在 [CONTRIBUTING.md](CONTRIBUTING.md) 中。

本仓库的部分内容是使用 Grok (xAI) 编写的。Grok、Kimi 和 Claude 的贡献在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 中注明。

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 构建

