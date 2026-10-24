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

> **原型版，0.x：一种研究工具。** 在 0.x 版本的发布之间，命令行界面、收据格式和实验规范可能会发生变化，恕不另行通知。interlock 仅供参考：它提供了一种模型、引擎和设备的建议，但它不会处理实际流量。其中的每个数值都是在一台设备上测得的（英特尔酷睿超 9 285K，配备其 NPU 和 iGPU，以及一张 RTX 5090 显卡）。请将其结果视为该设备的测试数据，而不是通用的基准测试。

铁路互锁系统可以防止列车被路由到冲突的轨道上。此工具对设备执行相同的操作：它将每个任务发送到一种模型、一种引擎和一种设备。

您提供一个任务（嵌入、重新排序、NLI、验证、生成）以及重要的约束条件（质量、延迟、上下文、推理级别）。它会返回一种模型、一种引擎、一种设备和设置，并显示每个数字背后的收据。OpenVINO 的 AUTO 和 HETERO 插件可以在您已经选择好模型后选择一种设备。该堆栈中的任何内容都不会选择模型。这个差距就是该产品的核心。

## 安装

从克隆版本：

```
uv sync
uv run interlock --help
```

从 PyPI，从 v0.1.0：

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

分发名称为 `mcptoolshop-interlock`。导入命令为 `interlock`。仅支持 Python 3.12（`requires-python` 是 `>=3.12,<3.13`）。CI 在 Ubuntu 上运行测试。针对本表中使用的 Windows 设备编写了一个实时测量运行脚本。

此程序允许访问的内容在 [威胁模型](#威胁模型) 中。

收据模式、路由表和测试框架是最先创建的。`interlock route` 读取该表。它不会消耗设备时间。

## 该表今天显示的内容

路由表包含来自已提交收据的 509 行。它已经提供了一些答案：

- **小型模型在 Intel 侧运行良好。** Phi-3.5-mini 在 CPU 和 iGPU 上以大约 20-22 个令牌/秒的速度解码，在 NPU 上以大约 15 个令牌/秒的速度解码，提示词长度为 1024 个令牌，且不进行推理。这使得 5090 可以用于处理更大的任务。
- **8B 模型不适合在 Intel 侧运行。** 在相同的 1024 个令牌的提示词长度且不进行推理的情况下，Qwen3-8B 在 CPU、iGPU 和 NPU 上以大约 10-12 个令牌/秒的速度解码。在 5090 上，它在不进行推理的情况下以大约 208 个令牌/秒的速度解码，在进行推理的情况下以 211 个令牌/秒的速度解码。如果要求在 5090 繁忙的情况下达到 15 个令牌/秒的速度，路由器会拒绝并说明导致差距的原因：`interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`。
- **某些形状根本无法运行。** NPU 在 512 个令牌的输入下拒绝了 Qwen3-Embedding，无论是逐个处理还是以 8 个批次的形式处理。这些行被保留为能力发现，而不是作为速度较慢的候选对象。

这些数字是 E3 和 e3-ref 收据中记录的解码速率。运行说明在 [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md) 中。`interlock route` 按最差重复次数进行排名，而不是按中位数进行排名，因此它的选择比这些摘要更谨慎。

## 它不会执行的操作

interlock 不处理流量，也不代理请求。它不训练模型。它不编辑 offrig、npu-serve 或 rnd。这些将由其所有者负责。服务层需要自己的简要说明。

## 设备

每个设备都有一个名称。

| 名称 | 它的用途 |
|---|---|
| `NPU` | Core Ultra 9 285K 上的 Intel AI Boost |
| `iGPU` | Intel GPU。实时枚举必须显示供应商 `0x8086` 和设备类型 `INTEGRATED`。记录完整的名称，而不是决策。 |
| `CPU` | 同一处理器的 CPU |
| `5090` | RTX 5090，仅通过 Ollama 在 `127.0.0.1:11434` 处访问 |

`GPU.1` 是 5090，这是 OpenVINO 在此设备上看到的。即使名称或供应商发生变化，解析器也会拒绝该索引，拒绝一个简单的 `GPU`，拒绝 `AUTO`、`HETERO` 和 `MULTI`，并且拒绝一个不是供应商 `0x8086` 和 `INTEGRATED` 的 GPU。即使设备属性列表中没有，也仍然需要完整的名称包含“Intel”。缺少设备是一个错误。interlock 不会降级到较弱的设备。

NPU 是独占的。5090 步骤仅在 Publisher 授权内运行。运行程序记录该授权。CPU 密集型步骤等待 Publisher 宣布的时间窗口。

## 收据

路由行命名任务、模型（仓库或标签、修订版或摘要、量化、许可证）、引擎及其版本、设备、设置，以及它测量的质量和延迟。质量指标带有自己的 `n`、间隔和数据集哈希。延迟将冷启动和热启动分开。单个运行被标记为临时。该行指向源收据和导入的提交，或者指向实验规范和运行 ID。

运行也可以记录能力行，而不是速度。`device_lost` 和 `hang` 不包含时间信息。`native_binary_rejected`——设备编译了模型，并在执行时拒绝了本机二进制文件——仅将编译时间作为证据，并且范围限定为模型、设备和静态形状：只有具有完全相同的键的条目才会被标记为未运行，并且该设备的其它形状仍然会运行。所有这三个条目都是临时的，并且永远不会被视为速度。如果一个步骤在计划中途被中止，则其收据会被标记为 `aborted: true`，并且记录实际完成的插槽；路由表拒绝中止的行，并且 `table/gaps.md` 列出了中止的运行。一个监视线程会保护每个运行，使其免受主线程之外的影响：如果本机编译运行时间超过 `compile_ceiling_s`（默认 900 秒），则会被记录为 `compile_timeout` 能力行，并且如果系统 RAM 达到 88%，则会中止运行；两者都会先刷新收据，然后使用退出代码 75（编译超时）或 76（RAM 停止）结束进程。每个运行还会写入 `receipts/<run_id>.run.json`，然后再访问设备（git 忽略，不是收据），因此即使从外部终止运行，也会留下一个记录，表明它已开始。要从外部干净地停止运行，请创建 `receipts/<run_id>.stop`：监视线程会刷新收据并使用代码 77 退出。

```
interlock check
```

这将验证 `receipts/` 和 `table/routing.jsonl` 下的每个收据。一个空的树是可以的。收据中的一个主目录路径是不可以的。

## 运行生命周期

一个运行结束时，会进入两种状态之一。一个完成的运行会留下 `receipts/<run-id>.jsonl`，以及 E4b 及其相关的序列。一个在途中停止的运行会首先写入该记录，包括已完成的调用和 `aborted: true`，然后将该记录、序列、日志和运行标记移动到 `aborted/<run-id>-<sha>/`，并附加一个 `NOTE.json`（运行 ID、构建 SHA、开始和结束时间、异常以及文件）。`aborted/` 会被 git 忽略。它不是一个结果：`interlock import` 不会读取它，并且这些运行永远不会到达路由表。

`receipts/` 不应包含一个没有记录的 `*-series/` 目录。如果存在这样的目录，下一个运行将拒绝启动，并命名 `interlock runs quarantine <run-id>`。

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` 打印每个运行 ID 和状态：`complete`、`aborted` 或 `stray`。`quarantine` 将一个孤立的记录移动到 `aborted/`，并附带一个注释，说明它是手动设置的。这两个命令都不会启动 Ollama 或任何设备。`--quarantine` 选择设置的目录；默认情况下，该目录位于运行根目录下的 `aborted/`。

如果没有 `--run-id`，E4b 将使用 `e4b-block` 或 `e4b-pilot`，然后是 `<base>-2`、`<base>-3`，依此类推。所选 ID 是运行的第一行，也是记录中的一个字段。如果一个名称已经被一个记录、一个序列目录、一个日志、一个编译缓存或一个隔离的运行使用，则该名称将被拒绝。

## 威胁模型

interlock 涉及本地设备和本地 HTTP，以及网络，仅用于下载一个实验规范已经命名的模型，从该模型的发布者处下载。

- 记录和路由表是本地文件。
- 当实际启动一个运行，Ollama 的调用会发送到 `127.0.0.1:11434`（5090）或 `127.0.0.1:11490`（CPU）。云模型标签将被拒绝。
- 当实际启动一个运行，OpenVINO 运行将使用 `NPU`、`CPU` 或 Intel iGPU（供应商 `0x8086` 和 `INTEGRATED`）。它们永远不会选择 `GPU.1`。
- 如果没有 `--grant`，5090 步骤将不会启动。如果没有 `--npu-session`，NPU 或 iGPU 步骤将不会启动。
- 5090 路径会检查 VRAM 看门狗心跳是否在 60 秒以内，并且没有加载任何模型。心跳文件不会被打开。只会读取它的修改时间。
- 这里没有任何东西监听端口。
- interlock 不发送遥测数据。
- interlock 不会租用 GPU、设置支出上限或打开 offrig 的数据库。远程价格仅以日期快照文件的形式出现。

## 布局

`devices` 是允许列表。`guards` 是授权、会话、心跳和空卡。`receipt` 是模式和 `interlock check`。`importers` 从已固定的记录中重建 `table/routing.jsonl`。`interlock table` 打印该表，并按任务或设备进行过滤。`table/gaps.md` 是实验积压。构建系统运行一个已提交的规范。`interlock route` 读取该表，并且不会触及任何设备。

构建摘要以及它命名的四个反馈注释都以快照的形式存储在 `docs/handoff/` 中。存储在工作室内存中的副本始终是权威版本。

## 路由

`interlock route` 从 `table/routing.jsonl` 回复。它不会启动任何设备、读取任何账本或调用 offrig。

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` 匹配仓库或标签的子字符串。连字符和冒号将被忽略，因此 `Qwen3-8B` 匹配 `qwen3:8b`。一个短的子字符串可能会匹配到一个非预期的仓库。匹配的模型将被打印。`--thinking off` 仅匹配 `off`。`--thinking on` 匹配任何记录的思考级别，并且该决策会保留该级别，因此一个 OpenVINO `low` 臂将保持 `low`。`low`、`medium` 和 `high` 在请求明确指定时才会完全匹配。`--min-tok-s` 是最小重复槽解码率的下限。如果拒绝，该拒绝会命名该下限 `min_tok_s`。`--budget-left-usd` 是一个输入，例如 `--busy`。一个最坏情况超过它的远程行是一个备选。如果没有标志，远程建议将被标记为预算未检查。`--as-of` 默认为今天。策略本身不会读取时钟。

按最低重复槽解码 tok/s 生成排名。所有其他任务按最高的重复槽延迟排名，或者当该行没有重复槽时，按最大的声明时间排名。边际是这些保守边界之间的差距，而不是中位数。平局会使 5090 保持空闲。超过 30 天的行将被视为过时。标记为临时或重复次数少于两次的行将被视为临时。两者都将保持有效，并被标记。

`native_binary_rejected`、`compile_rejected`、`compile_timeout`、`hang`、`device_lost` 以及一个 `parity_valid` 为 false（`parity_invalid`）的行，都将以这些代码的形式列在排除列表中。它们不是候选。当该行也为奇偶校验无效时，能力结果是显示的。如果没有行满足约束条件，该命令将退出并返回 1，并命名导致该集合为空的约束条件。它不会回退。`--json` 打印一个与 `src/interlock/schema/route-decision.v1.json` 匹配的文档。`--max-ttft-ms` 拒绝，除非每行比较的行都记录了相同的 `settings.prefix_caching`（`on`、`off` 或 `default`）。未设置表示未知。

一个远程建议包含一个 `offrig_plan` 对象。interlock 不会调用 `offrig_plan` 或 `offrig_launch`，并且它不会进行任何支出。请参见 `docs/offrig-integration.md`。

## 运行实验

每个实验都是预先注册的：其规范在 `experiments/<id>/spec.yaml` 下提交，包括问题、它提供的决策、各个分支、数据集、指标和停止规则，所有这些都发生在任何设备时间之前。`interlock run` 拒绝在提交后被编辑的规范。在运行之前（`require_unmodified`），会检查规范的 blob 是否与该提交匹配，并且由 `--dry-run` 打印。它不会被写入记录。一个 E4b 记录会记录构建系统的 `git_sha` 和 `clean_tree`。一个填充卡记录会记录 `experiment_spec` 和 `run_id`，并且当预检运行时，它还会记录 `git_sha` 和 `clean_tree`。任何其他共享运行器记录只会记录 `experiment_spec` 和 `run_id`。根据惯例，规则不会在看到结果后被更改；一个带有日期的修订将在产生结果的运行之前进行，并且被设置为无效的运行将被记录为无效。

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` 打印计划以及实际运行所需的保护措施，并且不会启动任何内容。实际运行需要其设备调用的标志：`--grant` 用于 5090，`--npu-session` 用于 NPU 或 iGPU，以及 `--cpu-window` 用于 CPU 密集型阶段。它会在 `receipts/` 下写入一个记录。`interlock import` 然后从已提交的记录中重建 `table/`，并且 `interlock check` 验证它们。

## 当 Ollama 挂起时

`interlock ollama-health` 向 `GET /api/version` 发送请求，地址为 `127.0.0.1:11434`（5090 服务器）和 `127.0.0.1:11490`（仅 CPU 服务器）。如果该答案在 5 秒内到达，则该端口是健康的。该命令会打印每行一个端口，如果任何一个端口出现故障，则退出并返回 1。

真正的 `interlock run` 仅在 `127.0.0.1:11434` 时重启 Ollama，仅在步骤之间，并且仅在某个步骤因传输错误、实际的超时或服务器关闭时的 5xx 错误而失败后重启。关闭意味着三次健康检查失败，间隔两秒。如果某个请求在其自身截止日期内完成，则只有在健康检查失败或 `/api/ps` 未显示模型正在取得进展时，才算作卡住。即使其版本仍在响应并且其模型仍在列表中，长时间提示也不会被视为卡住。118k 令牌的提示不会因为速度慢而被终止。

端口 `11490` 是 offrig 的 CPU 索引器。健康检查仍然报告其状态。如果 `11490` 停止运行，则会停止整个过程，interlock 永远不会停止该过程，并且没有授权可以覆盖其重启。

在停止任何内容之前，重启会重新检查 `--grant`、看门狗心跳和 `/api/ps`。如果列表中包含来自另一个会话的模型，则会被拒绝。即使 `/api/ps` 无法读取，也不会单独阻止已经停止的服务器的重启。在卡住期间，健康检查仍然可以响应。如果 `/api/ps` 无法读取，则会拒绝重启，因为另一个会话的模型将被丢弃，而没有被看到。

被停止的过程是监听 `11434` 的过程。不会通过名称终止任何内容。除非该监听器就是正在运行的程序，否则托盘应用程序将继续运行。`ollama serve` 将从为本次运行记录的起始行中重新播放：argv、每个 `OLLAMA_*` 和 `LLAMA_ARG_*` 变量以及日志文件。Stdout 和 stderr 将附加到该日志中。interlock 不会进行登录。没有 `OLLAMA_CUDA_BACKEND` 变量。如果必须强制后端，则变量为 `OLLAMA_LLM_LIBRARY`，并且它应该包含在记录的起始行中。

记录是一个 JSON 文件，其名称由 `INTERLOCK_OLLAMA_SERVE` 确定。`SWITCHYARD_OLLAMA_SERVE`、`SWITCHYARD_OLLAMA_LOG` 和 `SWITCHYARD_OLLAMA_EXE` 仍然适用于此版本，并会发出警告。字段为 `argv`、`env`、`log_path`、`library_path`、`ggml_cuda_sha256` 和 `backend_line`。在此设备上记录的一次启动是 `cmd /c ollama serve`，带有 `OLLAMA_HOST=127.0.0.1:11434` 和 `OLLAMA_MODELS=E:/AI-Models/Ollama`，并记录到 `E:/AI/ollama-cu134/system-11434.log`。加载的库是 `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`。发布者的 sha256 启动 `6d3128e946bc9c95`。运行会记录其给定的起始行并重新播放该行。如果缺少记录，则在停止任何内容之前会拒绝重启。

在运行开始时，记录的完整 `OLLAMA_*` 和 `LLAMA_ARG_*` 集合将与实时进程环境进行比较。该比较是精确的。当无法读取进程环境时，服务器的 `server config` 日志行将作为备用，并且该比较也是精确的：仅包含记录的默认值的记录会禁用重启。不匹配意味着本次运行不会进行任何重启。原因将作为 `ollama_restart_disabled` 写入运行标记中。不会停止任何内容。`OLLAMA_HOST` 匹配，无论是否有 `http://` 或 `https://` 方案。

服务器恢复后，运行会从 `library_path` 读取库 sha256。闪存注意力机制、KV 缓存类型、`load_backend` 行以及完整的 `OLLAMA_*` 和 `LLAMA_ARG_*` 集合都将从生成后写入的日志字节中读取。在此偏移量之前的字节是之前的服务器，并且该行不匹配。如果无法读取日志的大小，则该偏移量未知，并且运行将停止。缺失的日志为空，其偏移量为零。在第一次模型加载之后，但在保留该样本之前，会检查尚未写入的 `load_backend` 行。缺失或不同的行会停止运行。新的片段的服务器配置行必须包含运行期间测量的每个变量。进程未设置的默认值也可能会被记录。如果服务器在不同的库或不同的环境中恢复，则会停止运行。来自 `nvidia-smi` 的驱动程序 CUDA 版本不是检查内容：重启无法更改它。日志中未回显的闪存注意力机制和 KV 缓存类型会导致匹配失败。它们不会从记录中复制并被视为读取。

如果服务器在 60 秒内未恢复正常，或者显卡未恢复到其基线状态，则运行将停止，并提示用户必须启动 Ollama。每次运行最多进行两次重启。失败的重启会减少该限制。

每次重启都会写入运行标记和 `settings.ollama_restarts` 中：时间、哪个端口、原因（`down` 或 `hung`）、旧 PID 和新 PID、恢复正常的时间以及版本。正在进行的步骤是不确定的，永远不会通过。只有在规范设置 `retry` 时（默认值为无），才会再次尝试。保留的重试会设置 `settings.retried`。第一次保留的样本会在重启后设置 `settings.cold_after_restart`，并且不会包含在热百分位数中。在重启期间进行的测量不会被保留。

`--dry-run` 打印此策略，并且不会联系任何端口。

## 开发

Python 3.12。`uv sync --frozen` 从 lockfile 创建项目环境。它不共享 NPU 环境。OpenVINO 2026.4.1 是 `intel` 附加项。测试不需要它。

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## 贡献

错误报告到 GitHub Issues。安全报告遵循 [SECURITY.md](SECURITY.md)，并且不是公开问题。更改以拉取请求的形式提交到 `main`。标准以及首先要运行的命令在 [CONTRIBUTING.md](CONTRIBUTING.md) 中。

此仓库的某些部分是用 Grok (xAI) 编写的。Grok、Kimi 和 Claude 在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 中被提及。

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 构建

