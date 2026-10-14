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
  <a href="https://codecov.io/gh/mcp-tool-shop-org/interlock"><img src="https://codecov.io/gh/mcp-tool-shop-org/interlock/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/interlock"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/interlock/badge" alt="OpenSSF Scorecard"></a>
</p>

# interlock：本地 AI 设备路由器

> **原型，0.1.x：一种研究工具，而非正式发布版本。** CLI（命令行界面）、结果格式和实验规范可能会随时更改，并且目前尚未发布到 PyPI。interlock 仅提供建议：它会建议使用某个模型、引擎和设备，但它不会实际处理流量。其中每个数值都是在一套设备上测得的（Intel Core Ultra 9 285K，配备其 NPU 和 iGPU，以及一张 RTX 5090）。请将它的结果视为该设备上的数据，而不是通用的基准测试。

铁路互锁系统可以防止列车被引导到冲突的轨道上。此工具对设备执行相同的操作：它将每个任务发送到模型、引擎和设备。

您提供一个任务（嵌入、重新排序、NLI、验证、生成）以及重要的约束条件（质量、延迟、上下文、推理级别）。它会返回一个模型、一个引擎、一个设备和相应的设置，并显示每个数值背后的结果。OpenVINO 的 AUTO 和 HETERO 插件可以在您已经选择好模型后选择一个设备。该堆栈中的任何组件都不会选择模型。这个差距就是该产品的核心。

## 安装

尚未发布。在首次发布之前，请从克隆版本安装：

```
uv sync
uv run interlock --help
```

首次发布之后：

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

分发名称为 `mcptoolshop-interlock`。导入命令为 `interlock`。仅支持 Python 3.12（`requires-python` 是 `>=3.12,<3.13`）。CI 会在 Ubuntu 上运行测试。针对本表格中使用的 Windows 设备，编写了一个实时测量运行脚本。

该程序允许访问的内容在 [威胁模型](#threat-model) 中。

结果模式、路由表和测试框架是最先创建的。`interlock route` 读取该表。它不会消耗设备时间。

## 当前表格显示的内容

路由表中包含来自已提交结果的 502 行数据。其中一些结果如下：

- **小型模型在 Intel 设备上运行良好。** Phi-3.5-mini 在 CPU 和 iGPU 上以大约 20-22 个令牌/秒的速度进行解码，在 NPU 上以大约 15 个令牌/秒的速度进行解码，提示词长度为 1024 个令牌，且不进行推理。这样可以将 5090 释放出来，用于处理更大的任务。
- **8B 模型不适合在 Intel 设备上运行。** 在相同的 1024 个令牌的提示词长度且不进行推理的情况下，Qwen3-8B 在 CPU、iGPU 和 NPU 上以大约 10-12 个令牌/秒的速度进行解码。在 5090 上，它在不进行推理的情况下以大约 208 个令牌/秒的速度进行解码，进行推理时为 211 个令牌/秒。如果要求在 5090 繁忙的情况下达到 15 个令牌/秒的速度，路由器会拒绝并说明导致差距的原因：`interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`。
- **某些形状根本无法运行。** NPU 拒绝了 Qwen3-Embedding 在 512 个令牌输入下的运行，无论是逐个运行还是以 8 个批次的形式运行。这些行被保留为能力发现，并且不会被视为速度较慢的候选对象。

这些数据是在 E3 和 e3-ref 结果中记录的解码速率。运行说明位于 [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md)。`interlock route` 按照最差重复次数进行排名，而不是按照中位数进行排名，因此它的选择比这些摘要更谨慎。

## 它不会执行的操作

interlock 不会处理流量，也不会代理请求。它不会训练模型。它不会编辑 offrig、npu-serve 或 rnd。这些将由其所有者负责。服务层需要有自己的简要说明。

## 设备

每个设备都有一个名称。

| 名称 | 它的用途 |
|---|---|
| `NPU` | Core Ultra 9 285K 上的 Intel AI Boost |
| `iGPU` | Intel GPU。实时枚举必须显示供应商 `0x8086` 和设备类型 `INTEGRATED`。记录的是完整名称，而不是决策结果。 |
| `CPU` | 同一处理器的 CPU |
| `5090` | RTX 5090，只能通过 Ollama 在 `127.0.0.1:11434` 处访问 |

`GPU.1` 是 OpenVINO 在此设备上看到的 5090。即使名称或供应商发生更改，解析器也会拒绝该索引，拒绝使用 `GPU`、`AUTO`、`HETERO` 和 `MULTI`，并且拒绝使用不是供应商 `0x8086` 和 `INTEGRATED` 的 GPU。即使设备列表不包含任何设备属性，也需要完整名称包含“Intel”。缺少设备是一个错误。interlock 不会自动切换到性能较差的设备。

NPU 是独占的。5090 步骤仅在 Publisher 授权内运行。运行程序会记录该授权。CPU 密集型步骤会等待 Publisher 宣布的时间窗口。

## 结果

路由行命名任务、模型（仓库或标签、修订版或摘要、量化、许可证）、引擎及其版本、设备、设置，以及它测量的质量和延迟。质量指标带有自己的 `n`、时间间隔和数据集哈希。延迟会区分冷启动和热启动。单个运行被标记为临时状态。该行指向源结果和导入的提交，或者指向实验规范和运行 ID。

运行也可以记录能力行，而不是速度。`device_lost` 和 `hang` 不包含任何时间信息。`native_binary_rejected`——设备编译了模型，并在执行时拒绝了本机二进制文件——仅将编译时间作为证据，并且范围限定为模型、设备和静态形状：只有具有完全相同的键的条目才会被标记为未运行，并且设备的其它形状仍然会运行。所有这三个条目都是临时状态，并且不会被计入速度。如果某个步骤在计划中途被中止，则会在其结果中标记为 `aborted: true`，并记录实际完成的插槽；路由表会拒绝中止的行，并且 `table/gaps.md` 列出了中止的运行。一个监视线程会保护每个运行，使其不受主线程之外的影响：如果本机编译运行时间超过 `compile_ceiling_s`（默认 900 秒），则会被记录为 `compile_timeout` 能力行，并且如果系统 RAM 达到 88%，则会中止运行；两者都会先刷新结果，然后使用退出代码 75（编译超时）或 76（RAM 停止）结束进程。每个运行还会写入 `receipts/<run_id>.run.json`，然后再访问设备（git 忽略，不是结果），因此即使从外部终止运行，也会留下记录，表明它已开始。要从外部干净地停止运行，请创建 `receipts/<run_id>.stop`：监视线程会刷新结果并使用代码 77 退出。

```
interlock check
```

它会验证 `receipts/` 和 `table/routing.jsonl` 中的每个收据。一个空的树是可以的。收据中的一个主目录路径是不可以的。

## 运行生命周期

一个运行以两种状态之一结束。一个完成的运行会留下 `receipts/<run-id>.jsonl`，以及对于 E4b，其序列在其旁边。一个在途中停止的运行会首先写入该收据，其中包含已完成的调用和 `aborted: true`，然后将收据、序列、日志和运行标记移动到 `aborted/<run-id>-<sha>/`，并附加一个 `NOTE.json`（运行 ID、构建 SHA、开始和结束时间、异常以及文件）。`aborted/` 会被 git 忽略。它不是一个结果：`interlock import` 不会读取它，并且这些运行永远不会到达路由表。

`receipts/` 不应包含一个没有收据的 `*-series/` 目录。下一个运行将在存在该目录时拒绝启动，并命名 `interlock runs quarantine <run-id>`。

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` 打印每个运行 ID 和一个状态：`complete`、`aborted` 或 `stray`。`quarantine` 将一个孤立的条目移动到 `aborted/`，并附带一个注释，说明它是手动设置的。这两个命令都不会启动 Ollama 或设备。`--quarantine` 选择保留的目录；默认情况下，它是运行根目录下的 `aborted/`。

如果没有 `--run-id`，E4b 将使用 `e4b-block` 或 `e4b-pilot`，然后是 `<base>-2`、`<base>-3`，依此类推。所选 ID 是运行的第一行，也是收据中的一个字段。如果一个收据、序列目录、日志、编译缓存或隔离运行已经使用了该名称，则会拒绝。

## 威胁模型

interlock 涉及本地设备和本地 HTTP，以及网络，仅用于下载实验规范中已经命名的模型，并从该模型的发布者处下载。

- 收据和路由表是本地文件。
- 当实际启动一个运行时，Ollama 的调用会发送到 `127.0.0.1:11434`（5090）或 `127.0.0.1:11490`（CPU）。拒绝使用云模型标签。
- 当实际启动一个运行时，OpenVINO 运行会使用 `NPU`、`CPU` 或 Intel iGPU（供应商 `0x8086` 和 `INTEGRATED`）。它们永远不会选择 `GPU.1`。
- 如果没有 `--grant`，则不会启动 5090 步骤。如果没有 `--npu-session`，则不会启动 NPU 或 iGPU 步骤。
- 5090 路径会检查 VRAM 看门狗心跳是否不超过 60 秒，并且没有加载任何模型。心跳文件不会被打开。只会读取其修改时间。
- 这里没有任何内容监听端口。
- interlock 不发送遥测数据。
- interlock 不会租用 GPU、设置支出上限或打开 offrig 的数据库。远程价格仅作为带有日期的快照文件输入。

## 布局

`devices` 是允许列表。`guards` 是授权、会话、心跳和空卡。`receipt` 是模式和 `interlock check`。`importers` 从已固定的收据中重建 `table/routing.jsonl`。`interlock table` 打印该表，并按任务或设备进行过滤。`table/gaps.md` 是实验积压。构建会运行已提交的规范。`interlock route` 读取该表，并且不会触及任何设备。

构建摘要以及它命名的四个反馈注释会快照到 `docs/handoff/` 中。存储在工作室内存存储中的副本仍然是权威版本。

## 路由

`interlock route` 从 `table/routing.jsonl` 回复。它不会启动设备、读取分类账或调用 offrig。

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` 匹配仓库或标签的子字符串。连字符和冒号将被忽略，因此 `Qwen3-8B` 匹配 `qwen3:8b`。一个短的子字符串可能会匹配到一个非预期的仓库。匹配的模型将被打印。`--thinking off` 仅匹配 `off`。`--thinking on` 匹配任何记录的思考级别，并且该决策会保留该级别，因此 OpenVINO `low` 臂将保持 `low`。`low`、`medium` 和 `high` 在请求命名时才会完全匹配。`--min-tok-s` 是最小重复槽解码率的下限。如果拒绝，则会命名该下限 `min_tok_s`。`--budget-left-usd` 是一个输入，例如 `--busy`。一个最坏情况超过它的远程行是一个备选。如果没有标志，则远程建议将被标记为预算未检查。`--as-of` 默认为今天。策略本身不会读取时钟。

按最低重复槽解码 tok/s 生成排名。所有其他任务按最高的重复槽延迟排名，或者当该行没有重复槽时，按最大的声明时间排名。边际是这些保守界限之间的差距，而不是中位数。平局会使 5090 保持空闲。超过 30 天的行将被视为过时。标记为临时或重复次数少于两次的行将被视为临时。两者都将保持资格并被标记。

`native_binary_rejected`、`compile_rejected`、`compile_timeout`、`hang`、`device_lost` 以及一个具有 `parity_valid` false（`parity_invalid`）的行，都将在排除列表中列出，并附带这些代码。它们不是候选。当该行也为奇偶校验无效时，能力结果是显示的。如果没有行满足约束条件，则该命令将退出并返回 1，并命名使该集合为空的约束条件。它不会回退。`--json` 打印一个与 `src/interlock/schema/route-decision.v1.json` 对应的文档。`--max-ttft-ms` 除非每行比较的行都记录了相同的 `settings.prefix_caching`（`on`、`off` 或 `default`），否则会拒绝。未设置表示未知。

一个远程建议包含一个 `offrig_plan` 对象。interlock 不会调用 `offrig_plan` 或 `offrig_launch`，并且它不会进行支出。请参阅 `docs/offrig-integration.md`。

## 运行实验

每个实验都是预先注册的：其规范在 `experiments/<id>/spec.yaml` 中被提交，其中包含问题、它提供的决策、各个分支、数据集、指标和停止规则，所有这些都早于任何设备时间。`interlock run` 拒绝在提交后被编辑的规范。在运行之前（`require_unmodified`），会检查规范的 blob 是否与该提交匹配，并且由 `--dry-run` 打印。它不会被写入收据。E4b 收据会记录构建 `git_sha` 和 `clean_tree`。填充卡收据会记录 `experiment_spec` 和 `run_id`，并且当预检运行时，它还会记录 `git_sha` 和 `clean_tree`。任何其他共享运行器收据只会记录 `experiment_spec` 和 `run_id`。根据惯例，规则不会在看到结果后被更改；带有日期的修订会在结果被计入的运行之前进行，并且被设置为无效的运行会被记录为无效。

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` 打印计划以及实际运行所需的保护措施，并且不会启动任何内容。实际运行需要其设备调用的标志：`--grant` 用于 5090，`--npu-session` 用于 NPU 或 iGPU，以及 `--cpu-window` 用于 CPU 密集型阶段。它会在 `receipts/` 下写入一个收据。然后，`interlock import` 会从已提交的收据中重建 `table/`，并且 `interlock check` 会验证它们。

## 当 Ollama 挂起时

`interlock ollama-health` asks `GET /api/version` on `127.0.0.1:11434` (the 5090 server) and `127.0.0.1:11490` (the CPU-only server). A port is healthy when that answer arrives within 5 seconds. The command prints one line per port and exits 1 if either is down.

在实际情况下，`interlock run` 仅在 `127.0.0.1:11434` 上重启 Ollama，仅在步骤之间进行，并且仅在某个步骤因传输错误、实际的挂起或服务器出现故障时的 5xx 错误而失败后进行。出现故障意味着连续三次健康检查失败，每次间隔 2 秒。如果某个请求在其自身设定的时间内完成，则只有在健康检查失败或 `/api/ps` 未显示模型正在运行的情况下，才会被视为挂起。如果某个长时间的提示仍然可以得到响应，并且其模型仍然在列表中，则不被视为挂起。一个 118k 令牌的提示不会因为速度慢而被终止。

端口 `11490` 是 offrig 的 CPU 索引器。健康检查仍然会报告其状态。如果 `11490` 出现故障，则会停止运行。interlock 永远不会停止该进程，也没有任何授权可以重启它。

在停止任何内容之前，重启会重新检查 `--grant`、看门狗心跳和 `/api/ps`。如果列表中包含来自另一个会话的模型，则会被拒绝。如果无法读取 `/api/ps`，则本身不会阻止重启已经出现故障的服务器。在挂起期间，健康检查仍然可以返回结果。如果无法读取 `/api/ps`，则会拒绝重启，因为另一个会话的模型可能会在未被看到的情况下被丢弃。

被停止的进程是监听 `11434` 的进程。不会通过名称来终止任何进程。除非该监听器就是正在运行的进程，否则托盘应用程序会继续运行。`ollama serve` 将从为本次运行记录的起始行中重新开始：argv、每个 `OLLAMA_*` 和 `LLAMA_ARG_*` 变量以及日志文件。Stdout 和 stderr 将附加到该日志文件中。interlock 不会进行登录。没有 `OLLAMA_CUDA_BACKEND` 变量。如果必须强制执行后端，则变量为 `OLLAMA_LLM_LIBRARY`，并且它应该包含在记录的起始行中。

记录是一个 JSON 文件，其名称由 `INTERLOCK_OLLAMA_SERVE` 确定。`SWITCHYARD_OLLAMA_SERVE`、`SWITCHYARD_OLLAMA_LOG` 和 `SWITCHYARD_OLLAMA_EXE` 在本次发布中仍然有效，并会发出警告。字段包括 `argv`、`env`、`log_path`、`library_path`、`ggml_cuda_sha256` 和 `backend_line`。在此设备上的一次记录的启动信息是 `cmd /c ollama serve`，带有 `OLLAMA_HOST=127.0.0.1:11434` 和 `OLLAMA_MODELS=E:/AI-Models/Ollama`，并记录到 `E:/AI/ollama-cu134/system-11434.log`。加载的库是 `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`。发布者对该文件的 sha256 值以 `6d3128e946bc9c95` 开头。一次运行会记录其给定的起始行，并重新运行该行。如果缺少记录，则在停止任何内容之前会拒绝重启。

在运行开始时，记录的完整 `OLLAMA_*` 和 `LLAMA_ARG_*` 集合会与当前进程环境进行比较。该比较是精确的。如果无法读取进程环境，则服务器的 `server config` 日志行将作为备用方案，并且该比较也是精确的：仅包含记录的默认值的一部分的记录会禁用重启。不匹配意味着本次运行不会进行任何重启。原因会作为 `ollama_restart_disabled` 写入运行标记中。不会停止任何内容。`OLLAMA_HOST` 可以匹配，无论是否包含 `http://` 或 `https://` 方案。

服务器恢复后，运行会从 `library_path` 读取库的 sha256 值。从生成后写入日志的字节中读取闪存注意力、KV 缓存类型、`load_backend` 行以及完整的 `OLLAMA_*` 和 `LLAMA_ARG_*` 集合。在偏移量之前的字节是之前的服务器，并且那里的某一行不匹配。如果无法读取日志的大小，则该偏移量未知，并且运行会停止。缺失的日志为空，其偏移量为零。在第一次模型加载之后，但在保留该样本之前，会检查尚未写入的 `load_backend` 行。缺失或不同的行会停止运行。新的片段的服务器配置行必须包含运行期间测量的每个变量。进程未设置的默认值也可能会被记录。如果服务器在不同的库或不同的环境中恢复，则会停止运行。驱动程序的 CUDA 版本来自 `nvidia-smi`，这不是检查内容：重启无法更改它。日志中未回显的闪存注意力以及 KV 缓存类型会导致匹配失败。它们不会从记录中复制并被视为读取。

如果服务器在 60 秒内没有恢复健康，或者显卡没有恢复到其基线状态，则运行会停止，并提示用户必须启动 Ollama。每次运行最多进行两次重启。失败的重启会减少该限制。

每次重启都会写入运行标记和 `settings.ollama_restarts` 中：时间、哪个端口、原因（`down` 或 `hung`）、旧的 PID 和新的 PID、恢复健康所需的时间以及版本。正在执行的步骤是不确定的，永远不会被视为通过。只有在规范中设置了 `retry`（默认值为无）时，才会再次尝试。保留的重试会设置 `settings.retried`。在重启之后保留的第一个样本会设置 `settings.cold_after_restart`，并且不会包含在温暖的百分位数中。在重启期间进行的测量不会被保留。

`--dry-run` 打印此策略，并且不会联系任何端口。

## 开发

环境是独立的，位于 `E:/AI/envs/interlock`，Python 3.12。它不共享 NPU 环境。OpenVINO 2026.4.1 是 `intel` 附加项，其版本与该环境相同。测试不需要它。

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## 贡献

错误报告会发送到 GitHub Issues。安全报告遵循 [SECURITY.md](SECURITY.md)，并且不会作为公开问题。更改以拉取请求的形式提交到 `main`。标准以及首先要运行的命令位于 [CONTRIBUTING.md](CONTRIBUTING.md) 中。

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 构建

