<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# interlock：ローカルAIデバイスルーター

> **バージョン1.0.9：研究用ツール。** 1.0から、バージョンはセマンティックバージョニングに従います。CLI、レシート形式、または実験仕様形式に互換性のない変更があった場合、新しいメジャーバージョンがリリースされます。interlockはあくまで推奨ツールであり、モデル、エンジン、およびデバイスを提案するだけで、実際にトラフィックを処理することはありません。すべてのローカル番号は、1つの環境（Intel Core Ultra 9 285K、NPU、およびiGPU、さらにRTX 5090）で測定されており、リモートの`colab:<card>`行は、完全なカードID（`provider`、`card`、`vram_mib`、`driver_version`、`cuda_version`、`pool`）を持つ独自の個別の証拠プールに存在します。そのルーティングを、特定のハードウェアプールの測定された証拠として扱い、一般的なベンチマークとしては扱わないでください。 **1.0.6での修正：** Linux CUDAで1.0.5以前のバージョンで記録された、最初のトークンが表示されるまでの時間と、1以上の同時実行スレッドにおけるスループットは、有効な指標ではありませんでした（各プロンプトはOllamaのプロンプトキャッシュから提供され、同時リクエストは1つのスロットにキューイングされていました）。interlockはこれらを使用しなくなり、バージョン3のデータで置き換えられました。

鉄道のinterlockシステムは、列車が競合する経路に進入しないように制御します。このツールは、デバイスに対して同様の制御を行います。各ジョブをモデル、エンジン、およびデバイスに送信します。

タスク（埋め込み、再ランキング、NLI、検証、生成）と、重要な制約（品質、レイテンシー、コンテキスト、推論レベル）を指定します。interlockは、モデル、エンジン、デバイス、および設定を返します。また、各数値の根拠となるレシートも表示します。OpenVINOのAUTOおよびHETEROプラグインは、モデルをすでに選択した後でデバイスを選択できます。このスタックのどこにも、モデルを選択する機能はありません。このギャップが、このツールの価値です。

## インストール

クローンから：

```
uv sync
uv run interlock --help
```

PyPIから：

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

ディストリビューションの名前は`mcptoolshop-interlock`です。インポートとコマンドは`interlock`です。Python 3.13、3.14、および3.15（`requires-python`は`>=3.13`です。3.13は、Colabのデフォルトイメージで実行されるバージョンです）。CIは、これら3つすべてに対してUbuntu上でテストを実行し、3.15を主要バージョンとして使用します。3.15では、PyYAMLは、cp315ホイールが公開されるまでソースからインストールされます（libyamlが存在しない場合、純粋なPythonホイールをビルドします）。`intel`の追加機能には、Python 3.13または3.14が必要です。OpenVINO 2026.4.1にはまだcp315ホイールがないため、3.15では、追加機能が`transformers`をインストールしますが、`openvino*`はインストールされません（環境マーカー）。コア機能は3.15で動作します。この表で測定に使用されたWindows環境向けに、実際の測定実行を記述します。Linux CUDAパス（バージョン1.0.0で新規追加）は、リモートのColab VM上で実行されます。

このプログラムがアクセスできる範囲は、[脅威モデル](#脅威モデル)に記載されています。

レシートスキーマ、ルーティングテーブル、およびハーネスが最初に作成されました。`interlock route`は、そのテーブルを読み取ります。デバイスの処理時間を消費することはありません。

## 1.0.0で新しくなったこと

1.0.0は、0.2.0の後に続きます。0.3.0という個別のリリースはありません。その変更は、1.0.0に含められています。

- **リモートColabカードをデバイスとして使用。** `colab:A100-40`、`colab:A100-80`、`colab:L4`、`colab:T4`、および`colab:G4`の行には、カードのID（`provider`、`card`、`vram_mib`、ドライバーおよびCUDAバージョン、およびゾーン`pool`）が含まれており、それらは独自のプールに存在し、環境の行とは分離されています。[デバイス](#デバイス)を参照してください。
- **Linux CUDA を使用した測定パス**。対象のカード（`interlock run`、特定の `platform: linux-cuda` 仕様を持つもの）に対して、事前に登録された3つの仕様（`experiments/c2-fill-card/`、`experiments/c2-generate/`、`experiments/c2-throughput/`）を用いて測定を行います。測定に失敗した形状（`oom`、`load_refused`、`timeout`、`ctx_exceeded`）は、対応する機能の行として保持されます。測定されなかったセル（`generation_aborted`、Ollama による生成が中断された場合、または `unmeasured`、認識されないエラー、または指定された長さに達しなかったプロンプト）も保持されますが、カードがどのような処理を実行できるかについては何も示しません。各生成ラダーの行には、`settings.ladder_version`と、モデルが実際に評価したプロンプトの長さが記録されます。
- **すべてのC2モデルは、検証済みのソースを指します。** Ollamaタグとそのレジストリマニフェストダイジェスト、またはGGUFファイルとそのHugging Faceリポジトリ、コミット、およびsha256。Gemma 4 31Bは、OllamaライブラリのQ4_K_M、Q8_0、およびBF16ビルドを使用します。公開されているQ6_Kビルドはないため、そのセルは事前に`load_refused`として登録されています。`interlock c2-sources <spec>`は、各エントリのソースとプルコマンドを出力し、`--verify-live`は、すべてのソース、ダイジェスト、および量子化ラベルを`registry.ollama.ai`および`huggingface.co`と比較して検証します。
- **スループット入力は、正確なトークン長です。** 各スループットモデルは、固定の128トークンおよび510トークンの入力を使用します。これらは、固定されたリビジョンで独自のトークナイザーを使用して構築されます（512トークンの制限内で、特殊トークンに余裕を持たせるために510を使用します）。これは`experiments/c2-throughput/inputs.json`にコミットされています。再ランキングとNLIは、C2では対象外です。標準のOllamaではどちらもサポートされていないため、これらのセルは事前に`load_refused`として登録され、その理由も記載されています。
- **オプションのリモートルーティング。** `interlock route`は、リクエストが`--allow-remote`または`--device colab:<card>`に合格しない限り、リモート行を無視します。リモートの勝者は、その`remote_device`とともに`remote: true`としてマークされます。[ルート](#ルート)を参照してください。
- **リモートウィンドウのインポート。** `interlock import --remote-window <path>`は、すべての繰り返しスロットを保持し、`interlock table --device colab:<card>`（または`colab:*`）は、リモート行を一覧表示します。

テーブルには、まだリモート行は含まれていません。リモートウィンドウはまだインポートされていません。完全なリストについては、[CHANGELOG.md](CHANGELOG.md)を参照してください。

## 今日のテーブルが示すこと

ルーティングテーブルには、コミットされたレシートから509行が含まれており、これらはすべて環境で測定されています。すでにいくつかの結果が得られています。

- **小さなモデルは、Intel側でうまく動作します。** Phi-3.5-miniは、CPUおよびiGPUで約20〜22トークン/秒でデコードされ、NPUでは約15トークン/秒でデコードされます。プロンプトのトークン数は1024トークンで、推論はオフになっています。これにより、5090をより大きなタスクに使用できるようになります。
- **8Bモデルは、Intel側には適していません。** 同じ1024トークンのプロンプトで、推論をオフにすると、Qwen3-8BはCPU、iGPU、およびNPUで約10〜12トークン/秒でデコードされます。5090では、推論をオフにすると約208トークン/秒、オンにすると211トークン/秒でデコードされます。5090がビジー状態の場合に15トークン/秒を要求すると、ルーターは拒否し、そのギャップの理由を説明します。それは、`interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`に記録されています。
- **一部の形状は、まったく実行されません。** NPUは、512トークンの入力で、一度に1つずつ、または8個のバッチで、Qwen3-Embeddingを拒否しました。これらの行は、機能の発見として保持され、遅い候補としてランク付けされることはありません。

これらの数値は、E3およびe3-refのレシートに記録されているデコードレートです。実行ノートは、[experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md)にあります。`interlock route`は、中央値ではなく、最悪の繰り返しに基づいてランク付けするため、その選択は、これらの概要よりも慎重になります。

## このツールでは実行できないこと

interlockは、トラフィックを処理したり、リクエストをプロキシしたりすることはありません。モデルをトレーニングすることも、offrig、npu-serve、またはrndを編集することもありません。これらは、それぞれの所有者に委ねられます。サービングレイヤーには、独自の概要が必要です。

## デバイス

すべてのデバイスには名前が付けられています。

| 名前 | その内容 |
|---|---|
| `NPU` | Core Ultra 9 285KのIntel AI Boost |
| `iGPU` | Intel GPU。ライブ列挙では、ベンダーが`0x8086`、デバイスタイプが`INTEGRATED`であることが確認される必要があります。完全な名前が記録され、決定は行われません。 |
| `CPU` | 同じプロセッサのCPU |
| `5090` | RTX 5090。Ollamaを介してのみ、`127.0.0.1:11434`にアクセスできます。 |
| `colab:<card>` | リモートColab CUDAカード（`A100-40`、`A100-80`、`L4`、`T4`、`G4`）。Linux上でNVMLまたは`nvidia-smi`を使用して測定されます。 |

`GPU.1`は、この環境でOpenVINOが5090をどのように認識するかを示しています。リゾルバーは、名前やベンダーが変更されても、そのインデックスを拒否します。また、単純な`GPU`、`AUTO`、`HETERO`、`MULTI`、およびベンダーが`0x8086`と`INTEGRATED`でないGPUも拒否します。デバイスのプロパティを持たないリストであっても、完全な名前には「Intel」が含まれている必要があります。デバイスが存在しない場合はエラーとなります。interlockは、より性能の低いものに置き換えられることはありません。

リモートの行には、`remote_device` の識別情報（`provider: "colab"`、`card`、`vram_mib`、`driver_version`、`cuda_version`、およびゾーンを示す `pool` 文字列が含まれます。アカウントIDは含まれません）。 リモートの行は、ローカルの装置の行とは分離されており、装置プールに統合されることはありません。 `interlock table --device colab:<card>`（または`colab:*`）は、リモートの行を一覧表示し、`interlock route`は、オプションが有効になっている場合（`--allow-remote`または`--device colab:<card>`）にのみ、リモートのカードを考慮します。 リモートシェイプの処理に失敗した場合（`oom`、`load_refused`、`timeout`、`ctx_exceeded`のエラーが発生した場合）、その原因が記録された状態で、スキップされるのではなく、対応する機能の行として記録されます。 `interlock route`は、最新の`ladder_version`に存在する行のみを対象に、生成ラダーの行をランク付けし、その注釈には、比較しなかった古い行の数が記載されています。

NPUは排他的に使用されます。5090ステップは、パブリッシャーの許可範囲内でのみ実行されます。ランナーは、その許可を記録します。CPU負荷の高い処理は、パブリッシャーが発表するウィンドウを待ちます。

## レシート

ルーティング行には、タスク、モデル（リポジトリまたはタグ、リビジョンまたはダイジェスト、量子化、ライセンス）、エンジンとそのバージョン、デバイス、設定、および測定された品質とレイテンシーが記載されます。品質メトリックは、それぞれ独自の`n`、間隔、およびデータセットハッシュを持ちます。レイテンシーは、コールド状態とウォーム状態を区別します。単一の実行は、仮の状態としてマークされます。行は、ソースレシートと、インポート元のコミット、または実験仕様と実行IDを指します。

実行は、速度の代わりに機能行を記録することもできます。`device_lost`と`hang`には、タイミング情報は含まれません。`native_binary_rejected`—デバイスがモデルをコンパイルし、実行時にネイティブバイナリを拒否した—は、コンパイル時間のみを証拠として保持し、モデル、デバイス、および静的な形状に限定されます。同じキーを持つアームのみが、実行されなかったとマークされ、デバイスの他の形状は引き続き実行されます。これら3つすべては仮の状態であり、速度としてはカウントされません。中止によってスケジュール途中で停止したアームは、レシートに`aborted: true`とマークされ、実際に完了したスロットが保持されます。ルーティングテーブルは、中止された行を拒否し、`table/gaps.md`は中止された実行をリストします。監視スレッドは、メインスレッド外からすべての実行を保護します。ネイティブコンパイルが`compile_ceiling_s`（デフォルトは900秒）を超えて実行された場合、それは`compile_timeout`の機能行として記録され、システムRAMが88%に達すると、実行が中止されます。どちらの場合も、最初にレシートをフラッシュし、終了コード75（コンパイルタイムアウト）または76（RAM停止）でプロセスを終了します。すべての実行は、デバイスにアクセスする前に`receipts/<run_id>.run.json`を書き込みます（gitで無視され、レシートではありません）。そのため、外部から実行を停止しても、実行が開始されたという記録が残ります。外部から実行をクリーンに停止するには、`receipts/<run_id>.stop`を作成します。監視スレッドは、レシートをフラッシュし、コード77で終了します。

```
interlock check
```

これは、`receipts/`と`table/routing.jsonl`の下にあるすべてのレシートを検証します。空のツリーは問題ありません。レシート内のホームディレクトリパスは問題があります。

## 実行ライフサイクル

実行は、2つの状態のうちのいずれかで終了します。完了した実行は、`receipts/<run-id>.jsonl`と、E4bの場合は、そのシリーズをその隣に残します。途中で停止した実行は、最初にそのレシートを、完了した呼び出しと`aborted: true`とともに書き込み、次にレシート、シリーズ、ログ、および実行マーカーを`aborted/<run-id>-<sha>/`に、`NOTE.json`（実行ID、ハーネスSHA、開始と終了、例外、およびファイル）とともに移動します。`aborted/`はgitで無視されます。これは結果ではありません。`interlock import`はそれを読み取りません。また、これらの実行はルーティングテーブルに到達することはありません。

`receipts/`は、レシートを持たない`*-series/`ディレクトリを含んではいけません。次の実行は、そのようなディレクトリが存在する場合、開始を拒否し、`interlock runs quarantine <run-id>`という名前を付けます。

```
interlock runs list
interlock runs quarantine <run-id>
```

`list`は、各実行IDと状態（`complete`、`aborted`、または`stray`）を出力します。`quarantine`は、手動で設定されたものを`aborted/`に移動し、その旨をメモとして残します。どちらのコマンドも、Ollamaまたはデバイスを開始しません。`--quarantine`は、設定されたディレクトリを選択します。デフォルトは、実行ルートの下の`aborted/`です。

`--run-id`がない場合、E4bは`e4b-block`または`e4b-pilot`、次に`<base>-2`、`<base>-3`などを使用します。選択されたIDは、実行の最初の行であり、レシートのフィールドです。レシート、シリーズディレクトリ、ログ、コンパイルキャッシュ、または隔離された実行によってすでに使用されている名前は拒否されます。

## 脅威モデル

interlockは、ローカルデバイスとローカルHTTPにアクセスし、ネットワークは、実験仕様ですでに名前が付けられているモデルを、そのモデルのパブリッシャーからダウンロードするため、または要求されたときにC2ソースを確認するためにのみ使用します。

- レシートとルーティングテーブルはローカルファイルです。
- Ollamaは、実際に実行が開始されたときに、`127.0.0.1:11434`（5090）または`127.0.0.1:11490`（CPU）にアクセスします。クラウドモデルタグは拒否されます。
- OpenVINOは、実際に実行が開始されたときに、`NPU`、`CPU`、またはIntel iGPU（ベンダー`0x8086`および`INTEGRATED`）を使用します。決して`GPU.1`は選択しません。
- `--grant`なしの5090ステップは開始されません。NPUまたはiGPUステップで`--npu-session`がない場合も開始されません。
- 5090パスは、VRAMウォッチドッグのハートビートが最大60秒以内であり、モデルがロードされていないことを確認します。ハートビートファイルは開かれず、変更時刻のみが読み取られます。
- ここでポートをリッスンするものは何もありません。
- interlockはテレメトリを送信しません。
- Linux CUDAパス（`platform: linux-cuda`、リモートの`colab:<card>`ウィンドウ用）は、リグではなく、リモートVM上で実行されます。専用のループバックポート（デフォルトは`127.0.0.1:11441`）で独自のOllamaを開始し、共有ポート`11434`および`11490`を拒否し、ループバック以外のOllama URLも拒否します。ホストは、`localhost`がループバックにのみ解決される場合、`127.0.0.1:<port>`、`[::1]:<port>`、または`localhost:<port>`である必要があります。ポートはASCII数字で1から65535の間である必要があります。NVMLまたは`nvidia-smi`を通じてカードを読み取ります。モデルをダウンロードしたり、Windowsカウンターに触れたりすることはありません。
- `interlock c2-sources --verify-live`は、C2のためにインターネットにアクセスする唯一のコマンドです。読み取り専用のHTTPSリクエストを`registry.ollama.ai`（マニフェストと構成blob）、および`huggingface.co`（ファイルツリーと`tokenizer.json`）に送信するだけで、それ以上のことはありません。モデルの重みをダウンロードしたり、認証情報を送信したりすることはありません。`--verify-live`がない場合、`c2-sources`はローカルファイルのみを読み取ります。
- リモートウィンドウの結果は、`interlock import --remote-window`が読み取るレシートファイルとしてのみテーブルに到達します。interlockはColabに接続しません。アカウント（`account`、`user`、`owner`、それらの`_id`フォーム、`ggufRepo`、`gguf_repo`または`gguf_source`）という名前のフィールドには、`sha256:<64-hex>`が含まれている必要があります。リモート行も、文字列値またはキーにタグ付けされたアカウントトークン（例：`account:<id>`または`user=<id>`）を拒否し、`remote_device`は、独自のフィールドにあるアカウントID、メールアドレス、パス、およびリポジトリハンドルを拒否します。 "ユーザー向け"のような一般的な単語は、値またはキーとして使用できます。
- interlockはGPUをレンタルしたり、支出上限を設定したり、offrigのデータベースを開いたりすることはありません。リモート価格は、日付付きのスナップショットファイルとしてのみ入力されます。

## レイアウト

`devices`は許可リストです。`guards`は許可、セッション、ハートビート、および空のカードです。`receipt`はスキーマ、`interlock check`です。`importers`は、固定されたレシートから`table/routing.jsonl`を再構築し、`importers/remote.py`はリモートウィンドウのレシートを読み取ります。`interlock table`は、そのテーブルをタスクまたはデバイスでフィルタリングして出力します。`table/gaps.md`は、実験のバックログです。ハーネスはコミットされた仕様を実行します。`harness/linux_cuda.py`は、リモートカード用のLinux CUDAパスです。`interlock route`は、そのテーブルを読み取り、デバイスに触れません。

ビルドの概要と、それに含まれる4つのフィードバックノートは、`docs/handoff/`にスナップショットとして保存されます。スタジオメモリストア内のコピーは、常に最新の状態を保ちます。

## ルート

`interlock route`は、`table/routing.jsonl`から応答します。デバイスを開始したり、台帳を読み取ったり、offrigを呼び出したりすることはありません。

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model`は、リポジトリまたはタグのサブ文字列に一致します。ハイフンとコロンは無視されるため、`Qwen3-8B`は`qwen3:8b`に一致します。短いサブ文字列は、意図しないリポジトリに一致する可能性があります。一致したモデルが出力されます。`--thinking off`は、`off`にのみ一致します。`--thinking on`は、記録された思考レベルのいずれかに一致し、その決定はそのレベルを維持するため、OpenVINO `low`アームは`low`のままになります。`low`、`medium`、および`high`は、リクエストでそれらが指定された場合にのみ正確に一致します。`--min-tok-s`は、最小リピートスロットデコードレートの下限です。拒否は、その下限を`min_tok_s`として指定します。`--budget-left-usd`は、`--busy`のような入力です。最悪の場合がそれを超えるリモート行は、次点です。フラグがない場合、リモートの提案は、予算がチェックされていないとマークされます。`--allow-remote`は、リモートの`colab:<card>`行を候補プールに含めることを許可します（デフォルトはオフ）、および`--device`は、正確なデバイス（`colab:L4`、`5090`）またはプロバイダーのワイルドカード（`colab:*`）でフィルタリングします。`--as-of`はデフォルトで今日の日付になります。ポリシー自体は、時計を読み取りません。

最も低いリピートスロットデコードtok/sでランクを生成します。他のすべてのタスクは、最も高いリピートスロット遅延、または行にリピートスロットがない場合は、最大クレームウォールでランク付けされます。マージンは、これらの保守的な境界間のギャップであり、中央値ではありません。タイの場合、5090は空きになります。30日以上経過した行は、古いと見なされます。一時的とマークされた行、またはリピートが2つ未満の行は、一時的です。どちらも引き続き有効であり、フラグが立てられます。

`native_binary_rejected`、`compile_rejected`、`compile_timeout`、`hang`、`device_lost`、`oom`、`load_refused`、`timeout`、`ctx_exceeded`、測定されていない種類（`sampler_error`、`generation_aborted`、`unmeasured`）と、`parity_valid`がfalse（`parity_invalid`）である行が、それらのコードとともに、除外リストに記載されています。これらは候補ではありません。リモート行は、リクエストで明示的に許可されない限り（`--allow-remote`または`--device colab:...`）、結果ループの前にフィルタリングされるため、リグ専用リクエストの除外リストや拒否理由には決して表示されません。機能の実行結果は、その行がパリティチェックに失敗した場合に表示されるコードです。制約を満たす行がない場合、コマンドは1を返して終了し、空になったセットの原因となった制約の名前を表示します。フォールバックは行われません。`--json`は、`src/interlock/schema/route-decision.v1.json`に対して有効なドキュメントを出力します（リモートで勝った行には`remote: true`と`remote_device`を付加します）。`--max-ttft-ms`は、比較されたすべての行が同じ`settings.prefix_caching`（`on`、`off`、または`default`）を記録している場合にのみ、リクエストを受け入れます。未設定は不明を意味します。

リモートの提案には、RunPodのオファーによってバックアップされている場合は`offrig_plan`オブジェクト、または`colab:<card>`行によってバックアップされている場合は`remote: true`および`remote_device`が含まれます。interlockは`offrig_plan`または`offrig_launch`を呼び出したり、支出したりすることはありません。`docs/offrig-integration.md`を参照してください。

## 実験を実行する

すべての実験は事前に登録されます。その仕様は、質問、それが供給する決定、アーム、データセット、メトリック、および停止ルールとともに、デバイス時間が始まる前に`experiments/<id>/spec.yaml`にコミットされます。`interlock run`は、コミット後に編集された仕様を拒否します。仕様のblobは、実行前にそのコミットに対してチェックされ（`require_unmodified`）、`--dry-run`によって出力されます。レシートには書き込まれません。E4bレシートは、ハーネス`git_sha`および`clean_tree`を記録します。フィルカードレシートは、`experiment_spec`および`run_id`を記録し、プレフライトが実行された場合は、`git_sha`および`clean_tree`も記録します。他の共有ランナーレシートは、`experiment_spec`および`run_id`のみを記録します。慣例として、ルールは結果が確認された後に変更されません。日付付きの修正は、結果がカウントされる実行の前に発生し、無効として設定された実行は、そのように記録されます。

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`platform: linux-cuda`仕様（`experiments/c2-*/`の仕様の下にあるC2仕様）は、予約されたウィンドウ内のリモートColab VM上で、専用のループバックポート上の独自のOllamaに対して実行されます。

`--dry-run` は、実際の実行に必要な計画とガードを表示し、何も実行しません。実際の実行には、そのデバイスが必要とするフラグが必要です。具体的には、5090には`--grant`、NPUまたはiGPUには`--npu-session`、CPU負荷の高い処理には`--cpu-window`が必要です。そして、`receipts/`の下に1つのレシートを書き込みます。次に、`interlock import --remote-window <path>`はリモートのColabウィンドウのレシートを`table/`に追加し、`interlock check`はレシートを検証します。固定されたソースリポジトリから完全に再構築するには、`--source`が必要です（下記参照）。

## パスと監視機能

どの環境（リグ）のパスもコードには書かれていません。 以下に示す各場所は、まず環境変数として、次に作業ディレクトリ内の`interlock.toml`に`[paths]`として配置され、指定された値が使用されます。

- `INTERLOCK_SCRATCH_ROOT` / `scratch_root`：`offrig verify calibrate` コマンドを実行する際に、offrigがプロジェクトとして使用する一時ディレクトリ。デフォルト：システムの一時ディレクトリにある`interlock-scratch`。
- `INTERLOCK_WEIGHTS_HOT` / `weights_hot`: `hf_cache: hot` を設定した実験の特定のブランチが、そこから重みを読み込むホットウェイトキャッシュ。 コピーが見つからない場合は、拒否とみなされます。 デフォルト： `HF_HOME`ディレクトリの下にある`hot/`ディレクトリ、またはユーザーのHugging Faceキャッシュディレクトリの下。
- `WATCHDOG_HEARTBEAT` / `heartbeat`：VRAM監視機能のハートビートファイル。これは、5090で各ステップを実行する前に、ハートビート監視機能が読み取るファイルです。デフォルト値は設定されていません。設定しない場合、実行は拒否され、エラーメッセージに`WATCHDOG_HEARTBEAT`が表示されます。監視機能を実行しない場合は、`--no-watchdog`を指定してください。その設定は、ログの`notes`に記録されます。

```toml
[paths]
scratch_root = "scratch"
weights_hot = "hf-hot"
heartbeat = "watchdog/HEARTBEAT"
```

## レシートのインポート

ほとんどのユーザーは単にマージするだけです。`interlock import --remote-window <file>`は、Colabウィンドウのレシート（`interlock run`が書き込むルーティング行JSONL、または古い`{run_id, date, remote_device, records}`JSON）の行を、既存の`table/routing.jsonl`に追加します。他のリポジトリは必要なく、すでにテーブルに存在する行は拒否します。

完全な再構築（`--remote-window` オプションなし、または `--offers` オプション付きの `interlock import`）では、3つのソースリポジトリから固定された行を再生成します。各リポジトリでは、特定のコミットが固定として使用されます。 `rnd`（`08ca419f647a3ed69ee645610eaa1c7f35f40f91`）、`npu`（`275e51101f7557bf50898608812cd61d25d5ff17`、`rnd`の`npu-buildout`ブランチにあり、`data/npu-buildout-275e511`というタグが付けられているため、`rnd`のクローンである）、および`aspire`（`3b05eab928894311993ffa3af2075fef39cccce6`）。 各クローンの場所を `--source name=path` で指定します（複数回指定可能）。または、テーブルの隣にある `interlock.toml` で一度だけ指定します。

```toml
[import.sources]
rnd = "../rnd"
npu = "../rnd"
aspire = "../aspire-si"
```

フラグはファイルよりも優先され、デフォルトのパスは存在しません。 このコミットは設定変更できません。 保持されないクローンは拒否されます。 インポート処理では、各ソースの名前、ピン、およびパス（指定されたとおり）を`table/gaps.md`の末尾に書き込みますが、行には書き込みません。 再構築では、生成された行のみが置き換えられます。 テーブル内の他のすべての行はそのまま残り、その行を削除する再構築は、`--allow-shrink` を指定しない限り拒否されます。

## Ollama がハングアップした場合

`interlock ollama-health` は、`127.0.0.1:11434` (5090 サーバー) および `127.0.0.1:11490` (CPU 専用サーバー) で `GET /api/version` を要求します。その応答が 5 秒以内に到着した場合、ポートは正常です。コマンドは、各ポートに対して 1 行を出力し、どちらかがダウンしている場合は 1 で終了します。

実際の `interlock run` は、`127.0.0.1:11434` 上でのみ、ステップの間のみ、およびステップがトランスポートエラー、実際のハングアップ、またはそのサーバーがダウンしている間の 5xx エラーで失敗した場合にのみ、Ollama を再起動します。ダウンとは、2 秒間隔で 3 回の正常性チェックが失敗することを意味します。独自の期限内に完了したリクエストは、正常性チェックが失敗した場合、または `/api/ps` がモデルの進行状況を示さない場合にのみ、ハングアップと見なされます。バージョンがまだ応答し、モデルがまだリストされている長いプロンプトは、ハングアップではありません。118k トークンのプロンプトは、速度が遅いという理由で終了されません。

ポート `11490` は、offrig の CPU インデクサーです。正常性は、引き続きこれを報告します。ダウンした `11490` は、実行を停止します。interlock は、このプロセスを停止せず、再起動をカバーする許可もありません。

何かが停止される直前に、再起動は `--grant`、監視機能のハートビート、および `/api/ps` を再確認します。別のセッションからのモデルをリストするリストは拒否されます。読み取れない `/api/ps` は、それ自体では、すでにダウンしているサーバーの再起動をブロックするものではありません。ハングアップ中に、正常性は引き続き応答できます。読み取れない `/api/ps` は、別のセッションのモデルが確認されることなく削除されるため、再起動を拒否します。

停止されるプロセスは、`11434` でリッスンしているプロセスです。名前で何も終了されません。トレイアプリは、それがリスナーでない限り、起動したままになります。 `ollama serve` は、この実行のために記録された開始行から再生されます。argv、すべての `OLLAMA_*` および `LLAMA_ARG_*` 変数、およびログファイルです。標準出力と標準エラーは、そのログに追加されます。interlock はサインインしません。 `OLLAMA_CUDA_BACKEND` 変数はありません。バックエンドを強制する必要がある場合は、変数は `OLLAMA_LLM_LIBRARY` であり、記録された開始行に含める必要があります。

レコードは、`INTERLOCK_OLLAMA_SERVE` という名前の JSON ファイルです。 `SWITCHYARD_OLLAMA_SERVE`、`SWITCHYARD_OLLAMA_LOG`、および `SWITCHYARD_OLLAMA_EXE` は、このリリースでも引き続き機能し、警告を表示します。フィールドは、`argv`、`env`、`log_path`、`library_path`、`ggml_cuda_sha256`、および `backend_line` です。この環境での 1 つの記録された開始は、`cmd /c ollama serve` (`OLLAMA_HOST=127.0.0.1:11434` および `OLLAMA_MODELS=E:/AI-Models/Ollama` を使用) で、ログは `E:/AI/ollama-cu134/system-11434.log` に記録されます。ロードされたライブラリは `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll` です。パブリッシャーの sha256 がそのファイルの `6d3128e946bc9c95` を開始します。実行は、与えられた開始行を記録し、それを再生します。レコードがない場合、何かが停止される前に再起動は拒否されます。

実行の開始時に、レコードの完全な `OLLAMA_*` および `LLAMA_ARG_*` セットは、ライブプロセスの環境と比較されます。その比較は正確です。プロセスの環境を読み取れない場合、サーバーの `server config` ログ行がフォールバックとなり、その比較も正確です。ログに記録されたデフォルトの一部のみであるレコードは、再起動を無効にします。不一致がある場合、この実行では再起動は行われません。その理由は、実行マーカーに `ollama_restart_disabled` として書き込まれます。何も停止されません。 `OLLAMA_HOST` は、`http://` または `https://` スキーマの有無にかかわらず一致します。

サーバーが復旧した後、実行はライブラリsha256を`library_path`から読み込みます。フラッシュアテンション、KVキャッシュのタイプ、`load_backend`行、および完全な`OLLAMA_*`と`LLAMA_ARG_*`セットは、スプawn後に書き込まれたログバイトから読み込まれます。それより前のオフセットにあるバイトは、前のサーバーのものであり、その行は一致しません。ログのサイズを読み取れない場合、そのオフセットは不明であり、実行は停止します。ログが存在しない場合は、空であり、そのオフセットはゼロです。まだ書き込まれていない`load_backend`行は、最初のモデルのロード後、およびそのサンプルが保持される前にチェックされます。ログが存在しないか、異なる行がある場合、実行は停止します。新しいセグメントのサーバー構成行は、実行が測定されたすべての変数を網羅する必要があります。プロセスが設定しなかったデフォルト値もログに記録される場合があります。異なるライブラリ、または異なる環境で復旧したサーバーは、実行を停止します。`nvidia-smi`からのドライバーCUDAバージョンはチェック対象ではありません。再起動では変更できません。ログにエコーされないフラッシュアテンションとKVキャッシュのタイプは、一致に失敗します。これらはレコードから転送されず、読み込みとして扱われます。

サーバーが60秒以内に正常に動作しないか、カードが元の状態に戻らない場合、実行は停止し、Ollamaを手動で開始する必要があると表示されます。1回の実行で最大2回まで再起動できます。再起動が失敗した場合、その回数は上限にカウントされます。

各再起動は、実行マーカーと`settings.ollama_restarts`に記録されます。記録される内容は、いつ、どのポートで、なぜ（`down`または`hung`）、古いPIDと新しいPID、正常になるまでの時間、およびバージョンです。実行中に処理されていたステップは、完了せず、成功とはみなされません。これは、specで`retry`が設定されている場合（デフォルトは設定なし）にのみ、再度試行されます。保持された再試行は、`settings.retried`を設定します。再起動後の最初の保持されたサンプルは、`settings.cold_after_restart`を設定し、ウォームパーセンタイルから除外されます。再起動を跨いで取得された測定値は、保持されません。

`--dry-run`は、このポリシーを出力し、どちらのポートにも接続しません。

## 開発

Python 3.15（`.python-version`）。3.13と3.14も使用可能。`uv sync --frozen`は、ロックファイルからプロジェクト環境を作成する。NPU環境は共有されない。OpenVINO 2026.4.1は、`intel`に追加されたものである。テストには必要ない。

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## 貢献

バグは、GitHubのIssuesに報告してください。セキュリティレポートは、[SECURITY.md](SECURITY.md)に従い、公開の問題として扱われません。変更は、`main`に対するプルリクエストとして提出してください。基準と最初に実行するコマンドは、[CONTRIBUTING.md](CONTRIBUTING.md)に記載されています。

このリポジトリの一部は、Grok（xAI）を使用して作成されました。Grok、Kimi、Claudeは、[CONTRIBUTORS.md](CONTRIBUTORS.md)にクレジットされています。

<a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>によって作成されました。

