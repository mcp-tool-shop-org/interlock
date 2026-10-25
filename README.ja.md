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

> **バージョン1.0.1：研究用ツール。** 1.0以降のバージョンはセマンティックバージョニングに従います。CLI、レシート形式、または実験仕様形式に互換性のない変更を加えた場合、新しいメジャーバージョンがリリースされます。interlockはあくまで推奨ツールであり、モデル、エンジン、デバイスを提案するだけで、実際にトラフィックを処理することはありません。すべてのローカル番号は、1つの環境（Intel Core Ultra 9 285K、NPU、およびiGPU、さらにRTX 5090）で測定され、リモートの`colab:<card>`行は、完全なカードID（`provider`、`card`、`vram_mib`、`driver_version`、`cuda_version`、`pool`）を持つ独自の独立した証拠プールに保存されます。このツールのルーティングは、特定のハードウェアプールに対する測定された証拠として扱い、一般的なベンチマークとしては使用しないでください。

鉄道のinterlockシステムは、列車が競合する経路に進入しないように制御します。このツールは、デバイスに対して同様の機能を提供します。各ジョブをモデル、エンジン、デバイスに送信します。

タスク（埋め込み、再ランキング、NLI、検証、生成）と、重要な制約（品質、レイテンシー、コンテキスト、推論レベル）を指定します。このツールは、モデル、エンジン、デバイス、および設定を提示し、各数値の根拠となるレシートを表示します。OpenVINOのAUTOおよびHETEROプラグインは、モデルを選択した後でデバイスを選択できます。この一連の処理の中でモデルを選択する機能はありません。このギャップが、このツールの提供する価値です。

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

ディストリビューションの名前は`mcptoolshop-interlock`です。インポートとコマンドは`interlock`です。Python 3.12のみ（`requires-python`は`>=3.12,<3.13`）。CIはUbuntuでテストを実行します。この表の測定に使用されたWindows環境で実行される、実際の測定実行スクリプトがあります。Linux CUDAパス（1.0.0で新規）は、リモートのColab VMで実行されます。

このプログラムがアクセスできる範囲は、[脅威モデル](#脅威モデル)に記載されています。

レシートスキーマ、ルーティングテーブル、およびハーネスが最初に作成されました。`interlock route`がそのテーブルを読み込みます。デバイスの処理時間は消費されません。

## 1.0.0の新機能

1.0.0は0.2.0の後に続きます。0.3.0は個別にリリースされず、その変更は1.0.0に含められています。

- **リモートColabカードをデバイスとして使用。** `colab:A100-40`、`colab:A100-80`、`colab:L4`、`colab:T4`、および`colab:G4`の行には、カードのID（`provider`、`card`、`vram_mib`、ドライバーおよびCUDAバージョン、およびゾーン`pool`）が含まれており、それらは独自のプールに保存され、環境の行とは区別されます。[デバイス](#デバイス)を参照してください。
- **これらのカード用のLinux CUDA測定パス**（`interlock run`、仕様は`platform: linux-cuda`）、3つの事前登録された仕様：`experiments/c2-fill-card/`、`experiments/c2-generate/`、および`experiments/c2-throughput/`。失敗した形状（`oom`、`load_refused`、`timeout`、`ctx_exceeded`）は、機能行として保持されます。
- **すべてのC2モデルは、検証済みのソースを参照します。** Ollamaタグとそのレジストリマニフェストダイジェスト、またはGGUFファイルとそのHugging Faceリポジトリ、コミット、およびsha256。Gemma 4 31Bは、OllamaライブラリのQ4_K_M、Q8_0、およびBF16ビルドを使用します。公開されているQ6_Kビルドはないため、そのセルは事前に`load_refused`として登録されています。`interlock c2-sources <spec>`は、各エントリのソースとプルコマンドを出力し、`--verify-live`は、すべてのソース、ダイジェスト、および量子化ラベルを`registry.ollama.ai`および`huggingface.co`と比較して検証します。
- **スループット入力は正確なトークン長です。** 各スループットモデルは、固定の128トークンおよび510トークンの入力を使用します。これらの入力は、固定されたリビジョンで独自のトークナイザーを使用して構築されます（512トークンの制限内で、特殊トークンに余裕を持たせるために510トークンを使用）。これらの入力は`experiments/c2-throughput/inputs.json`にコミットされています。再ランキングとNLIは、C2では対象外です。標準のOllamaではどちらもサポートされていないため、これらのセルは事前に`load_refused`として登録され、その理由も記載されています。
- **オプションのリモートルーティング。** `interlock route`は、リクエストが`--allow-remote`または`--device colab:<card>`に合格しない限り、リモート行を無視します。リモートの勝者は`remote: true`としてマークされ、その`remote_device`も記録されます。[ルート](#ルート)を参照してください。
- **リモートウィンドウのインポート。** `interlock import --remote-window <path>`は、すべての繰り返しスロットを保持し、`interlock table --device colab:<card>`（または`colab:*`）は、リモート行を一覧表示します。

テーブルにはまだリモート行は含まれていません。リモートウィンドウはまだインポートされていません。完全なリストについては、[CHANGELOG.md](CHANGELOG.md)を参照してください。

## 今日のテーブルが示す内容

ルーティングテーブルには、コミットされたレシートから得られた509行が含まれており、これらはすべて環境で測定されています。すでに得られている答えのいくつか：

- **小型モデルはIntel側で良好に動作します。** Phi-3.5-miniは、CPUおよびiGPUで約20〜22トークン/秒の速度でデコードされ、NPUでは約15トークン/秒の速度でデコードされます。プロンプトのトークン数は1024トークンで、推論はオフになっています。これにより、5090をより大きなタスクに使用できるようになります。
- **8BモデルはIntel側には適していません。** 同じ1024トークンのプロンプトで、推論をオフにした場合、Qwen3-8BはCPU、iGPU、およびNPUで約10〜12トークン/秒の速度でデコードされます。5090では、推論をオフにした場合、約208トークン/秒、オンにした場合、約211トークン/秒でデコードされます。5090がビジー状態の場合に15トークン/秒を要求すると、ルーターは拒否し、そのギャップの理由を説明します：`interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`。
- **一部の形状はまったく実行されません。** NPUは、512トークンの入力で、一度に1つずつ、または8個のバッチでQwen3-Embeddingを拒否しました。これらの行は、機能の発見として保持され、遅い候補としてランク付けされることはありません。

これらの数値は、E3およびe3-refのレシートに記録されたデコードレートです。実行ノートは、[experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md)にあります。`interlock route`は、中央値ではなく、最悪の繰り返しに基づいてランク付けするため、その選択はこれらの概要よりも慎重です。

## このツールが実行しないこと

interlockはトラフィックを処理せず、リクエストをプロキシしません。モデルをトレーニングすることも、offrig、npu-serve、またはrndを編集することもありません。これらは、それぞれの所有者が管理します。サービングレイヤーには、独自の概要が必要です。

## デバイス

すべてのデバイスには名前が付けられています。

| 名前 | 内容 |
|---|---|
| `NPU` | Core Ultra 9 285KのIntel AI Boost |
| `iGPU` | Intel GPU。ライブ列挙では、ベンダーが`0x8086`、デバイスタイプが`INTEGRATED`であることが確認される必要があります。完全な名前が記録され、決定は行われません。 |
| `CPU` | 同じプロセッサのCPU |
| `5090` | RTX 5090。Ollamaを介してのみ、`127.0.0.1:11434`でアクセスできます。 |
| `colab:<card>` | リモートColab CUDAカード（`A100-40`、`A100-80`、`L4`、`T4`、`G4`）。Linux上でNVMLまたは`nvidia-smi`を使用して測定されます。 |

`GPU.1`は、この環境でOpenVINOが5090をどのように認識するかを示しています。リゾルバーは、名前やベンダーが変更されても、そのインデックスを拒否します。また、単純な`GPU`、`AUTO`、`HETERO`、`MULTI`、およびベンダーが`0x8086`と`INTEGRATED`でないGPUも拒否します。デバイスのプロパティを持たないリストであっても、完全な名前には「Intel」が含まれている必要があります。デバイスが存在しない場合はエラーとなります。interlockは、より弱いものに影響を与えることはありません。

リモートの行には、`remote_device`の識別情報（`provider: "colab"`、`card`、`vram_mib`、`driver_version`、`cuda_version`、およびゾーン文字列`pool`—アカウントIDではない）が含まれます。リモートの行は、ローカルの行とは区別され、リソースプールにまとめられることはありません。`interlock table --device colab:<card>`（または`colab:*`）は、リモートの行をリストし、`interlock route`は、オプションが有効になっている場合（`--allow-remote`または`--device colab:<card>`）にのみ、リモートのカードを考慮します。失敗したリモートの形状（`oom`、`load_refused`、`timeout`、`ctx_exceeded`）は、スキップされるのではなく、測定された理由とともに、機能行として記録されます。

NPUは排他的に使用されます。5090ステップは、パブリッシャーの許可範囲内でのみ実行されます。ランナーは、その許可を記録します。CPU負荷の高い処理は、パブリッシャーが発表するウィンドウを待ちます。

## レシート

ルーティング行には、タスク、モデル（リポジトリまたはタグ、リビジョンまたはダイジェスト、量子化、ライセンス）、エンジンとそのバージョン、デバイス、設定、および測定された品質とレイテンシーが記載されます。品質メトリックには、それぞれ固有の`n`、間隔、およびデータセットのハッシュが含まれます。レイテンシーは、コールド状態とウォーム状態を区別します。単一の実行は、仮の状態としてマークされます。行は、ソースレシートと、インポート元のコミット、または実験仕様と実行IDを指します。

実行は、速度の代わりに機能行を記録することもできます。`device_lost`と`hang`には、タイミング情報は含まれません。`native_binary_rejected`—デバイスがモデルをコンパイルし、実行時にネイティブバイナリを拒否した—は、コンパイル時間のみを証拠として保持し、モデル、デバイス、および静的な形状に限定されます。同じキーを持つアームのみが実行されなかったとマークされ、デバイスの他の形状は引き続き実行されます。これら3つすべては仮の状態であり、速度としてはカウントされません。中止によってスケジュール途中で停止したアームは、レシートに`aborted: true`とマークされ、実際に完了したスロットが記録されます。ルーティングテーブルは、中止された行を拒否し、`table/gaps.md`は中止された実行をリストします。監視スレッドは、メインスレッド外からすべての実行を保護します。ネイティブコンパイルが`compile_ceiling_s`（デフォルトは900秒）を超えて実行された場合、それは`compile_timeout`の機能行として記録され、システムRAMが88%に達すると実行が中止されます。どちらの場合も、最初にレシートをフラッシュし、終了コード75（コンパイルタイムアウト）または76（RAM停止）でプロセスを終了します。すべての実行は、デバイスにアクセスする前に`receipts/<run_id>.run.json`を書き込みます（gitで無視され、レシートではありません）。そのため、外部から実行が強制終了された場合でも、実行が開始されたという記録が残ります。外部から実行をクリーンに停止するには、`receipts/<run_id>.stop`を作成します。監視スレッドはレシートをフラッシュし、コード77で終了します。

```
interlock check
```

これは、`receipts/`と`table/routing.jsonl`の下にあるすべてのレシートを検証します。空のツリーは問題ありません。レシート内のホームディレクトリへのパスは問題があります。

## 実行ライフサイクル

実行は、2つの状態のいずれかで終了します。完了した実行は、`receipts/<run-id>.jsonl`と、E4bの場合は、そのシリーズを隣に置きます。途中で停止した実行は、最初にそのレシートを、完了した呼び出しと`aborted: true`とともに書き込み、次にレシート、シリーズ、ログ、および実行マーカーを`aborted/<run-id>-<sha>/`に、`NOTE.json`（実行ID、ハーネスSHA、開始と終了、例外、およびファイル）とともに移動します。`aborted/`はgitで無視されます。これは結果ではありません。`interlock import`はそれを読み取りません。また、これらの実行はルーティングテーブルに到達することはありません。

`receipts/`は、レシートを持たない`*-series/`ディレクトリを含んではいけません。次の実行は、そのようなディレクトリが存在する場合は開始を拒否し、`interlock runs quarantine <run-id>`という名前を付けます。

```
interlock runs list
interlock runs quarantine <run-id>
```

`list`は、各実行IDと状態（`complete`、`aborted`、または`stray`）を出力します。`quarantine`は、手動で設定されたものを`aborted/`に移動し、その旨をメモとして残します。どちらのコマンドも、Ollamaまたはデバイスを開始しません。`--quarantine`は、設定されたディレクトリを選択します。デフォルトは、実行ルートの下の`aborted/`です。

`--run-id`がない場合、E4bは`e4b-block`または`e4b-pilot`、次に`<base>-2`、`<base>-3`などを利用します。選択されたIDは、実行の最初の行であり、レシートのフィールドです。レシート、シリーズディレクトリ、ログ、コンパイルキャッシュ、または隔離された実行によってすでに使用されている名前は拒否されます。

## 脅威モデル

interlockは、ローカルデバイスとローカルHTTPにアクセスし、ネットワークは、実験仕様ですでに名前が付けられているモデルを、そのモデルのパブリッシャーからダウンロードするため、または要求されたときにC2ソースをチェックするためにのみ使用します。

- レシートとルーティングテーブルはローカルファイルです。
- Ollamaは、実際に実行が開始されたときに、`127.0.0.1:11434`（5090）または`127.0.0.1:11490`（CPU）にアクセスします。クラウドモデルタグは拒否されます。
- OpenVINOは、実際に実行が開始されたときに、`NPU`、`CPU`、またはIntel iGPU（ベンダー`0x8086`および`INTEGRATED`）を使用します。決して`GPU.1`は選択しません。
- `--grant`なしの5090ステップは開始されません。NPUまたはiGPUステップで`--npu-session`がない場合も開始されません。
- 5090パスは、VRAM監視のハートビートが最大60秒以内であり、モデルがロードされていないことを確認します。ハートビートファイルは開かれず、変更時刻のみが読み取られます。
- ここでポートをリッスンするものは何もありません。
- interlockはテレメトリを送信しません。
- Linux CUDAパス（`platform: linux-cuda`、リモートの`colab:<card>`ウィンドウ用）は、リグではなく、リモートVM上で実行されます。専用のループバックポート（デフォルトは`127.0.0.1:11441`）で独自のOllamaを開始し、共有ポート`11434`と`11490`を拒否し、ループバック以外のOllama URLも拒否します。ホストは、`localhost`がループバックにのみ解決される場合、`127.0.0.1:<port>`、`[::1]:<port>`、または`localhost:<port>`である必要があります。ポートはASCII数字で1から65535までです。NVMLまたは`nvidia-smi`を通じてカードを読み取ります。モデルをダウンロードしたり、Windowsカウンターに触れたりすることはありません。
- `interlock c2-sources --verify-live`は、C2のためにインターネットにアクセスする唯一のコマンドです。読み取り専用のHTTPSリクエストを`registry.ollama.ai`（マニフェストと構成blob）、および`huggingface.co`（ファイルツリーと`tokenizer.json`）に送信するだけで、それ以外は何も行いません。モデルの重みをダウンロードしたり、認証情報を送信したりすることはありません。`--verify-live`がない場合、`c2-sources`はローカルファイルのみを読み取ります。
- リモートウィンドウの結果は、`interlock import --remote-window`が読み取るレシートファイルとしてのみテーブルに到達します。interlockはColabに接続しません。アカウント（`account`、`user`、`owner`、それらの`_id`フォーム、`ggufRepo`、`gguf_repo`または`gguf_source`）という名前のフィールドには、`sha256:<64-hex>`が含まれている必要があります。リモート行も、文字列値またはキーのいずれかに、タグ付けされたアカウントトークン（例：`account:<id>`または`user=<id>`）を含めることを拒否し、`remote_device`は、独自のフィールド内のアカウントID、メールアドレス、パス、およびリポジトリハンドルを拒否します。 "user-facing"のような一般的な単語は、値またはキーとして使用できます。
- interlockはGPUをレンタルしたり、支出上限を設定したり、オフリグのデータベースを開いたりすることはありません。リモート価格は、日付付きのスナップショットファイルとしてのみ入力されます。

## レイアウト

`devices`は許可リストです。`guards`は許可、セッション、ハートビート、および空のカードです。`receipt`はスキーマ、`interlock check`です。`importers`は、固定されたレシートから`table/routing.jsonl`を再構築し、`importers/remote.py`はリモートウィンドウのレシートを読み取ります。`interlock table`は、タスクまたはデバイスでフィルタリングされたテーブルを出力します。`table/gaps.md`は、実験のバックログです。ハーネスはコミットされた仕様を実行します。`harness/linux_cuda.py`は、リモートカード用のLinux CUDAパスです。`interlock route`は、そのテーブルを読み取り、デバイスに触れません。

ビルドの概要と、それに含まれる4つのフィードバックノートは、`docs/handoff/`にスナップショットとして保存されます。スタジオメモリストア内のコピーは、常に信頼できるものとして保持されます。

## ルート

`interlock route`は、`table/routing.jsonl`から応答します。デバイスを開始したり、台帳を読み取ったり、オフリグを呼び出したりすることはありません。

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model`は、リポジトリまたはタグのサブ文字列に一致します。ハイフンとコロンは無視されるため、`Qwen3-8B`は`qwen3:8b`に一致します。短いサブ文字列は、意図しないリポジトリに一致する可能性があります。一致したモデルが出力されます。`--thinking off`は、`off`にのみ一致します。`--thinking on`は、記録された思考レベルのいずれかに一致し、そのレベルを維持するため、OpenVINO `low`アームは`low`のままになります。`low`、`medium`、および`high`は、リクエストでそれらが指定された場合にのみ正確に一致します。`--min-tok-s`は、最小リピートスロットデコードレートの下限です。拒否は、その下限を`min_tok_s`として指定します。`--budget-left-usd`は、`--busy`のような入力です。最悪の場合がそれを超えるリモート行は、次点です。フラグがない場合、リモートの提案は、予算がチェックされていないとマークされます。`--allow-remote`は、リモートの`colab:<card>`行を候補プールに含めることを許可します（デフォルトはオフ）、および`--device`は、正確なデバイス（`colab:L4`、`5090`）またはプロバイダーワイルドカード（`colab:*`）でフィルタリングします。`--as-of`はデフォルトで今日の日付になります。ポリシー自体は、時計を読み取りません。

最も低いリピートスロットデコードtok/sでランクを生成します。他のすべてのタスクは、最も高いリピートスロットレイテンシ、または行にリピートスロットがない場合は、最大クレームウォールでランク付けされます。マージンは、これらの保守的な境界間のギャップであり、中央値ではありません。タイの場合、5090は空きになります。30日以上経過した行は、古いと見なされます。一時的とマークされている行、またはリピートが2つ未満の行は、一時的です。どちらも引き続き有効であり、フラグが立てられます。

`native_binary_rejected`、`compile_rejected`、`compile_timeout`、`hang`、`device_lost`、`oom`、`load_refused`、`timeout`、`ctx_exceeded`、および`parity_valid`がfalse（`parity_invalid`）の行は、それらのコードとともに、除外リストにリストされます。これらは候補になりません。リモート行は、リクエストで明示的に許可されない限り（`--allow-remote`または`--device colab:...`）、結果ループの前にフィルタリングされるため、リグ専用のリクエストの除外リストまたは拒否理由には表示されません。機能の出力は、行がパリティが無効な場合にも表示されるコードです。制約を満たす行がない場合、コマンドは1を終了し、セットを空にした制約の名前を表示します。フォールバックは行われません。`--json`は、`src/interlock/schema/route-decision.v1.json`に対して有効なドキュメントを出力します（リモートの勝者を`remote: true`と`remote_device`でマークします）。`--max-ttft-ms`は、比較されたすべての行が同じ`settings.prefix_caching`（`on`、`off`、または`default`）を記録しない場合、拒否します。未設定は不明を意味します。

リモートの提案には、RunPodのオファーによってバックアップされている場合は`offrig_plan`オブジェクトが含まれ、`colab:<card>`の行によってバックアップされている場合は`remote: true`と`remote_device`が含まれます。interlockは`offrig_plan`または`offrig_launch`を呼び出したり、支出を行ったりすることはありません。`docs/offrig-integration.md`を参照してください。

## 実験を実行する

すべての実験は事前に登録されます。その仕様は、質問、それが供給する決定、アーム、データセット、メトリック、および停止ルールとともに、デバイス時間が始まる前に`experiments/<id>/spec.yaml`にコミットされます。`interlock run`は、コミット後に編集された仕様を拒否します。仕様のblobは、実行前にそのコミットに対してチェックされ（`require_unmodified`）、`--dry-run`によって出力されます。レシートには書き込まれません。E4bレシートは、ハーネス`git_sha`と`clean_tree`を記録します。フィルカードレシートは、`experiment_spec`と`run_id`を記録し、プレフライトが実行された場合は、`git_sha`と`clean_tree`も記録します。他の共有ランナーレシートは、`experiment_spec`と`run_id`のみを記録します。慣例として、ルールは結果が確認された後に変更されません。日付付きの修正は、結果がカウントされる実行の前に行われ、無効として設定された実行は、そのように記録されます。

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`platform: linux-cuda`仕様（`experiments/c2-*/`の仕様の下にあるC2仕様）は、予約されたウィンドウ内のリモートColab VM上で、専用のループバックポート上の独自のOllamaに対して実行されます。

`--dry-run` は、プランを出力し、実際の実行に必要なガードを表示し、何も開始しません。実際の実行には、そのデバイスが要求するフラグが必要です。 `--grant` は 5090 用、`--npu-session` は NPU または iGPU 用、そして `--cpu-window` は CPU を多用するステップ用です。 `receipts/` の下に 1 つのレシートを書き込みます。 `interlock import` (オプションで `--remote-window <path>` を使用して、リモート Colab ウィンドウの JSON レシート用) が、次にコミットされたレシートから `table/` を再ビルドし、`interlock check` がそれらを検証します。

## Ollama がフリーズした場合

`interlock ollama-health` は、`127.0.0.1:11434` (5090 サーバー) と `127.0.0.1:11490` (CPU 専用サーバー) で `GET /api/version` を要求します。ポートが正常であるのは、その応答が 5 秒以内に届いた場合です。コマンドは、ポートごとに 1 行出力し、どちらかがダウンしている場合は 1 で終了します。

実際の `interlock run` は、`127.0.0.1:11434` でのみ Ollama を再起動し、ステップの間のみ、そしてステップがトランスポートエラー、実際のフリーズであるタイムアウト、またはそのサーバーがダウンしている間の 5xx エラーで失敗した場合にのみ再起動します。ダウンとは、2 秒間隔で 3 回の正常性チェックに失敗することを意味します。自身の期限に間に合ったリクエストは、正常性チェックに失敗した場合、または `/api/ps` がモデルの進行状況を示さない場合にのみ、フリーズと見なされます。バージョンがまだ応答し、モデルがまだリストされている長いプロンプトは、フリーズではありません。118k トークンのプロンプトは、速度が遅いという理由で終了されません。

ポート `11490` は、offrig の CPU インデクサーです。正常性チェックは、まだ正常と報告します。ダウンした `11490` は、実行を停止します。interlock は、そのプロセスを停止せず、再起動をカバーする許可もありません。

何かが停止される直前に、再起動は `--grant`、ウォッチドッグのハートビート、および `/api/ps` を再確認します。別のセッションからのモデルを指すリストは拒否されます。読み取れない `/api/ps` は、それ自体では、すでにダウンしているサーバーの再起動をブロックしません。フリーズ中に、正常性チェックはまだ応答できます。読み取れない `/api/ps` は、次に再起動を拒否します。なぜなら、別のセッションのモデルが、確認されることなく削除されるからです。

停止されるプロセスは、`11434` でリッスンしているプロセスです。名前で何も終了されません。トレイアプリは、それがリスナーでない限り、起動したままになります。 `ollama serve` は、この実行のために記録された開始行から再生されます。argv、すべての `OLLAMA_*` および `LLAMA_ARG_*` 変数、およびログファイルです。Stdout と stderr は、そのログに追加されます。interlock はサインインしません。 `OLLAMA_CUDA_BACKEND` 変数はありません。バックエンドを強制する必要がある場合、その変数は `OLLAMA_LLM_LIBRARY` であり、記録された開始行に含める必要があります。

レコードは、`INTERLOCK_OLLAMA_SERVE` という名前の JSON ファイルです。 `SWITCHYARD_OLLAMA_SERVE`、`SWITCHYARD_OLLAMA_LOG`、および `SWITCHYARD_OLLAMA_EXE` は、このリリースでも機能し、警告を表示します。フィールドは、`argv`、`env`、`log_path`、`library_path`、`ggml_cuda_sha256`、および `backend_line` です。このリグでの 1 つの記録された開始は、`cmd /c ollama serve` で、`OLLAMA_HOST=127.0.0.1:11434` と `OLLAMA_MODELS=E:/AI-Models/Ollama` を使用し、`E:/AI/ollama-cu134/system-11434.log` にログを記録します。ロードされたライブラリは `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll` です。パブリッシャーの sha256 がそのファイルの `6d3128e946bc9c95` を開始します。実行は、与えられた開始行を記録し、それを再生します。レコードが見つからない場合、何かが停止される前に再起動は拒否されます。

実行の開始時に、レコードの完全な `OLLAMA_*` および `LLAMA_ARG_*` セットは、ライブプロセスの環境と比較されます。その比較は正確です。プロセス環境を読み取れない場合、サーバーの `server config` ログ行がフォールバックとなり、その比較も正確です。ログに記録されたデフォルトの一部のみを含むレコードは、再起動を無効にします。不一致は、この実行では再起動が行われないことを意味します。その理由は、実行マーカーに `ollama_restart_disabled` として書き込まれます。何も停止されません。 `OLLAMA_HOST` は、`http://` または `https://` スキーマの有無にかかわらず一致します。

サーバーが復旧した後、実行はライブラリ sha256 を `library_path` から読み取ります。フラッシュアテンション、KV キャッシュタイプ、`load_backend` 行、および完全な `OLLAMA_*` および `LLAMA_ARG_*` セットは、スポーン後に書き込まれたログバイトから読み取られます。それより前のバイトは、前のサーバーであり、その行は一致しません。ログのサイズを読み取れない場合、そのオフセットは不明であり、実行は停止します。ログが見つからない場合、それは空であり、そのオフセットはゼロです。まだ書き込まれていない `load_backend` 行は、最初のモデルのロード後、およびそのサンプルが保持される前にチェックされます。行が見つからないか、異なる場合、実行は停止します。新しいセグメントのサーバー構成行は、実行が測定されたすべての変数をカバーする必要があります。プロセスが設定しなかったデフォルトもログに記録される場合があります。異なるライブラリまたは異なる環境で復旧したサーバーは、実行を停止します。 `nvidia-smi` からのドライバー CUDA バージョンは、チェックではありません。再起動では変更できません。ログにエコーされないフラッシュアテンションと KV キャッシュタイプは、一致に失敗します。それらはレコードからコピーされず、読み取りとして扱われます。

サーバーが 60 秒以内に正常に動作しないか、カードがベースラインに戻らない場合、実行は停止し、ユーザーが Ollama を開始する必要があると表示されます。1 回の実行で再起動できるのは最大 2 回です。再起動が失敗した場合、その制限にカウントされます。

各再起動は、実行マーカーと `settings.ollama_restarts` に書き込まれます。時間、どのポート、理由 (`down` または `hung`)、古い PID と新しい PID、正常になるまでの時間、およびバージョンです。実行中にあったステップは、決定的なものではなく、合格することはありません。それは、仕様で `retry` が設定されている場合にのみ (デフォルトはなし) 再試行されます。保持された再試行は、`settings.retried` を設定します。再起動後の最初の保持されたサンプルは、`settings.cold_after_restart` を設定し、ウォームパーセンタイルから除外されます。再起動をまたいでの測定は、保持されません。

`--dry-run` は、このポリシーを出力し、どちらのポートにも連絡しません。

## 開発

Python 3.12。 `uv sync --frozen` は、ロックファイルからプロジェクト環境を作成します。NPU 環境は共有しません。 OpenVINO 2026.4.1 は、`intel` の追加です。テストには必要ありません。

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## 貢献

バグは GitHub の Issues に報告します。セキュリティレポートは、[SECURITY.md](SECURITY.md) に従って、公開されない問題として報告します。変更は、`main` に対するプルリクエストとして送信します。基準と、最初に実行するコマンドは、[CONTRIBUTING.md](CONTRIBUTING.md) にあります。

このリポジトリの一部は、Grok (xAI) を使用して作成されました。Grok、Kimi、および Claude は、[CONTRIBUTORS.md](CONTRIBUTORS.md) にクレジットされます。

MCP Tool Shop によって構築されました <a href="https://mcp-tool-shop.github.io/"></a>

