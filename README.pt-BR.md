<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

# interlock: roteador de dispositivos de IA local

> **Versão 1.0.0: um instrumento de pesquisa.** A partir da versão 1.0, as versões seguem o versionamento semântico: uma alteração que prejudica a CLI, o formato do recibo ou o formato da especificação do experimento vem com uma nova versão principal. O interlock é apenas uma sugestão: ele sugere um modelo, um motor e um dispositivo, e nunca atende a solicitações. Cada número local nele foi medido em um único dispositivo (Intel Core Ultra 9 285K com sua NPU e iGPU, mais uma RTX 5090), e as linhas remotas `colab:<card>` residem em seu próprio conjunto de evidências separado, com identificação completa da placa (`provider`, `card`, `vram_mib`, `driver_version`, `cuda_version`, `pool`). Considere suas rotas como evidências medidas para esse conjunto de hardware específico, e não como um benchmark geral.

O intertravamento ferroviário impede que um trem seja direcionado para um caminho conflitante. Esta ferramenta faz isso para dispositivos: ela envia cada tarefa para um modelo, um motor e um dispositivo.

Você fornece uma tarefa (incorporação, reclassificação, NLI, verificação, geração) e as restrições que importam (qualidade, latência, contexto, nível de raciocínio). Ele responde com um modelo, um motor, um dispositivo e as configurações, e mostra o recibo por trás de cada número. Os plugins AUTO e HETERO de OpenVINO podem selecionar um dispositivo depois que você já tiver selecionado o modelo. Nada nessa pilha seleciona o modelo. Essa lacuna é o produto.

## Instalar

A partir de um clone:

```
uv sync
uv run interlock --help
```

A partir de PyPI:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

O nome da distribuição é `mcptoolshop-interlock`. O comando de importação é `interlock`. Apenas Python 3.12 (`requires-python` é `>=3.12,<3.13`). O CI executa os testes no Ubuntu. Uma execução de medição ao vivo no dispositivo é escrita para o dispositivo Windows no qual esta tabela foi medida. O caminho Linux CUDA (novo na versão 1.0.0) é executado em uma VM Colab remota.

O que este programa tem permissão para acessar está no [modelo de ameaças](#modelo-de-ameaças).

O esquema do recibo, a tabela de roteamento e o conjunto de testes foram criados primeiro. `interlock route` lê essa tabela. Ele não usa o tempo do dispositivo.

## O que há de novo na versão 1.0.0

A versão 1.0.0 segue a versão 0.2.0. Não houve uma versão separada 0.3.0; suas alterações são incluídas na versão 1.0.0.

- **Placas Colab remotas como dispositivos.** As linhas `colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4` e `colab:G4` contêm a identificação da placa (`provider`, `card`, `vram_mib`, versão do driver e CUDA, e uma zona `pool`) e estão em seu próprio conjunto, separadas das linhas do dispositivo. Consulte [Dispositivos](#dispositivos).
- **Um caminho de medição Linux CUDA** para essas placas (`interlock run` com uma especificação `platform: linux-cuda`), com três especificações pré-registradas: `experiments/c2-fill-card/`, `experiments/c2-generate/` e `experiments/c2-throughput/`. As formas com falha (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) são mantidas como linhas de capacidade.
- **Cada modelo C2 nomeia uma fonte verificada:** uma tag Ollama com seu resumo do manifesto do registro ou um arquivo GGUF com seu repositório Hugging Face, commit e sha256. O Gemma 4 31B usa as compilações Q4_K_M, Q8_0 e BF16 da biblioteca Ollama; ele não tem uma compilação Q6_K publicada, portanto, essa célula é pré-registrada como `load_refused`. `interlock c2-sources <spec>` imprime a fonte e o comando de extração de cada entrada, e `--verify-live` verifica cada fonte, resumo e rótulo de quantização em relação a `registry.ollama.ai` e `huggingface.co`.
- **As entradas de taxa de transferência são comprimentos exatos de token.** Cada modelo de taxa de transferência recebe entradas fixas de 128 tokens e 510 tokens, criadas com seu próprio tokenizador em uma revisão fixa (510 deixa espaço para os tokens especiais dentro de um limite de 512 tokens), confirmadas em `experiments/c2-throughput/inputs.json`. A reclassificação e o NLI estão fora do escopo em C2: o Ollama padrão não oferece nenhum dos dois, portanto, essas células são pré-registradas como `load_refused` com esse motivo.
- **Roteamento remoto opcional.** `interlock route` ignora as linhas remotas, a menos que a solicitação passe por `--allow-remote` ou `--device colab:<card>`. Um vencedor remoto é marcado como `remote: true` com seu `remote_device`. Consulte [Rota](#rota).
- **Importando uma janela remota:** `interlock import --remote-window <path>` mantém cada slot de repetição, e `interlock table --device colab:<card>` (ou `colab:*`) lista as linhas remotas.

A tabela ainda não contém linhas remotas: nenhuma janela remota foi importada. Consulte [CHANGELOG.md](CHANGELOG.md) para a lista completa.

## O que a tabela diz hoje

A tabela de roteamento contém 509 linhas de recibos confirmados, todos medidos no dispositivo. Algumas das respostas que ela já fornece:

- **Um modelo pequeno funciona bem no lado Intel.** O Phi-3.5-mini decodifica a cerca de 20–22 tokens/s na CPU e no iGPU, e a cerca de 15 tokens/s na NPU, em um prompt de 1024 tokens com o raciocínio desativado. Isso deixa o 5090 livre para trabalhos maiores.
- **Um modelo de 8B não pertence ao lado Intel.** No mesmo prompt de 1024 tokens com o raciocínio desativado, Qwen3-8B decodifica a cerca de 10–12 tokens/s na CPU, iGPU e NPU. No 5090, ele decodifica a cerca de 208 tokens/s com o raciocínio desativado e 211 com ele ativado. Solicite 15 tokens/s com o 5090 ocupado, e o roteador se recusa e nomeia o experimento por trás da diferença: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Algumas formas não são executadas.** A NPU rejeitou Qwen3-Embedding em entradas de 512 tokens, tanto um de cada vez quanto em lotes de 8. Essas linhas são mantidas como descobertas de capacidade, nunca classificadas como candidatas lentas.

Essas informações são as taxas de decodificação registradas nos recibos E3 e e3-ref. As notas de execução estão em [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classifica com base na repetição mais lenta, e não na mediana, portanto, suas escolhas são mais cautelosas do que esses resumos.

## O que ele não fará

O interlock não atende a solicitações e não faz proxy de solicitações. Ele não treina modelos. Ele não edita offrig, npu-serve ou rnd. Esses permanecem com seus proprietários. Uma camada de atendimento precisa de seu próprio briefing.

## Dispositivos

Cada dispositivo tem um nome.

| Nome | O que é |
|---|---|
| `NPU` | Intel AI Boost no Core Ultra 9 285K |
| `iGPU` | A GPU Intel. Uma enumeração ao vivo deve indicar o fornecedor `0x8086` e o tipo de dispositivo `INTEGRATED`. O nome completo é registrado, não a decisão |
| `CPU` | A CPU do mesmo processador |
| `5090` | A RTX 5090, acessada apenas por meio do Ollama em `127.0.0.1:11434` |
| `colab:<card>` | Placas CUDA Colab remotas (`A100-40`, `A100-80`, `L4`, `T4`, `G4`), medidas no Linux por meio de NVML ou `nvidia-smi` |

`GPU.1` é o 5090, como OpenVINO o identifica neste sistema. O resolvedor recusa esse índice, mesmo que o nome ou o fornecedor mudem, recusa um `GPU` simples, recusa `AUTO`, `HETERO` e `MULTI`, e recusa uma GPU que não seja do fornecedor `0x8086` e `INTEGRATED`. Uma lista que não contenha nenhuma propriedade do dispositivo ainda exige que o nome completo contenha "Intel". Um dispositivo ausente é um erro. O interlock não se desliga em um dispositivo mais fraco.

As linhas remotas carregam a identidade `remote_device` (`provider: "colab"`, `card`, `vram_mib`, `driver_version`, `cuda_version` e uma string de zona `pool` — nunca um ID de conta). As linhas remotas são mantidas separadas das linhas locais do sistema e nunca são agrupadas no conjunto do sistema. `interlock table --device colab:<card>` (ou `colab:*`) lista as linhas remotas, e `interlock route` considera apenas as placas remotas quando a opção estiver ativada (`--allow-remote` ou `--device colab:<card>`). As formas remotas com falha (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) são registradas como linhas de capacidade com o motivo medido, em vez de serem ignoradas.

A NPU é exclusiva. Uma etapa 5090 é executada apenas dentro de uma permissão do Publisher. O executor registra essa permissão. As etapas que exigem muito da CPU aguardam uma janela que o Publisher anuncia.

## Recibos

Uma linha de roteamento nomeia a tarefa, o modelo (repositório ou tag, revisão ou hash, quantização, licença), o mecanismo e sua versão, o dispositivo, as configurações e a qualidade e a latência que mediu. As métricas de qualidade carregam seu próprio `n`, intervalo e hash do conjunto de dados. A latência mantém o estado "frio" e "quente" separados. Uma única execução é marcada como provisória. A linha aponta para o recibo de origem e o commit do qual foi importada, ou para uma especificação de experimento e um ID de execução.

Uma execução também pode registrar uma linha de capacidade em vez de uma velocidade. `device_lost` e `hang` não carregam informações de tempo. `native_binary_rejected` — o dispositivo compilou o modelo e recusou o binário nativo na execução — mantém apenas o tempo de compilação como evidência e tem escopo para o modelo, dispositivo e forma estática: apenas as execuções com essa chave idêntica são marcadas como não executadas, e as outras formas do dispositivo ainda são executadas. As três são provisórias e nunca contam como velocidades. Uma execução que foi interrompida no meio da programação é mantida em seu recibo, marcada com `aborted: true`, com os slots realmente concluídos; a tabela de roteamento recusa as linhas interrompidas, e `table/gaps.md` lista a execução interrompida. Uma thread de observação protege cada execução de fora da thread principal: uma compilação nativa que é executada por mais de `compile_ceiling_s` (padrão de 900 s) é registrada como uma linha de capacidade `compile_timeout`, e a RAM do sistema em 88% interrompe a execução; ambas primeiro limpam o recibo e, em seguida, encerram o processo com o código de saída 75 (tempo limite de compilação) ou 76 (interrupção da RAM). Cada execução também grava `receipts/<run_id>.run.json` antes de acessar um dispositivo (ignorado pelo Git, não é um recibo), para que uma execução interrompida de fora ainda deixe um registro de que foi iniciada. Para interromper uma execução de forma limpa de fora, crie `receipts/<run_id>.stop`: a thread de observação limpa o recibo e sai com o código 77.

```
interlock check
```

Isso valida cada recibo em `receipts/` e `table/routing.jsonl`. Uma árvore vazia é aceitável. Um caminho de diretório pessoal em um recibo não é.

## Ciclo de vida da execução

Uma execução termina em um de dois estados. Uma execução concluída deixa `receipts/<run-id>.jsonl` e, para E4b, sua série ao lado. Uma execução que é interrompida no meio do caminho grava esse recibo primeiro, com as chamadas que foram concluídas e `aborted: true`, e, em seguida, move o recibo, a série, o log e o marcador de execução para `aborted/<run-id>-<sha>/` com um `NOTE.json` (o ID da execução, o hash do conjunto de ferramentas, o início e o fim, a exceção e os arquivos). `aborted/` é ignorado pelo Git. Não é um resultado: `interlock import` não o lê, e essas execuções nunca chegam à tabela de roteamento.

`receipts/` não deve conter um diretório `*-series/` que não tenha um recibo. A próxima execução se recusará a iniciar enquanto houver um, e nomeará `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` imprime cada ID de execução e um estado: `complete`, `aborted` ou `stray`. `quarantine` move uma execução perdida para `aborted/` com uma nota de que foi definida manualmente. Nenhum dos comandos inicia o Ollama ou um dispositivo. `--quarantine` escolhe o diretório reservado; o padrão é `aborted/` sob a raiz da execução.

Sem `--run-id`, E4b usa `e4b-block` ou `e4b-pilot`, depois `<base>-2`, `<base>-3` e assim por diante. O ID escolhido é a primeira linha da execução e um campo no recibo. Um nome já usado por um recibo, um diretório de série, um log, um cache de compilação ou uma execução em quarentena é recusado.

## Modelo de ameaças

O interlock acessa dispositivos locais e HTTP local, e a rede apenas para baixar um modelo que uma especificação de experimento já nomeou, do editor desse modelo, ou para verificar as fontes C2 quando solicitado.

- Os arquivos de recibos e a tabela de roteamento são arquivos locais.
- As chamadas do Ollama, quando uma execução é realmente iniciada, são direcionadas para `127.0.0.1:11434` (5090) ou `127.0.0.1:11490` (CPU). As tags de modelo em nuvem são rejeitadas.
- As OpenVINO execuções, quando uma execução é realmente iniciada, usam `NPU`, `CPU` ou a Intel iGPU (fornecedores `0x8086` e `INTEGRATED`). Elas nunca selecionam `GPU.1`.
- Uma etapa 5090 sem `--grant` não é iniciada. Uma etapa NPU ou iGPU sem `--npu-session` não é iniciada.
- O caminho 5090 verifica se o tempo de atividade do watchdog da VRAM tem, no máximo, 60 segundos e se nenhum modelo está carregado. O arquivo de tempo de atividade não é aberto. Apenas seu tempo de modificação é lido.
- Nada aqui escuta em uma porta.
- O interlock não envia dados de telemetria.
- O caminho Linux CUDA (`platform: linux-cuda`, para janelas remotas `colab:<card>`) é executado na VM remota, e não no dispositivo. Ele inicia seu próprio Ollama em uma porta de loopback dedicada (`127.0.0.1:11441` por padrão), rejeita as portas compartilhadas `11434` e `11490` e rejeita qualquer URL do Ollama que não seja de loopback: o host deve ser `127.0.0.1:<port>`, `[::1]:<port>` ou `localhost:<port>` quando `localhost` se resolver apenas para loopback, com a porta em dígitos ASCII de 1 a 65535. Ele lê a placa por meio de NVML ou `nvidia-smi`. Ele não baixa modelos e não altera os contadores do Windows.
- `interlock c2-sources --verify-live` é o único comando que acessa a internet para C2. Ele envia solicitações HTTPS somente leitura para `registry.ollama.ai` (manifestos e blocos de configuração) e `huggingface.co` (árvores de arquivos e `tokenizer.json`) e nada mais. Ele não baixa pesos de modelo e não envia credenciais. Sem `--verify-live`, `c2-sources` lê apenas arquivos locais.
- Os resultados de uma janela remota chegam à tabela apenas como um arquivo de recibo que `interlock import --remote-window` lê. O interlock não se conecta ao Colab. Um campo nomeado para uma conta (`account`, `user`, `owner`, seus `_id` formulários, `ggufRepo`, `gguf_repo` ou `gguf_source`) deve conter `sha256:<64-hex>`. As linhas remotas também rejeitam um token de conta com tag, como `account:<id>` ou `user=<id>`, em qualquer valor ou chave de string, e `remote_device` rejeita os IDs de conta, e-mails, caminhos e identificadores de repositório em seus próprios campos. Palavras comuns, como "orientado ao usuário", são aceitas em um valor ou chave.
- O interlock não aluga uma GPU, define um limite de gastos ou abre o banco de dados do dispositivo. Um preço remoto entra apenas como um arquivo de instantâneo com data.

## Layout

`devices` é a lista de permissões. `guards` é a concessão, a sessão, o tempo de atividade e a placa vazia. `receipt` é o esquema e `interlock check`. `importers` reconstrói `table/routing.jsonl` a partir de recibos fixados, e `importers/remote.py` lê um recibo de janela remota. `interlock table` imprime essa tabela, filtrada por tarefa ou dispositivo. `table/gaps.md` é o backlog de experimentos. O conjunto de ferramentas executa uma especificação confirmada; `harness/linux_cuda.py` é o caminho Linux CUDA para placas remotas. `interlock route` lê essa tabela e não altera um dispositivo.

O resumo da construção e as quatro notas de feedback que ele nomeia são armazenadas em `docs/handoff/`. A cópia no armazenamento de memória do estúdio permanece autoritativa.

## Rota

`interlock route` responde de `table/routing.jsonl`. Ele não inicia um dispositivo, lê um livro-razão ou chama o dispositivo.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` corresponde a uma substring do repositório ou tag. Hífens e dois pontos são ignorados, portanto, `Qwen3-8B` corresponde a `qwen3:8b`. Uma substring curta pode corresponder a um repositório não intencional. O modelo correspondente é impresso. `--thinking off` corresponde apenas a `off`. `--thinking on` corresponde a qualquer nível de raciocínio registrado, e a decisão mantém esse nível, portanto, um braço OpenVINO `low` permanece `low`. `low`, `medium` e `high` correspondem exatamente quando a solicitação nomeia um. `--min-tok-s` é o limite mínimo da taxa de decodificação do slot de repetição. Uma rejeição nomeia esse limite `min_tok_s`. `--budget-left-usd` é uma entrada, como `--busy`. Uma linha remota cujo pior caso exceda esse limite é uma concorrente. Sem uma flag, uma sugestão remota é marcada como orçamento não verificado. `--allow-remote` permite que as linhas remotas `colab:<card>` entrem no conjunto de candidatos (desativado por padrão), e `--device` filtra por um dispositivo exato (`colab:L4`, `5090`) ou curinga de provedor (`colab:*`). `--as-of` tem como padrão o dia atual. A própria política não lê o relógio.

Gere classificações com base na menor taxa de decodificação de token/s do slot de repetição. Todas as outras tarefas são classificadas pela maior latência do slot de repetição ou pela maior contagem quando a linha não tem slots de repetição. A margem é a diferença entre esses limites conservadores, nunca uma mediana. Empates deixam o 5090 livre. Uma linha com mais de 30 dias é obsoleta. Uma linha marcada como provisória ou com menos de duas repetições é provisória. Ambas permanecem elegíveis e são sinalizadas.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost`, `oom`, `load_refused`, `timeout`, `ctx_exceeded` e uma linha com `parity_valid` falso (`parity_invalid`) são listadas como excluídas, com esses códigos. Eles não são candidatos. As linhas remotas são filtradas antes do loop de resultados, a menos que a solicitação opte por incluí-las (`--allow-remote` ou `--device colab:...`), portanto, elas nunca aparecem na lista de excluídos ou no motivo de rejeição de uma solicitação apenas para o dispositivo. Um resultado de capacidade é o código exibido quando a linha também é inválida em termos de paridade. Quando nenhuma linha atende às restrições, o comando sai com 1 e nomeia a restrição que esvaziou o conjunto. Ele não faz fallback. `--json` imprime um documento válido em relação a `src/interlock/schema/route-decision.v1.json` (marcando os vencedores remotos com `remote: true` e `remote_device`). `--max-ttft-ms` rejeita, a menos que cada linha comparada registre o mesmo `settings.prefix_caching` (`on`, `off` ou `default`). Não definido significa desconhecido.

Uma sugestão remota inclui um objeto `offrig_plan` quando é suportada por uma oferta RunPod ou `remote: true` e `remote_device` quando é suportada por uma linha `colab:<card>`. O interlock não chama `offrig_plan` ou `offrig_launch` e não gasta. Consulte `docs/offrig-integration.md`.

## Execute um experimento

Cada experimento é pré-registrado: sua especificação em `experiments/<id>/spec.yaml` é confirmada, com a pergunta, a decisão que ele alimenta, os braços, o conjunto de dados, as métricas e as regras de parada, antes de qualquer tempo de dispositivo. `interlock run` rejeita uma especificação que foi editada após sua confirmação. O blob da especificação é verificado em relação a essa confirmação antes da execução (`require_unmodified`) e impresso por `--dry-run`. Ele não é gravado no recibo. Um recibo E4b registra o conjunto de ferramentas `git_sha` e `clean_tree`. Um recibo de preenchimento de cartão registra `experiment_spec` e `run_id`, e, quando o pré-voo foi executado, também registra `git_sha` e `clean_tree`. Qualquer outro recibo de executor compartilhado registra apenas `experiment_spec` e `run_id`. Por prática, uma regra não é alterada após a visualização dos resultados; uma emenda com data vem antes da execução cujos resultados contam, e uma execução definida como inválida é registrada como tal.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

Uma especificação `platform: linux-cuda` (as especificações C2 em `experiments/c2-*/`) é executada em uma VM Colab remota dentro de uma janela reservada, em seu próprio Ollama em uma porta de loopback dedicada.

`--dry-run` imprime o plano e os parâmetros que uma execução real necessitaria, e não inicia nada. Uma execução real precisa dos parâmetros que seus dispositivos exigem: `--grant` para a 5090, `--npu-session` para a NPU ou iGPU, e `--cpu-window` para uma etapa que exige muito da CPU. Ele grava um registro em `receipts/`. `interlock import` (com `--remote-window <path>` opcional para registros JSON da janela remota do Colab) reconstrói `table/` a partir dos registros confirmados e `interlock check` os valida.

## Quando o Ollama trava

`interlock ollama-health` pergunta a `GET /api/version` em `127.0.0.1:11434` (o servidor 5090) e `127.0.0.1:11490` (o servidor apenas CPU). Uma porta está saudável quando essa resposta chega em 5 segundos. O comando imprime uma linha por porta e sai com código 1 se alguma delas estiver inativa.

Uma execução real de `interlock run` reinicia o Ollama apenas em `127.0.0.1:11434`, apenas entre as etapas e apenas após uma etapa falhar com um erro de transporte, um tempo limite que representa uma travamento real ou um erro 5xx enquanto esse servidor está inativo. Inativo significa três verificações de integridade falhadas, com dois segundos de intervalo. Uma solicitação que passou do prazo é considerada um travamento apenas quando a verificação de integridade falha ou `/api/ps` não mostra o modelo progredindo. Um prompt longo cuja versão ainda responde e cujo modelo ainda está listado não é um travamento. Um prompt de 118 mil tokens não é interrompido por ser lento.

A porta `11490` é o indexador de CPU do offrig. A verificação de integridade ainda o relata. Um `11490` inativo interrompe a execução. O interlock nunca interrompe esse processo e nenhum comando cobre sua reinicialização.

Imediatamente antes de qualquer coisa ser interrompida, a reinicialização verifica novamente `--grant`, o watchdog heartbeat e `/api/ps`. Uma listagem que nomeia um modelo de outra sessão é rejeitada. Um `/api/ps` ilegível não impede, por si só, a reinicialização de um servidor que já está inativo. Durante um travamento, a verificação de integridade ainda pode responder. Um `/api/ps` ilegível, então, rejeita a reinicialização, porque o modelo de outra sessão seria descartado sem ser visto.

O processo que é interrompido é aquele que está ouvindo em `11434`. Nada é interrompido pelo nome. O aplicativo na bandeja permanece ativo, a menos que seja esse ouvinte. `ollama serve` é reproduzido a partir da linha inicial registrada para esta execução: o argv, cada variável `OLLAMA_*` e `LLAMA_ARG_*` e o arquivo de log. Stdout e stderr são anexados a esse log. O interlock não faz login. Não há variável `OLLAMA_CUDA_BACKEND`. Se um backend precisar ser forçado, a variável é `OLLAMA_LLM_LIBRARY` e ela deve estar na linha inicial registrada.

O registro é um arquivo JSON nomeado por `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` e `SWITCHYARD_OLLAMA_EXE` ainda funcionam para esta versão e emitem avisos. Os campos são `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` e `backend_line`. Uma execução inicial registrada neste dispositivo é `cmd /c ollama serve` com `OLLAMA_HOST=127.0.0.1:11434` e `OLLAMA_MODELS=E:/AI-Models/Ollama`, registrando em `E:/AI/ollama-cu134/system-11434.log`. A biblioteca carregada é `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. O sha256 do editor desse arquivo inicia `6d3128e946bc9c95`. Uma execução registra a linha inicial que recebeu e reproduz essa mesma. Um registro ausente rejeita a reinicialização antes que qualquer coisa seja interrompida.

No início de uma execução, o conjunto completo de `OLLAMA_*` e `LLAMA_ARG_*` do registro é comparado com o ambiente do processo ativo. Essa comparação é exata. Quando o ambiente do processo não pode ser lido, a linha de log `server config` do servidor é o fallback, e essa comparação também é exata: um registro que é apenas parte dos valores padrão registrados desativa as reinicializações. Uma incompatibilidade significa que esta execução não fará reinicializações. O motivo é escrito no marcador da execução como `ollama_restart_disabled`. Nada é interrompido. `OLLAMA_HOST` corresponde com ou sem um esquema `http://` ou `https://`.

Após o servidor retornar, a execução lê a biblioteca sha256 de `library_path`. A atenção flash, o tipo de cache KV, a linha `load_backend` e o conjunto completo de `OLLAMA_*` e `LLAMA_ARG_*` são lidos dos bytes do log gravados após o início. Os bytes anteriores a esse deslocamento são do servidor anterior, e uma linha lá não corresponde. Se o tamanho do log não puder ser lido, esse deslocamento é desconhecido e a execução é interrompida. Um log ausente está vazio e seu deslocamento é zero. Uma linha `load_backend` que ainda não foi gravada é verificada após o primeiro carregamento do modelo e antes que essa amostra seja mantida. Uma linha ausente ou diferente interrompe a execução. A linha de configuração do servidor do novo segmento deve cobrir todas as variáveis sob as quais a execução foi medida. Os valores padrão que o processo não definiu também podem ser registrados. Um servidor que retorna com uma biblioteca diferente ou sob um ambiente diferente interrompe a execução. A versão do driver CUDA de `nvidia-smi` não é a verificação: uma reinicialização não pode alterá-la. A atenção flash e o tipo de cache KV que o log não ecoam fazem com que a correspondência falhe. Eles não são copiados do registro e considerados uma leitura.

Se o servidor não estiver saudável em 60 segundos, ou se a placa não retornar ao seu estado base, a execução é interrompida e informa que uma pessoa precisa iniciar o Ollama. No máximo, duas reinicializações por execução. Uma reinicialização com falha conta para esse limite.

Cada reinicialização é gravada no marcador da execução e em `settings.ollama_restarts`: quando, qual porta, por quê (`down` ou `hung`), os PIDs antigos e novos, o tempo para ficar saudável e a versão. A etapa que estava em andamento é inconclusiva, nunca uma aprovação. Ela é tentada novamente apenas quando a especificação define `retry` (o padrão é nenhum). Uma tentativa mantida define `settings.retried`. A primeira amostra mantida após uma reinicialização define `settings.cold_after_restart` e é excluída dos percentis de aquecimento. Uma medição feita durante uma reinicialização não é mantida.

`--dry-run` imprime esta política e não entra em contato com nenhuma das portas.

## Desenvolvimento

Python 3.12. `uv sync --frozen` cria o ambiente do projeto a partir do arquivo de bloqueio. Ele não compartilha o ambiente da NPU. OpenVINO 2026.4.1 é o `intel` extra. Os testes não precisam dele.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribuindo

Os bugs são enviados para GitHub Issues. Um relatório de segurança segue [SECURITY.md](SECURITY.md) e não é um problema público. As alterações são enviadas como solicitações de pull para `main`. O padrão e os comandos para executar primeiro estão em [CONTRIBUTING.md](CONTRIBUTING.md).

Partes deste repositório foram escritas com Grok (xAI). Grok, Kimi e Claude são creditados em [CONTRIBUTORS.md](CONTRIBUTORS.md).

Criado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

