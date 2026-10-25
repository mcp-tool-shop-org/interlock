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

> **Protótipo, 0.3.x: um instrumento de pesquisa.** A CLI, o formato do recibo e as especificações do experimento podem mudar sem aviso entre as versões 0.x. O interlock é apenas uma sugestão: ele sugere um modelo, um motor e um dispositivo, e nunca processa tráfego. Cada número nele foi medido em um único equipamento (Intel Core Ultra 9 285K com sua NPU e iGPU, mais uma RTX 5090). Considere suas rotas como evidências desse equipamento, e não como um benchmark geral.

O intertravamento ferroviário impede que um trem seja direcionado para um caminho conflitante. Esta ferramenta faz o mesmo para dispositivos: ela envia cada tarefa para um modelo, um motor e um dispositivo.

Você fornece uma tarefa (incorporação, reclassificação, NLI, verificação, geração) e as restrições importantes (qualidade, latência, contexto, nível de raciocínio). Ele responde com um modelo, um motor, um dispositivo e as configurações, e mostra o recibo por trás de cada número. Os plugins AUTO e HETERO do OpenVINO podem selecionar um dispositivo depois que você já tiver selecionado o modelo. Nada nessa pilha seleciona o modelo. Essa lacuna é o produto.

## Instalar

A partir de um clone:

```
uv sync
uv run interlock --help
```

A partir do PyPI, do v0.1.0:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

O nome da distribuição é `mcptoolshop-interlock`. O comando de importação é `interlock`. Apenas Python 3.12 (`requires-python` é `>=3.12,<3.13`). O CI executa os testes no Ubuntu. Uma execução de medição ao vivo foi escrita para o equipamento Windows no qual esta tabela foi medida.

O que este programa tem permissão para acessar está no [modelo de ameaças](#modelo-de-ameaças).

O esquema do recibo, a tabela de roteamento e o conjunto de testes foram criados primeiro. `interlock route` lê essa tabela. Ele não usa o tempo do dispositivo.

## O que a tabela diz hoje

A tabela de roteamento contém 509 linhas de recibos confirmados. Algumas das respostas que ela já fornece:

- **Um modelo pequeno funciona bem no lado Intel.** O Phi-3.5-mini decodifica a cerca de 20–22 tokens/s na CPU e no iGPU, e cerca de 15 tokens/s na NPU, com um prompt de 1024 tokens e raciocínio desativado. Isso deixa a 5090 livre para tarefas maiores.
- **Um modelo de 8 GB não pertence ao lado Intel.** No mesmo prompt de 1024 tokens com raciocínio desativado, Qwen3-8B decodifica a cerca de 10–12 tokens/s na CPU, iGPU e NPU. Na 5090, ele decodifica a cerca de 208 tokens/s com o raciocínio desativado e 211 com ele ativado. Se você solicitar 15 tokens/s com a 5090 ocupada, o roteador se recusará e indicará o experimento por trás da lacuna: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Algumas configurações não funcionam.** A NPU rejeitou Qwen3-Embedding com entradas de 512 tokens, tanto um de cada vez quanto em lotes de 8. Essas linhas são mantidas como descobertas de capacidade, nunca classificadas como candidatas lentas.

Essas informações são as taxas de decodificação registradas nos recibos E3 e e3-ref. As notas da execução estão em [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classifica com base na repetição mais lenta, e não na mediana, portanto, suas escolhas são mais cautelosas do que esses resumos.

## O que ele não fará

O interlock não processa tráfego e não faz proxy de solicitações. Ele não treina modelos. Ele não edita offrig, npu-serve ou rnd. Esses permanecem com seus proprietários. Uma camada de serviço precisa de sua própria descrição.

## Dispositivos

Cada dispositivo tem um nome.

| Nome | O que é |
|---|---|
| `NPU` | Intel AI Boost no Core Ultra 9 285K |
| `iGPU` | A GPU Intel. Uma enumeração ao vivo deve indicar o fornecedor `0x8086` e o tipo de dispositivo `INTEGRATED`. O nome completo é registrado, não a decisão. |
| `CPU` | A CPU do mesmo processador |
| `5090` | A RTX 5090, acessada apenas por meio do Ollama em `127.0.0.1:11434` |

`GPU.1` é a 5090 como OpenVINO a vê neste equipamento. O resolvedor se recusa a usar esse índice, mesmo que o nome ou o fornecedor mudem, se recusa a usar um `GPU` simples, se recusa a usar `AUTO`, `HETERO` e `MULTI`, e se recusa a usar uma GPU que não seja do fornecedor `0x8086` e `INTEGRATED`. Uma lista sem nenhuma dessas propriedades do dispositivo ainda exige que o nome completo contenha "Intel". Um dispositivo ausente é um erro. O interlock não usa um dispositivo mais fraco.

A NPU é exclusiva. Uma etapa da 5090 é executada apenas dentro de uma concessão do Publisher. O executor registra essa concessão. As etapas que usam muito a CPU aguardam uma janela que o Publisher anuncia.

## Recibos

Uma linha de roteamento indica a tarefa, o modelo (repositório ou tag, revisão ou hash, quantização, licença), o motor e sua versão, o dispositivo, as configurações e a qualidade e a latência que ele mediu. As métricas de qualidade carregam seu próprio `n`, intervalo e hash do conjunto de dados. A latência mantém o frio e o quente separados. Uma única execução é marcada como provisória. A linha aponta para o recibo de origem e o commit do qual foi importada, ou para uma especificação de experimento e um ID de execução.

Uma execução também pode registrar uma linha de capacidade em vez de uma velocidade. `device_lost` e `hang` não contêm informações de tempo. `native_binary_rejected` — o dispositivo compilou o modelo e se recusou a usar o binário nativo na execução — mantém apenas o tempo de compilação como evidência e tem escopo para o modelo, dispositivo e forma estática: apenas os elementos com essa chave idêntica são marcados como não executados, e as outras formas do dispositivo ainda são executadas. Todos os três são provisórios e nunca contam como velocidades. Um elemento que foi interrompido no meio da programação é mantido em seu recibo, marcado como `aborted: true`, com os slots realmente concluídos; a tabela de roteamento se recusa a usar linhas interrompidas, e `table/gaps.md` lista a execução interrompida. Um thread de observação protege cada execução de fora do thread principal: uma compilação nativa que é executada por mais de `compile_ceiling_s` (padrão de 900 s) é registrada como uma linha de capacidade `compile_timeout`, e a memória RAM do sistema em 88% interrompe a execução; ambos limpam o recibo primeiro e encerram o processo com o código de saída 75 (tempo limite de compilação) ou 76 (interrupção da RAM). Cada execução também grava `receipts/<run_id>.run.json` antes de acessar um dispositivo (ignorado pelo Git, não é um recibo), para que uma execução interrompida de fora ainda deixe um registro de que foi iniciada. Para interromper uma execução de forma limpa de fora, crie `receipts/<run_id>.stop`: o observador limpa o recibo e sai com o código 77.

```
interlock check
```

Isso valida cada recibo em `receipts/` e `table/routing.jsonl`. Uma árvore vazia é aceitável. Um caminho no diretório inicial em um recibo não é.

## Ciclo de vida da execução

Uma execução termina em um de dois estados. Uma execução concluída deixa `receipts/<run-id>.jsonl` e, para E4b, sua série ao lado. Uma execução que é interrompida no meio do processo grava esse recibo primeiro, com as chamadas que foram concluídas e `aborted: true`, e então move o recibo, a série, o registro e o marcador de execução para `aborted/<run-id>-<sha>/` com um `NOTE.json` (o ID da execução, o hash do ambiente, o início e o fim, a exceção e os arquivos). `aborted/` é ignorado pelo Git. Não é um resultado: `interlock import` não o lê, e essas execuções nunca chegam à tabela de roteamento.

`receipts/` não deve conter um diretório `*-series/` que não tenha nenhum recibo. A próxima execução se recusará a iniciar enquanto houver um, e nomeará `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` imprime o ID de cada execução e um estado: `complete`, `aborted` ou `stray`. `quarantine` move um item solto para `aborted/` com uma nota de que ele foi reservado manualmente. Nenhum dos comandos inicia o Ollama ou um dispositivo. `--quarantine` escolhe o diretório reservado; o padrão é `aborted/`, na raiz da execução.

Sem `--run-id`, E4b usa `e4b-block` ou `e4b-pilot`, depois `<base>-2`, `<base>-3` e assim por diante. O ID escolhido é a primeira linha da execução e um campo no recibo. Um nome já usado por um recibo, um diretório de série, um registro, um cache de compilação ou uma execução em quarentena é recusado.

## Modelo de ameaças

o interlock interage com dispositivos locais e HTTP local, e com a rede apenas para baixar um modelo que uma especificação de experimento já nomeou, do editor desse modelo.

- Os recibos e a tabela de roteamento são arquivos locais.
- As chamadas do Ollama, quando uma execução é realmente iniciada, vão para `127.0.0.1:11434` (o 5090) ou `127.0.0.1:11490` (CPU). As tags de modelo em nuvem são recusadas.
- OpenVINO executa, quando uma execução é realmente iniciada, usando `NPU`, `CPU` ou a Intel iGPU (o fornecedor `0x8086` e `INTEGRATED`). Eles nunca selecionam `GPU.1`.
- Uma etapa 5090 sem `--grant` não inicia. Uma etapa NPU ou iGPU sem `--npu-session` não inicia.
- O caminho 5090 verifica se o watchdog VRAM está ativo há no máximo 60 segundos e se nenhum modelo está carregado. O arquivo de monitoramento não é aberto. Apenas seu tempo de modificação é lido.
- Nada aqui escuta em uma porta.
- O interlock não envia telemetria.
- O interlock não aluga uma GPU, define um limite de gastos ou abre o banco de dados do offrig. Um preço remoto entra apenas como um arquivo de instantâneo com data.

## Layout

`devices` é a lista de permissões. `guards` é a concessão, a sessão, o monitoramento e o cartão vazio. `receipt` é o esquema e `interlock check`. `importers` reconstrói `table/routing.jsonl` a partir de recibos fixados. `interlock table` imprime essa tabela, filtrada por tarefa ou dispositivo. `table/gaps.md` é o backlog de experimentos. O ambiente executa uma especificação confirmada. `interlock route` lê essa tabela e não interage com um dispositivo.

O resumo da construção e as quatro notas de feedback que ele nomeia são armazenadas em `docs/handoff/`. A cópia na memória do estúdio permanece como a versão autoritativa.

## Roteamento

`interlock route` responde a partir de `table/routing.jsonl`. Não inicia um dispositivo, lê um livro-razão ou chama o offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` corresponde a uma substring do repositório ou da tag. Hífens e dois pontos são ignorados, então `Qwen3-8B` corresponde a `qwen3:8b`. Uma substring curta pode corresponder a um repositório não intencional. O modelo correspondente é impresso. `--thinking off` corresponde apenas a `off`. `--thinking on` corresponde a qualquer nível de raciocínio registrado, e a decisão mantém esse nível, então um braço OpenVINO `low` permanece `low`. `low`, `medium` e `high` correspondem exatamente quando a solicitação nomeia um. `--min-tok-s` é o limite mínimo da taxa de decodificação de repetição. Uma recusa nomeia esse limite `min_tok_s`. `--budget-left-usd` é uma entrada, como `--busy`. Uma linha remota cujo pior caso excede esse limite é uma concorrente. Sem uma flag, uma sugestão remota é marcada como orçamento não verificado. `--as-of` tem como padrão o dia atual. A própria política não lê o relógio.

Gere classificações com base na menor taxa de decodificação de repetição (tok/s). Todas as outras tarefas são classificadas pela maior latência de repetição ou pela máxima duração da afirmação quando a linha não tem slots de repetição. A margem é a diferença entre esses limites conservadores, nunca uma mediana. Empates deixam o 5090 livre. Uma linha com mais de 30 dias é obsoleta. Uma linha marcada como provisória ou com menos de duas repetições é provisória. Ambas permanecem elegíveis e são sinalizadas.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost` e uma linha com `parity_valid` falso (`parity_invalid`) são listadas em "excluídos", com esses códigos. Eles não são candidatos. Um resultado de capacidade é o código mostrado quando a linha também é inválida em termos de paridade. Quando nenhuma linha atende às restrições, o comando sai com 1 e nomeia a restrição que esvaziou o conjunto. Não há fallback. `--json` imprime um documento válido em relação a `src/interlock/schema/route-decision.v1.json`. `--max-ttft-ms` recusa, a menos que cada linha comparada registre o mesmo `settings.prefix_caching` (`on`, `off` ou `default`). Não definido significa desconhecido.

Uma sugestão remota inclui um objeto `offrig_plan`. O interlock não chama `offrig_plan` ou `offrig_launch`, e não faz gastos. Veja `docs/offrig-integration.md`.

## Execute um experimento

Cada experimento é pré-registrado: sua especificação em `experiments/<id>/spec.yaml` é confirmada, com a pergunta, a decisão que ela alimenta, os braços, o conjunto de dados, as métricas e as regras de parada, antes de qualquer tempo de dispositivo. `interlock run` recusa uma especificação que foi editada após sua confirmação. O blob da especificação é verificado em relação a essa confirmação antes da execução (`require_unmodified`) e impresso por `--dry-run`. Não é gravado no recibo. Um recibo E4b registra o ambiente `git_sha` e `clean_tree`. Um recibo de cartão de preenchimento registra `experiment_spec` e `run_id`, e quando o pré-voo foi executado, também registra `git_sha` e `clean_tree`. Qualquer outro recibo de executor compartilhado registra apenas `experiment_spec` e `run_id`. Por prática, uma regra não é alterada após a visualização dos resultados; uma emenda com data vem antes da execução cujos resultados contam, e uma execução definida como inválida é registrada como tal.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` imprime o plano e as proteções que uma execução real precisaria, e não inicia nada. Uma execução real precisa das flags que seus dispositivos exigem: `--grant` para o 5090, `--npu-session` para a NPU ou iGPU, e `--cpu-window` para uma etapa com uso intensivo de CPU. Ele grava um recibo em `receipts/`. `interlock import` então reconstrói `table/` a partir dos recibos confirmados, e `interlock check` os valida.

## Quando o Ollama trava

`interlock ollama-health` pergunta a `GET /api/version` em `127.0.0.1:11434` (o servidor 5090) e `127.0.0.1:11490` (o servidor somente CPU). Uma porta está saudável quando essa resposta chega em 5 segundos. O comando imprime uma linha por porta e sai com 1 se algum dos servidores estiver inativo.

Um reinício real do `interlock run` reinicia o Ollama apenas em `127.0.0.1:11434`, apenas entre as etapas e apenas após uma etapa ter falhado devido a um erro de transporte, um tempo limite que representa uma interrupção real ou um erro 5xx enquanto o servidor está inativo. Inativo significa três verificações de integridade falhadas, com dois segundos de intervalo. Uma solicitação que ultrapassou seu próprio prazo é considerada uma interrupção apenas quando a verificação de integridade falha ou `/api/ps` não mostra que o modelo está progredindo. Um prompt longo cuja versão ainda responde e cujo modelo ainda está listado não é considerado uma interrupção. Um prompt de 118 mil tokens não é interrompido por ser lento.

A porta `11490` é o indexador de CPU do offrig. A verificação de integridade ainda o relata. Um `11490` inativo interrompe a execução. O interlock nunca interrompe esse processo, e nenhuma permissão cobre um reinício dele.

Imediatamente antes de qualquer coisa ser interrompida, o reinício verifica novamente o `--grant`, o monitoramento do watchdog e `/api/ps`. Uma listagem que nomeia um modelo de outra sessão é rejeitada. Um `/api/ps` ilegível não impede, por si só, o reinício de um servidor que já está inativo. Durante uma interrupção, a verificação de integridade ainda pode responder. Um `/api/ps` ilegível, então, rejeita o reinício, porque o modelo de outra sessão seria descartado sem ser visto.

O processo que é interrompido é aquele que está ouvindo na porta `11434`. Nada é interrompido pelo nome. O aplicativo na bandeja permanece ativo, a menos que seja esse ouvinte. `ollama serve` é reproduzido a partir da linha inicial registrada para esta execução: o argv, cada variável `OLLAMA_*` e `LLAMA_ARG_*` e o arquivo de log. Stdout e stderr são anexados a esse log. O interlock não faz login. Não há variável `OLLAMA_CUDA_BACKEND`. Se um backend precisar ser forçado, a variável é `OLLAMA_LLM_LIBRARY` e deve estar na linha inicial registrada.

O registro é um arquivo JSON nomeado por `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` e `SWITCHYARD_OLLAMA_EXE` ainda funcionam para esta versão e emitem avisos. Os campos são `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` e `backend_line`. Um início de execução registrado neste sistema é `cmd /c ollama serve` com `OLLAMA_HOST=127.0.0.1:11434` e `OLLAMA_MODELS=E:/AI-Models/Ollama`, registrando em `E:/AI/ollama-cu134/system-11434.log`. A biblioteca carregada é `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. O sha256 do arquivo do editor inicia `6d3128e946bc9c95`. Uma execução registra a linha inicial que recebeu e reproduz essa mesma. Um registro ausente rejeita o reinício antes que qualquer coisa seja interrompida.

No início de uma execução, o conjunto completo de `OLLAMA_*` e `LLAMA_ARG_*` do registro é comparado com o ambiente do processo ativo. Essa comparação é exata. Quando o ambiente do processo não pode ser lido, a linha de log `server config` do servidor é o fallback, e essa comparação também é exata: um registro que é apenas parte dos valores padrão registrados desativa os reinícios. Uma incompatibilidade significa que esta execução não fará reinícios. O motivo é escrito no marcador da execução como `ollama_restart_disabled`. Nada é interrompido. `OLLAMA_HOST` corresponde com ou sem um esquema `http://` ou `https://`.

Após o servidor retornar, a execução lê a biblioteca sha256 de `library_path`. A atenção flash, o tipo de cache KV, a linha `load_backend` e o conjunto completo de `OLLAMA_*` e `LLAMA_ARG_*` são lidos dos bytes do log gravados após o início. Os bytes anteriores a esse deslocamento são do servidor anterior, e uma linha ali não corresponde. Se o tamanho do log não puder ser lido, esse deslocamento é desconhecido e a execução é interrompida. Um log ausente está vazio, e seu deslocamento é zero. Uma linha `load_backend` que ainda não foi gravada é verificada após o primeiro carregamento do modelo e antes que essa amostra seja mantida. Uma linha ausente ou diferente interrompe a execução. A linha de configuração do servidor do novo segmento deve abranger todas as variáveis sob as quais a execução foi medida. Os valores padrão que o processo não definiu também podem ser registrados. Um servidor que retorna com uma biblioteca diferente ou sob um ambiente diferente interrompe a execução. A versão do driver CUDA de `nvidia-smi` não é a verificação: um reinício não pode alterá-la. A atenção flash e o tipo de cache KV que o log não ecoam fazem com que a correspondência falhe. Eles não são copiados do registro e considerados uma leitura.

Se o servidor não estiver íntegro em 60 segundos, ou se a placa não retornar ao seu estado base, a execução é interrompida e informa que uma pessoa deve iniciar o Ollama. No máximo, duas tentativas de reinício por execução. Uma tentativa de reinício malsucedida conta para esse limite.

Cada reinício é gravado no marcador da execução e em `settings.ollama_restarts`: quando, qual porta, por quê (`down` ou `hung`), os PIDs antigos e novos, o tempo para ficar íntegro e a versão. A etapa que estava em andamento é inconclusiva, nunca é considerada bem-sucedida. Ela é tentada novamente apenas quando a especificação define `retry` (o padrão é nenhum). Uma nova tentativa mantida define `settings.retried`. A primeira amostra mantida após um reinício define `settings.cold_after_restart` e é excluída dos percentis de aquecimento. Uma medição feita durante um reinício não é mantida.

`--dry-run` imprime esta política e não entra em contato com nenhuma das portas.

## Desenvolvimento

Python 3.12. `uv sync --frozen` cria o ambiente do projeto a partir do arquivo de bloqueio. Não compartilha o ambiente da NPU. OpenVINO 2026.4.1 é o `intel` extra. Os testes não precisam dele.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribuições

Os bugs são enviados para GitHub Issues. Um relatório de segurança segue [SECURITY.md](SECURITY.md) e não é um problema público. As alterações são enviadas como solicitações de pull para `main`. O padrão e os comandos para executar primeiro estão em [CONTRIBUTING.md](CONTRIBUTING.md).

Partes deste repositório foram escritas com Grok (xAI). Grok, Kimi e Claude são creditados em [CONTRIBUTORS.md](CONTRIBUTORS.md).

Criado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

