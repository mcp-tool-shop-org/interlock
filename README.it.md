<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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
  <a href="https://www.bestpractices.dev/projects/15467"><img src="https://www.bestpractices.dev/projects/15467/badge" alt="OpenSSF Best Practices"></a>
</p>

# interlock: router per dispositivi AI locali

> **Prototipo, 0.1.x: uno strumento di ricerca.** La CLI, il formato della ricevuta e le specifiche dell'esperimento possono cambiare senza preavviso tra le versioni 0.x. Interlock è solo a scopo informativo: suggerisce un modello, un motore e un dispositivo, e non gestisce mai il traffico. Ogni valore in esso è stato misurato su un'unica configurazione (Intel Core Ultra 9 285K con la sua NPU e iGPU, più una RTX 5090). Considera i suoi risultati come dati relativi a tale configurazione, non come un benchmark generale.

L'interblocco ferroviario impedisce che un treno venga instradato su un percorso in conflitto. Questo strumento fa lo stesso per i dispositivi: invia ogni attività a un modello, un motore e un dispositivo.

Gli si fornisce un compito (embedding, riordinamento, NLI, verifica, generazione) e i vincoli importanti (qualità, latenza, contesto, livello di ragionamento). Risponde con un modello, un motore, un dispositivo e le impostazioni, e mostra la ricevuta alla base di ogni valore. I plugin AUTO e HETERO di OpenVINO possono selezionare un dispositivo una volta che è stato scelto il modello. Nessun elemento in questa catena seleziona il modello. Questa lacuna è il punto di forza del prodotto.

## Installazione

Da una copia:

```
uv sync
uv run interlock --help
```

Da PyPI, a partire dalla versione 0.1.0:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

Il nome della distribuzione è `mcptoolshop-interlock`. L'importazione e il comando sono `interlock`. Solo Python 3.12 (`requires-python` è `>=3.12,<3.13`). I test vengono eseguiti su Ubuntu. Un esempio di misurazione in tempo reale è disponibile per la configurazione Windows su cui è stata eseguita questa tabella.

Le aree a cui questo programma può accedere sono indicate nel [modello di minaccia](#threat-model).

Lo schema della ricevuta, la tabella di routing e il framework sono stati creati per primi. `interlock route` legge quella tabella. Non utilizza tempo di elaborazione del dispositivo.

## Cosa indica la tabella oggi

La tabella di routing contiene 509 righe provenienti da ricevute registrate. Alcune delle risposte che fornisce già:

- **Un modello piccolo funziona bene sul lato Intel.** Phi-3.5-mini esegue il decoding a circa 20-22 token/s sulla CPU e sull'iGPU, e a circa 15 token/s sulla NPU, con un prompt di 1024 token e il ragionamento disattivato. Questo lascia la 5090 libera per compiti più impegnativi.
- **Un modello da 8 miliardi di parametri non è adatto al lato Intel.** Con lo stesso prompt di 1024 token e il ragionamento disattivato, Qwen3-8B esegue il decoding a circa 10-12 token/s sulla CPU, sull'iGPU e sulla NPU. Sulla 5090, esegue il decoding a circa 208 token/s con il ragionamento disattivato e 211 con il ragionamento attivato. Se si richiedono 15 token/s con la 5090 occupata, il router rifiuta e indica l'esperimento alla base di questa differenza: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Alcune configurazioni non funzionano affatto.** La NPU ha rifiutato Qwen3-Embedding con input di 512 token, sia uno alla volta che in batch di 8. Queste righe vengono conservate come risultati di capacità, ma non vengono mai considerate come candidati lenti.

Questi valori rappresentano le velocità di decoding registrate nelle ricevute E3 ed e3-ref. Le note sull'esecuzione sono disponibili in [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classifica in base al valore peggiore, non alla mediana, quindi le sue scelte sono più prudenti rispetto a questi riepiloghi.

## Cosa non farà

Interlock non gestisce il traffico e non funge da proxy per le richieste. Non addestra modelli. Non modifica offrig, npu-serve o rnd. Questi rimangono di proprietà dei rispettivi sviluppatori. Un livello di servizio richiede le proprie specifiche.

## Dispositivi

Ogni dispositivo ha un nome.

| Nome | Cos'è |
|---|---|
| `NPU` | Intel AI Boost sul Core Ultra 9 285K |
| `iGPU` | La GPU Intel. Un'enumerazione in tempo reale deve indicare il fornitore `0x8086` e il tipo di dispositivo `INTEGRATED`. Viene registrato il nome completo, non la decisione. |
| `CPU` | La CPU dello stesso processore |
| `5090` | La RTX 5090, accessibile solo tramite Ollama all'indirizzo `127.0.0.1:11434` |

`GPU.1` è la 5090 come la vede OpenVINO su questa configurazione. Il resolver rifiuta questo indice anche se il nome o il fornitore cambiano, rifiuta un semplice `GPU`, rifiuta `AUTO`, `HETERO` e `MULTI`, e rifiuta una GPU che non ha il fornitore `0x8086` e `INTEGRATED`. Un elenco che non contiene nessuna di queste proprietà del dispositivo richiede comunque che il nome completo contenga "Intel". Un dispositivo mancante è un errore. Interlock non passa a un dispositivo meno potente.

La NPU è esclusiva. Un passaggio sulla 5090 viene eseguito solo all'interno di un grant del Publisher. Il runner registra questo grant. Le fasi che richiedono molta CPU attendono una finestra annunciata dal Publisher.

## Ricevute

Una riga di routing indica il compito, il modello (repository o tag, revisione o digest, quantizzazione, licenza), il motore e la sua versione, il dispositivo, le impostazioni e la qualità e la latenza misurate. Le metriche di qualità contengono il proprio `n`, intervallo e hash del set di dati. La latenza distingue tra valori a freddo e a caldo. Un'unica esecuzione è contrassegnata come provvisoria. La riga punta alla ricevuta di origine e al commit da cui è stata importata, o a una specifica dell'esperimento e a un ID di esecuzione.

Un'esecuzione può anche registrare una riga di capacità invece di una velocità. `device_lost` e `hang` non contengono dati temporali. `native_binary_rejected` (il dispositivo ha compilato il modello e ha rifiutato il binario nativo all'esecuzione) conserva solo il tempo di compilazione come prova e si applica al modello, al dispositivo e alla forma statica: solo le voci con la stessa chiave vengono contrassegnate come non eseguite e le altre forme del dispositivo vengono comunque eseguite. Tutte e tre sono provvisorie e non vengono mai considerate come velocità. Un'esecuzione interrotta a metà programma viene conservata nella sua ricevuta contrassegnata come `aborted: true`, con gli slot effettivamente completati; la tabella di routing rifiuta le righe interrotte e `table/gaps.md` elenca l'esecuzione interrotta. Un thread di controllo protegge ogni esecuzione da processi esterni al thread principale: una compilazione nativa che supera `compile_ceiling_s` (valore predefinito 900 s) viene registrata come riga di capacità `compile_timeout` e l'utilizzo della RAM al 88% interrompe l'esecuzione; entrambi svuotano prima la ricevuta e terminano il processo con il codice di uscita 75 (timeout di compilazione) o 76 (interruzione della RAM). Ogni esecuzione scrive anche `receipts/<run_id>.run.json` prima di accedere a un dispositivo (ignorato da Git, non è una ricevuta), quindi un'esecuzione interrotta dall'esterno lascia comunque una traccia del suo avvio. Per interrompere un'esecuzione in modo pulito dall'esterno, creare `receipts/<run_id>.stop`: il thread di controllo svuota la ricevuta ed esce con il codice 77.

```
interlock check
```

Questo convalida ogni ricevuta in base a `receipts/` e `table/routing.jsonl`. Un albero vuoto è accettabile. Un percorso nella directory home in una ricevuta non lo è.

## Esecuzione del ciclo di vita

Un'esecuzione termina in uno di due stati. Un'esecuzione completata lascia `receipts/<run-id>.jsonl` e, per E4b, la sua serie accanto. Un'esecuzione che si interrompe a metà scrive prima tale ricevuta, con le chiamate che sono state completate e `aborted: true`, quindi sposta la ricevuta, la serie, il registro e il marcatore di esecuzione in `aborted/<run-id>-<sha>/` con un `NOTE.json` (l'ID dell'esecuzione, l'hash del framework, l'inizio e la fine, l'eccezione e i file). `aborted/` viene ignorato da Git. Non è un risultato: `interlock import` non lo legge e tali esecuzioni non raggiungono mai la tabella di routing.

`receipts/` non deve contenere una directory `*-series/` che non abbia alcuna ricevuta. L'esecuzione successiva si rifiuta di avviarsi finché questa esiste e assegna il nome `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` stampa l'ID di ciascuna esecuzione e uno stato: `complete`, `aborted` o `stray`. `quarantine` sposta un elemento isolato in `aborted/` con una nota che indica che è stato messo da parte manualmente. Nessuno di questi comandi avvia Ollama o un dispositivo. `--quarantine` sceglie la directory messa da parte; il valore predefinito è `aborted/` nella directory principale dell'esecuzione.

In assenza di `--run-id`, E4b utilizza `e4b-block` o `e4b-pilot`, quindi `<base>-2`, `<base>-3` e così via. L'ID scelto è la prima riga dell'esecuzione e un campo nella ricevuta. Un nome già utilizzato da una ricevuta, una directory di serie, un registro, una cache di compilazione o un'esecuzione in quarantena viene rifiutato.

## Modello di minaccia

interlock interagisce con i dispositivi locali e con HTTP locale e con la rete solo per scaricare un modello che una specifica di esperimento ha già nominato, dal publisher di tale modello.

- Le ricevute e la tabella di routing sono file locali.
- Le chiamate a Ollama, quando un'esecuzione viene effettivamente avviata, vengono indirizzate a `127.0.0.1:11434` (la porta 5090) o a `127.0.0.1:11490` (CPU). I tag dei modelli cloud vengono rifiutati.
- Le esecuzioni OpenVINO, quando un'esecuzione viene effettivamente avviata, utilizzano `NPU`, `CPU` o la Intel iGPU (vendor `0x8086` e `INTEGRATED`). Non selezionano mai `GPU.1`.
- Un passaggio 5090 senza `--grant` non viene avviato. Un passaggio NPU o iGPU senza `--npu-session` non viene avviato.
- Il percorso 5090 verifica che il battito cardiaco del watchdog VRAM abbia un'età massima di 60 secondi e che nessun modello sia caricato. Il file del battito cardiaco non viene aperto. Viene letto solo il suo tempo di modifica.
- Nessuno di questi elementi è in ascolto su una porta.
- interlock non invia dati di telemetria.
- interlock non noleggia una GPU, non imposta un limite di spesa e non apre il database di offrig. Un prezzo remoto viene inserito solo come file di snapshot datato.

## Layout

`devices` è la lista di controllo delle autorizzazioni. `guards` è la concessione, la sessione, il battito cardiaco e la scheda vuota. `receipt` è lo schema e `interlock check`. `importers` ricostruisce `table/routing.jsonl` dalle ricevute salvate. `interlock table` stampa tale tabella, filtrata per attività o dispositivo. `table/gaps.md` è l'elenco di attesa degli esperimenti. Il framework esegue una specifica confermata. `interlock route` legge tale tabella e non interagisce con un dispositivo.

Il riepilogo della build e le quattro note di feedback a cui fa riferimento vengono salvate in `docs/handoff/`. La copia nell'archivio di memoria dello studio rimane autorevole.

## Percorso

`interlock route` risponde da `table/routing.jsonl`. Non avvia un dispositivo, non legge un registro e non chiama offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` corrisponde a una sottostringa del repository o del tag. I trattini e i due punti vengono ignorati, quindi `Qwen3-8B` corrisponde a `qwen3:8b`. Una sottostringa breve può corrispondere a un repository non previsto. Il modello corrispondente viene stampato. `--thinking off` corrisponde solo a `off`. `--thinking on` corrisponde a qualsiasi livello di ragionamento registrato e la decisione mantiene tale livello, quindi un braccio OpenVINO `low` rimane `low`. `low`, `medium` e `high` corrispondono esattamente quando la richiesta ne nomina uno. `--min-tok-s` è il limite minimo per la frequenza di decodifica del ripetitore. Un rifiuto indica tale limite `min_tok_s`. `--budget-left-usd` è un input, come `--busy`. Una riga remota il cui caso peggiore supera tale valore è un candidato di riserva. In assenza di un flag, un suggerimento remoto viene contrassegnato come budget non verificato. `--as-of` ha come valore predefinito oggi. La politica stessa non legge l'orologio.

Genera classifiche in base alla frequenza di decodifica del ripetitore più bassa. Tutte le altre attività vengono classificate in base alla latenza del ripetitore più alta o alla massima richiesta quando la riga non ha ripetitori. Il margine è il divario tra questi limiti conservativi, mai una mediana. I pareggi lasciano il 5090 libero. Una riga più vecchia di 30 giorni è obsoleta. Una riga contrassegnata come provvisoria o con meno di due ripetizioni è provvisoria. Entrambe rimangono idonee e vengono contrassegnate.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost` e una riga con `parity_valid` impostato su false (`parity_invalid`) sono elencate sotto "escluse", con tali codici. Non sono candidati. Un risultato di capacità è il codice visualizzato quando la riga è anche parità-invalida. Quando nessuna riga soddisfa i vincoli, il comando esce con il codice 1 e indica il vincolo che ha svuotato l'insieme. Non esegue un fallback. `--json` stampa un documento valido rispetto a `src/interlock/schema/route-decision.v1.json`. `--max-ttft-ms` rifiuta a meno che ogni riga confrontata non registri lo stesso `settings.prefix_caching` (`on`, `off` o `default`). Non impostato significa sconosciuto.

Un suggerimento remoto include un oggetto `offrig_plan`. interlock non chiama `offrig_plan` o `offrig_launch` e non effettua acquisti. Vedere `docs/offrig-integration.md`.

## Esegui un esperimento

Ogni esperimento è preregistrato: la sua specifica in `experiments/<id>/spec.yaml` è confermata, con la domanda, la decisione a cui fornisce input, i bracci, il set di dati, le metriche e le regole di arresto, prima di qualsiasi tempo di dispositivo. `interlock run` rifiuta una specifica che è stata modificata dopo la sua conferma. Il blob della specifica viene controllato rispetto a tale conferma prima dell'esecuzione (`require_unmodified`) e stampato da `--dry-run`. Non viene scritto nella ricevuta. Una ricevuta E4b registra il framework `git_sha` e `clean_tree`. Una ricevuta di riempimento della scheda registra `experiment_spec` e `run_id` e, quando è stata eseguita la fase di pre-volo, registra anche `git_sha` e `clean_tree`. Qualsiasi altra ricevuta di runner condiviso registra solo `experiment_spec` e `run_id`. Per prassi, una regola non viene modificata dopo aver visto i risultati; una modifica datata precede l'esecuzione i cui risultati contano e un'esecuzione messa da parte come non valida viene registrata come tale.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` stampa il piano e le protezioni di cui un'esecuzione reale avrebbe bisogno e non avvia nulla. Un'esecuzione reale necessita dei flag richiesti dai suoi dispositivi: `--grant` per il 5090, `--npu-session` per l'NPU o l'iGPU e `--cpu-window` per una fase con un carico elevato sulla CPU. Scrive una ricevuta in `receipts/`. `interlock import` quindi ricostruisce `table/` dalle ricevute confermate e `interlock check` le convalida.

## Quando Ollama si blocca

`interlock ollama-health` chiede a `GET /api/version` su `127.0.0.1:11434` (il server 5090) e a `127.0.0.1:11490` (il server solo CPU). Una porta è funzionante quando tale risposta arriva entro 5 secondi. Il comando stampa una riga per porta ed esce con il codice 1 se una delle due non è funzionante.

Un vero e proprio `interlock run` riavvia Ollama solo su `127.0.0.1:11434`, solo tra le fasi, e solo dopo che una fase è fallita a causa di un errore di trasmissione, un timeout che indica un blocco reale, o un errore 5xx mentre il server è inattivo. Inattivo significa tre controlli di integrità falliti, con due secondi di intervallo. Una richiesta che ha superato il proprio limite di tempo è considerata un blocco solo quando il controllo di integrità fallisce successivamente, o quando `/api/ps` non mostra che il modello sta facendo progressi. Un prompt lungo la cui versione continua a rispondere e il cui modello è ancora elencato, non è considerato un blocco. Un prompt di 118.000 token non viene interrotto per essere troppo lento.

La porta `11490` è l'indicizzatore CPU di offrig. Il controllo di integrità continua a segnalarlo. Un `11490` inattivo interrompe l'esecuzione. interlock non interrompe mai questo processo e nessuna autorizzazione copre il suo riavvio.

Immediatamente prima che qualcosa venga interrotto, il riavvio ricontrolla `--grant`, il controllo heartbeat e `/api/ps`. Un elenco che indica un modello da un'altra sessione viene rifiutato. Un `/api/ps` illeggibile non blocca, di per sé, il riavvio di un server che è già inattivo. Durante un blocco, il controllo di integrità può comunque rispondere. Un `/api/ps` illeggibile rifiuta quindi il riavvio, perché il modello di un'altra sessione verrebbe eliminato senza essere visualizzato.

Il processo che viene interrotto è quello che ascolta sulla porta `11434`. Nulla viene interrotto per nome. L'app nella barra delle applicazioni rimane attiva a meno che non sia proprio quel processo di ascolto. `ollama serve` viene riprodotto dall'inizio, a partire dalla riga registrata per questa esecuzione: gli argomenti della riga di comando, ogni variabile `OLLAMA_*` e `LLAMA_ARG_*` e il file di registro. L'output standard e l'output di errore vengono aggiunti a quel registro. interlock non effettua l'accesso. Non esiste una variabile `OLLAMA_CUDA_BACKEND`. Se un backend deve essere forzato, la variabile è `OLLAMA_LLM_LIBRARY` e deve essere inclusa nella riga di avvio registrata.

Il record è un file JSON denominato da `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` e `SWITCHYARD_OLLAMA_EXE` funzionano ancora per questa versione e avvisano. I campi sono `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` e `backend_line`. Un avvio registrato su questo sistema è `cmd /c ollama serve` con `OLLAMA_HOST=127.0.0.1:11434` e `OLLAMA_MODELS=E:/AI-Models/Ollama`, con il registro che viene scritto in `E:/AI/ollama-cu134/system-11434.log`. La libreria caricata è `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. L'hash SHA256 del file del publisher inizia con `6d3128e946bc9c95`. Un'esecuzione registra la riga di avvio che le è stata fornita e la riproduce. Un record mancante rifiuta il riavvio prima che qualcosa venga interrotto.

All'inizio di un'esecuzione, l'insieme completo di `OLLAMA_*` e `LLAMA_ARG_*` del record viene confrontato con l'ambiente del processo attivo. Questo confronto è esatto. Quando l'ambiente del processo non può essere letto, la riga `server config` del registro del server viene utilizzata come fallback e anche questo confronto è esatto: un record che è solo una parte dei valori predefiniti registrati disabilita i riavvii. Una discrepanza significa che questa esecuzione non effettuerà alcun riavvio. Il motivo viene scritto sul marcatore dell'esecuzione come `ollama_restart_disabled`. Nulla viene interrotto. `OLLAMA_HOST` corrisponde con o senza uno schema `http://` o `https://`.

Dopo che il server è tornato online, l'esecuzione legge l'hash SHA256 della libreria da `library_path`. L'attenzione flash, il tipo di cache KV, la riga `load_backend` e l'insieme completo di `OLLAMA_*` e `LLAMA_ARG_*` vengono letti dai byte del registro scritti dopo l'avvio. I byte precedenti a quell'offset rappresentano il server precedente e una riga lì non corrisponde. Se la dimensione del registro non può essere letta, quell'offset è sconosciuto e l'esecuzione si interrompe. Un registro mancante è vuoto e il suo offset è zero. Una riga `load_backend` che non è ancora stata scritta viene controllata dopo il primo caricamento del modello e prima che quel campione venga mantenuto. Una riga mancante o diversa interrompe l'esecuzione. La riga di configurazione del server del nuovo segmento deve coprire ogni variabile in base alla quale l'esecuzione è stata misurata. I valori predefiniti che il processo non ha impostato possono anche essere registrati. Un server che torna online con una libreria diversa o in un ambiente diverso, interrompe l'esecuzione. La versione del driver CUDA da `nvidia-smi` non è il controllo: un riavvio non può modificarla. L'attenzione flash e il tipo di cache KV che il registro non riflette, fanno fallire la corrispondenza. Non vengono copiati in avanti dal record e definiti come lettura.

Se il server non è integro entro 60 secondi, o se la scheda non è tornata al suo valore di base, l'esecuzione si interrompe e indica che una persona deve avviare Ollama. Al massimo due riavvii per esecuzione. Un riavvio fallito viene conteggiato rispetto a questo limite.

Ogni riavvio viene scritto sul marcatore dell'esecuzione e su `settings.ollama_restarts`: quando, quale porta, perché (`down` o `hung`), i vecchi e i nuovi PID, il tempo per raggiungere lo stato di integrità e la versione. La fase che era in corso è inconclusiva, mai considerata un successo. Viene riprovata solo quando la specifica imposta `retry` (il valore predefinito è nessuno). Un tentativo mantenuto imposta `settings.retried`. Il primo campione mantenuto dopo un riavvio imposta `settings.cold_after_restart` e viene escluso dai percentili caldi. Una misurazione effettuata durante un riavvio non viene mantenuta.

`--dry-run` stampa questa politica e non contatta nessuna delle due porte.

## Sviluppo

Python 3.12. `uv sync --frozen` crea l'ambiente del progetto dal file lock. Non condivide l'ambiente NPU. OpenVINO 2026.4.1 è l'extra `intel`. I test non ne hanno bisogno.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contributi

I bug vengono segnalati su GitHub Issues. Un rapporto sulla sicurezza segue [SECURITY.md](SECURITY.md) e non è un problema pubblico. Le modifiche vengono inviate come richieste di pull a `main`. I requisiti e i comandi da eseguire per primi sono in [CONTRIBUTING.md](CONTRIBUTING.md).

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

