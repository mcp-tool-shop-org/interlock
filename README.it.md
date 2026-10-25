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
  <a href="https://pypi.org/project/mcptoolshop-interlock/"><img src="https://img.shields.io/pypi/v/mcptoolshop-interlock" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/interlock"><img src="https://codecov.io/gh/mcp-tool-shop-org/interlock/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/interlock"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/interlock/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/15467"><img src="https://www.bestpractices.dev/projects/15467/badge" alt="OpenSSF Best Practices"></a>
</p>

# interlock: router per dispositivi AI locali

> **Versione 1.0.0: uno strumento di ricerca.** A partire dalla versione 1.0, le versioni successive seguono il versionamento semantico: una modifica che interrompe la CLI, il formato della ricevuta o il formato delle specifiche dell'esperimento comporta una nuova versione principale. interlock è solo a scopo informativo: suggerisce un modello, un motore e un dispositivo e non gestisce mai il traffico. Ogni numero locale è stato misurato su un'unica configurazione (Intel Core Ultra 9 285K con la sua NPU e iGPU, più una RTX 5090) e le righe remote `colab:<card>` sono memorizzate in un pool di dati separato con l'identità completa della scheda (`provider`, `card`, `vram_mib`, `driver_version`, `cuda_version`, `pool`). Considera i suoi percorsi come dati misurati per quella specifica configurazione hardware, non come un benchmark generale.

L'interblocco ferroviario impedisce che un treno venga instradato su un percorso in conflitto. Questo strumento fa lo stesso per i dispositivi: invia ogni attività a un modello, un motore e un dispositivo.

Gli si fornisce un compito (embedding, riordinamento, NLI, verifica, generazione) e i vincoli importanti (qualità, latenza, contesto, livello di ragionamento). Risponde con un modello, un motore, un dispositivo e le impostazioni, e mostra la ricevuta alla base di ogni numero. I plugin AUTO e HETERO di OpenVINO possono selezionare un dispositivo una volta che è stato selezionato il modello. Nessuno degli elementi in questa sequenza seleziona il modello. Questa lacuna è il punto di forza del prodotto.

## Installazione

Da un clone:

```
uv sync
uv run interlock --help
```

Da PyPI:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

Il nome della distribuzione è `mcptoolshop-interlock`. L'importazione e il comando sono `interlock`. Solo Python 3.12 (`requires-python` è `>=3.12,<3.13`). Il CI esegue i test su Ubuntu. Un test di misurazione in tempo reale sulla configurazione è stato scritto per la configurazione Windows su cui è stata eseguita la misurazione di questa tabella. Il percorso CUDA per Linux (novità nella versione 1.0.0) viene eseguito su una VM Colab remota.

Ciò a cui questo programma può accedere è indicato nel [modello di minaccia](#modello-di-minaccia).

Lo schema della ricevuta, la tabella di routing e il framework sono stati creati per primi. `interlock route` legge quella tabella. Non utilizza tempo di elaborazione del dispositivo.

## Cosa c'è di nuovo nella versione 1.0.0

La versione 1.0.0 segue la 0.2.0. Non è stata rilasciata una versione 0.3.0 separata; le sue modifiche sono incluse nella versione 1.0.0.

- **Schede Colab remote come dispositivi.** Le righe `colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4` e `colab:G4` contengono l'identità della scheda (`provider`, `card`, `vram_mib`, versione del driver e di CUDA e una zona `pool`) e sono memorizzate nel proprio pool, separatamente dalle righe della configurazione. Vedi [Dispositivi](#dispositivi).
- **Un percorso di misurazione CUDA per Linux** per queste schede (`interlock run` con una specifica `platform: linux-cuda`), con tre specifiche preregistrate: `experiments/c2-fill-card/`, `experiments/c2-generate/` e `experiments/c2-throughput/`. Le forme non valide (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) vengono conservate come righe di capacità.
- **Ogni modello C2 indica una fonte verificata:** un tag Ollama con il digest del suo manifesto di registro o un file GGUF con il suo repository Hugging Face, commit e sha256. Gemma 4 31B utilizza le build Q4_K_M, Q8_0 e BF16 della libreria Ollama; non ha una build pubblicata Q6_K, quindi quella cella è preregistrata come `load_refused`. `interlock c2-sources <spec>` stampa la fonte e il comando di estrazione di ogni voce e `--verify-live` verifica ogni fonte, digest ed etichetta di quantizzazione rispetto a `registry.ollama.ai` e `huggingface.co`.
- **Gli input di throughput sono lunghezze di token esatte.** Ogni modello di throughput riceve input fissi di 128 token e 510 token creati con il proprio tokenizer a una revisione fissa (510 lascia spazio per i token speciali all'interno di un limite di 512 token), inclusi in `experiments/c2-throughput/inputs.json`. Il riordinamento e l'NLI non sono supportati su C2: Ollama standard non li supporta, quindi queste celle sono preregistrate come `load_refused` con tale motivazione.
- **Routing remoto attivabile.** `interlock route` ignora le righe remote a meno che la richiesta non superi `--allow-remote` o `--device colab:<card>`. Un vincitore remoto è contrassegnato come `remote: true` con il suo `remote_device`. Vedi [Percorso](#percorso).
- **Importazione di una finestra remota:** `interlock import --remote-window <path>` conserva ogni slot di ripetizione e `interlock table --device colab:<card>` (o `colab:*`) elenca le righe remote.

La tabella non contiene ancora righe remote: non è stata importata alcuna finestra remota. Vedi [CHANGELOG.md](CHANGELOG.md) per l'elenco completo.

## Cosa dice la tabella oggi

La tabella di routing contiene 509 righe provenienti da ricevute registrate, tutte misurate sulla configurazione. Alcune delle risposte che fornisce già:

- **Un modello piccolo funziona bene sul lato Intel.** Phi-3.5-mini decodifica a circa 20-22 token/s sulla CPU e sull'iGPU e a circa 15 token/s sulla NPU, con un prompt di 1024 token e il ragionamento disattivato. Questo lascia la 5090 libera per lavori più grandi.
- **Un modello da 8B non è adatto al lato Intel.** Con lo stesso prompt di 1024 token e il ragionamento disattivato, Qwen3-8B decodifica a circa 10-12 token/s sulla CPU, iGPU e sulla NPU. Sulla 5090 decodifica a circa 208 token/s con il ragionamento disattivato e 211 con esso attivato. Se si richiede 15 token/s con la 5090 occupata, il router rifiuta e indica l'esperimento alla base della differenza: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Alcune forme non funzionano affatto.** La NPU ha rifiutato Qwen3-Embedding con input di 512 token, sia uno alla volta che in batch di 8. Queste righe vengono conservate come risultati di capacità, mai classificate come candidati lenti.

Questi valori sono le velocità di decodifica registrate nelle ricevute E3 e e3-ref. Le note sull'esecuzione sono disponibili in [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classifica in base alla ripetizione peggiore, non alla mediana, quindi le sue scelte sono più prudenti rispetto a questi riepiloghi.

## Cosa non farà

interlock non gestisce il traffico e non esegue il proxy delle richieste. Non addestra modelli. Non modifica offrig, npu-serve o rnd. Questi rimangono di proprietà dei rispettivi proprietari. Un livello di servizio richiede una sua breve descrizione.

## Dispositivi

Ogni dispositivo ha un nome.

| Nome | Cos'è |
|---|---|
| `NPU` | Intel AI Boost sulla Core Ultra 9 285K |
| `iGPU` | La GPU Intel. Un'enumerazione in tempo reale deve indicare il fornitore `0x8086` e il tipo di dispositivo `INTEGRATED`. Viene registrato il nome completo, non la decisione. |
| `CPU` | La CPU della stessa unità di elaborazione |
| `5090` | La RTX 5090, accessibile solo tramite Ollama all'indirizzo `127.0.0.1:11434` |
| `colab:<card>` | Schede CUDA Colab remote (`A100-40`, `A100-80`, `L4`, `T4`, `G4`), misurate su Linux tramite NVML o `nvidia-smi` |

`GPU.1` è il 5090, così come OpenVINO lo vede in questa configurazione. Il resolver rifiuta quell'indice anche se il nome o il fornitore cambiano, rifiuta un `GPU` semplice, rifiuta `AUTO`, `HETERO` e `MULTI`, e rifiuta una GPU che non è del fornitore `0x8086` e `INTEGRATED`. Un elenco che non contiene alcuna proprietà del dispositivo richiede comunque che il nome completo contenga "Intel". Un dispositivo mancante è un errore. interlock non si stacca da un dispositivo meno potente.

Le righe remote contengono l'identità `remote_device` (`provider: "colab"`, `card`, `vram_mib`, `driver_version`, `cuda_version` e una stringa di zona `pool`, mai un ID account). Le righe remote sono mantenute separate dalle righe locali e non vengono mai raggruppate nel pool. `interlock table --device colab:<card>` (o `colab:*`) elenca le righe remote e `interlock route` considera solo le schede remote quando l'opzione è abilitata (`--allow-remote` o `--device colab:<card>`). Le configurazioni remote non riuscite (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) vengono registrate come righe di capacità con la relativa motivazione misurata, anziché essere saltate.

L'NPU è esclusiva. Un passaggio 5090 viene eseguito solo all'interno di un'autorizzazione Publisher. Il runner registra tale autorizzazione. Le fasi che richiedono molta CPU attendono una finestra annunciata dal Publisher.

## Ricevute

Una riga di routing indica il nome dell'attività, il modello (repository o tag, revisione o digest, quantizzazione, licenza), il motore e la sua versione, il dispositivo, le impostazioni e la qualità e la latenza misurate. Le metriche di qualità contengono il proprio `n`, l'intervallo e l'hash del set di dati. La latenza distingue tra "cold" e "warm". Una singola esecuzione è contrassegnata come provvisoria. La riga punta alla ricevuta di origine e all'impegno da cui è stata importata, oppure a una specifica di esperimento e a un ID di esecuzione.

Un'esecuzione può anche registrare una riga di capacità anziché una velocità. `device_lost` e `hang` non contengono informazioni sui tempi. `native_binary_rejected` (il dispositivo ha compilato il modello e ha rifiutato il binario nativo all'esecuzione) conserva solo il tempo di compilazione come prova ed è limitata al modello, al dispositivo e alla forma statica: solo le esecuzioni con quella chiave identica sono contrassegnate come non eseguite e le altre forme del dispositivo vengono comunque eseguite. Tutti e tre sono provvisori e non vengono mai conteggiati come velocità. Un'esecuzione interrotta da un abort durante la pianificazione viene conservata nella sua ricevuta contrassegnata con `aborted: true`, con gli slot effettivamente completati; la tabella di routing rifiuta le righe interrotte e `table/gaps.md` elenca l'esecuzione interrotta. Un thread di controllo protegge ogni esecuzione dall'esterno del thread principale: una compilazione nativa che supera `compile_ceiling_s` (predefinito 900 s) viene registrata come riga di capacità `compile_timeout` e la RAM del sistema all'88% interrompe l'esecuzione; entrambi cancellano prima la ricevuta e quindi terminano il processo con il codice di uscita 75 (timeout di compilazione) o 76 (interruzione della RAM). Ogni esecuzione scrive anche `receipts/<run_id>.run.json` prima di accedere a un dispositivo (ignorato da Git, non una ricevuta), quindi un'esecuzione interrotta dall'esterno lascia comunque una traccia del suo avvio. Per interrompere un'esecuzione in modo pulito dall'esterno, creare `receipts/<run_id>.stop`: il thread di controllo cancella la ricevuta ed esce con il codice 77.

```
interlock check
```

Questo convalida ogni ricevuta in `receipts/` e `table/routing.jsonl`. Un albero vuoto va bene. Un percorso della directory home in una ricevuta non è accettabile.

## Ciclo di vita dell'esecuzione

Un'esecuzione termina in uno dei due stati. Un'esecuzione completata lascia `receipts/<run-id>.jsonl` e, per E4b, la sua serie accanto. Un'esecuzione che si interrompe a metà percorso scrive prima quella ricevuta, con le chiamate che sono state completate e `aborted: true`, quindi sposta la ricevuta, la serie, il log e il marcatore di esecuzione in `aborted/<run-id>-<sha>/` con un `NOTE.json` (l'ID di esecuzione, l'hash dell'imbracatura, l'inizio e la fine, l'eccezione e i file). `aborted/` è ignorato da Git. Non è un risultato: `interlock import` non lo legge e tali esecuzioni non raggiungono mai la tabella di routing.

`receipts/` non deve contenere una directory `*-series/` che non abbia una ricevuta. L'esecuzione successiva rifiuta di avviarsi finché ce n'è una, e indica `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` stampa ogni ID di esecuzione e uno stato: `complete`, `aborted` o `stray`. `quarantine` sposta un elemento isolato in `aborted/` con una nota che è stato messo da parte manualmente. Nessuno dei due comandi avvia Ollama o un dispositivo. `--quarantine` sceglie la directory isolata; il valore predefinito è `aborted/` nella directory principale dell'esecuzione.

In assenza di `--run-id`, E4b utilizza `e4b-block` o `e4b-pilot`, quindi `<base>-2`, `<base>-3` e così via. L'ID scelto è la prima riga dell'esecuzione e un campo nella ricevuta. Un nome già utilizzato da una ricevuta, una directory di serie, un log, una cache di compilazione o un'esecuzione in quarantena viene rifiutato.

## Modello di minaccia

interlock accede ai dispositivi locali e all'HTTP locale e alla rete solo per scaricare un modello che una specifica di esperimento ha già indicato, dal publisher di quel modello, o per controllare le fonti C2 quando richiesto.

- Le ricevute e la tabella di routing sono file locali.
- Le chiamate a Ollama, quando viene effettivamente avviata un'esecuzione, vengono indirizzate a `127.0.0.1:11434` (la 5090) o a `127.0.0.1:11490` (CPU). Le etichette dei modelli cloud vengono rifiutate.
- Le OpenVINO esecuzioni, quando viene effettivamente avviata un'esecuzione, utilizzano `NPU`, `CPU` o l'Intel iGPU (fornitore `0x8086` e `INTEGRATED`). Non selezionano mai `GPU.1`.
- Un passaggio 5090 senza `--grant` non viene avviato. Un passaggio NPU o iGPU senza `--npu-session` non viene avviato.
- Il percorso 5090 verifica che il battito cardiaco del watchdog VRAM abbia una frequenza massima di 60 secondi e che non sia caricato alcun modello. Il file del battito cardiaco non viene aperto. Viene letta solo la sua data di modifica.
- Qui non è in ascolto nulla su una porta.
- interlock non invia dati di telemetria.
- Il percorso Linux CUDA (`platform: linux-cuda`, per le finestre remote `colab:<card>`) viene eseguito sulla VM remota, non sul dispositivo. Avvia la propria istanza di Ollama su una porta loopback dedicata (`127.0.0.1:11441` per impostazione predefinita), rifiuta le porte condivise `11434` e `11490` e rifiuta qualsiasi URL di Ollama che non sia loopback: l'host deve essere `127.0.0.1:<port>`, `[::1]:<port>` o `localhost:<port>` quando `localhost` si risolve solo su loopback, con la porta in cifre ASCII da 1 a 65535. Legge la scheda tramite NVML o `nvidia-smi`. Non scarica modelli e non modifica i contatori di Windows.
- `interlock c2-sources --verify-live` è l'unico comando che accede a Internet per C2. Invia richieste HTTPS in sola lettura a `registry.ollama.ai` (manifesti e blocchi di configurazione) e `huggingface.co` (alberi di file e `tokenizer.json`) e nient'altro. Non scarica pesi di modelli e non invia credenziali. Senza `--verify-live`, `c2-sources` legge solo i file locali.
- I risultati di una finestra remota raggiungono la tabella solo come file di ricevuta che `interlock import --remote-window` legge. interlock non si connette a Colab. Un campo denominato per un account (`account`, `user`, `owner`, i loro `_id` moduli, `ggufRepo`, `gguf_repo` o `gguf_source`) deve contenere `sha256:<64-hex>`. Le righe remote rifiutano anche un token di account con etichetta, come `account:<id>` o `user=<id>`, in qualsiasi valore o chiave stringa, e `remote_device` rifiuta gli ID account, le e-mail, i percorsi e gli identificatori del repository nei propri campi. Parole comuni come "user-facing" sono accettate in un valore o in una chiave.
- interlock non noleggia una GPU, non imposta un limite di spesa e non apre il database di offrig. Un prezzo remoto viene inserito solo come file di istantanea datato.

## Layout

`devices` è la lista di autorizzazioni. `guards` è la concessione, la sessione, il battito cardiaco e la scheda vuota. `receipt` è lo schema e `interlock check`. `importers` ricostruisce `table/routing.jsonl` dalle ricevute salvate e `importers/remote.py` legge la ricevuta di una finestra remota. `interlock table` stampa quella tabella, filtrata per attività o dispositivo. `table/gaps.md` è l'elenco di attesa degli esperimenti. L'ambiente di test esegue una specifica definita; `harness/linux_cuda.py` è il percorso Linux CUDA per le schede remote. `interlock route` legge quella tabella e non tocca un dispositivo.

Il riepilogo della build e le quattro note di feedback a cui fa riferimento sono memorizzate in `docs/handoff/`. La copia nel repository di memoria dello studio rimane autorevole.

## Percorso

`interlock route` risponde da `table/routing.jsonl`. Non avvia un dispositivo, non legge un registro e non chiama offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` corrisponde a una sottostringa del repository o dell'etichetta. I trattini e i due punti vengono ignorati, quindi `Qwen3-8B` corrisponde a `qwen3:8b`. Una sottostringa breve può corrispondere a un repository non previsto. Il modello corrispondente viene stampato. `--thinking off` corrisponde solo a `off`. `--thinking on` corrisponde a qualsiasi livello di ragionamento registrato e la decisione mantiene tale livello, quindi un ramo OpenVINO `low` rimane `low`. `low`, `medium` e `high` corrispondono esattamente quando la richiesta ne specifica uno. `--min-tok-s` è il limite minimo per la frequenza di decodifica del ripetitore. Un rifiuto specifica tale limite `min_tok_s`. `--budget-left-usd` è un input, come `--busy`. Una riga remota il cui caso peggiore lo supera è un contendente. Senza un flag, un suggerimento remoto è contrassegnato come budget non verificato. `--allow-remote` consente alle righe remote `colab:<card>` di entrare nel pool di candidati (impostazione predefinita disattivata) e `--device` filtra per un dispositivo esatto (`colab:L4`, `5090`) o un carattere jolly del fornitore (`colab:*`). `--as-of` per impostazione predefinita è oggi. La politica stessa non legge l'orologio.

Genera classifiche in base al valore più basso di tok/s per la decodifica del ripetitore. Tutte le altre attività sono classificate in base alla latenza più alta del ripetitore o alla massima richiesta quando la riga non ha ripetitori. Il margine è il divario tra questi limiti conservativi, mai una mediana. I pareggi lasciano la 5090 libera. Una riga più vecchia di 30 giorni è obsoleta. Una riga contrassegnata come provvisoria o con meno di due ripetizioni è provvisoria. Entrambe rimangono idonee e sono contrassegnate.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost`, `oom`, `load_refused`, `timeout`, `ctx_exceeded` e una riga con `parity_valid` false (`parity_invalid`) sono elencate sotto "escluse", con tali codici. Non sono candidati. Le righe remote vengono filtrate prima del ciclo di risultati, a meno che la richiesta non scelga di includerle (`--allow-remote` o `--device colab:...`), quindi non compaiono mai nell'elenco delle esclusioni o nel motivo del rifiuto di una richiesta che riguarda solo il dispositivo. Un risultato di capacità è il codice visualizzato quando la riga è anche non valida in termini di parità. Quando nessuna riga soddisfa i vincoli, il comando esce con il codice 1 e indica il vincolo che ha svuotato l'insieme. Non esegue il fallback. `--json` stampa un documento valido rispetto a `src/interlock/schema/route-decision.v1.json` (contrassegnando i vincitori remoti con `remote: true` e `remote_device`). `--max-ttft-ms` rifiuta a meno che ogni riga confrontata non registri lo stesso `settings.prefix_caching` (`on`, `off` o `default`). Non impostato significa sconosciuto.

Un suggerimento remoto include un oggetto `offrig_plan` quando è supportato da un'offerta RunPod, o `remote: true` e `remote_device` quando è supportato da una riga `colab:<card>`. interlock non chiama `offrig_plan` o `offrig_launch` e non effettua acquisti. Vedere `docs/offrig-integration.md`.

## Esegui un esperimento

Ogni esperimento è preregistrato: la sua specifica in `experiments/<id>/spec.yaml` è definita, con la domanda, la decisione a cui porta, i rami, il set di dati, le metriche e le regole di arresto, prima di qualsiasi tempo di dispositivo. `interlock run` rifiuta una specifica che è stata modificata dopo il suo commit. Il blob della specifica viene controllato rispetto a tale commit prima dell'esecuzione (`require_unmodified`) e stampato da `--dry-run`. Non viene scritto nella ricevuta. Una ricevuta E4b registra l'ambiente di test `git_sha` e `clean_tree`. Una ricevuta di riempimento della scheda registra `experiment_spec` e `run_id` e, quando è stata eseguita la fase di pre-volo, registra anche `git_sha` e `clean_tree`. Qualsiasi altra ricevuta di esecuzione condivisa registra solo `experiment_spec` e `run_id`. Per prassi, una regola non viene modificata dopo aver visto i risultati; una modifica datata precede l'esecuzione i cui risultati contano e un'esecuzione impostata come nulla viene registrata come tale.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

Una specifica `platform: linux-cuda` (le C2 specifiche in `experiments/c2-*/`) viene eseguita su una VM Colab remota all'interno di una finestra prenotata, contro la sua istanza di Ollama su una porta loopback dedicata.

`--dry-run` stampa la pianificazione e le istruzioni per i controlli che sarebbero necessari per un test reale, ma non avvia nulla. Un test reale richiede i flag specificati dai suoi dispositivi: `--grant` per la 5090, `--npu-session` per l'NPU o iGPU, e `--cpu-window` per una fase che richiede molta potenza di calcolo della CPU. Scrive una singola voce sotto `receipts/`. `interlock import` (con `--remote-window <path>` opzionale per i file JSON delle finestre Colab remote) ricostruisce quindi `table/` dalle voci registrate e `interlock check` le convalida.

## Quando Ollama si blocca

`interlock ollama-health` interroga `GET /api/version` su `127.0.0.1:11434` (il server 5090) e `127.0.0.1:11490` (il server solo CPU). Una porta è funzionante quando questa risposta arriva entro 5 secondi. Il comando stampa una riga per ogni porta ed esce con codice 1 se una delle due non risponde.

A real `interlock run` restarts Ollama only on `127.0.0.1:11434`, only between steps, and only after a step failed with a transport error, a timeout that is a real hang, or a 5xx while that server is down. Down means three failed health checks, two seconds apart. A request that passed its own deadline is a hang only when health then fails, or `/api/ps` does not show the model making progress. A long prompt whose version still answers, and whose model is still listed, is not a hang. A 118k-token prompt is not killed for being slow.

La porta `11490` è l'indicizzatore CPU di offrig. Il controllo di integrità continua a segnalare che è attivo. Un `11490` non funzionante interrompe il test. interlock non interrompe mai questo processo e nessun permesso copre il suo riavvio.

Immediatamente prima che qualcosa venga interrotto, il riavvio verifica nuovamente `--grant`, il controllo del watchdog e `/api/ps`. Un elenco che indica un modello da un'altra sessione viene rifiutato. Un `/api/ps` illeggibile non blocca di per sé il riavvio di un server che è già offline. Durante un blocco, il controllo di integrità può comunque rispondere. Un `/api/ps` illeggibile rifiuta quindi il riavvio, perché il modello di un'altra sessione verrebbe eliminato senza essere visualizzato.

Il processo che viene interrotto è quello in ascolto su `11434`. Nulla viene interrotto per nome. L'app nella barra delle applicazioni rimane attiva a meno che non sia quel processo in ascolto. `ollama serve` viene riprodotto dall'inizio, a partire dalla riga registrata per questo test: gli argomenti della riga di comando, ogni variabile `OLLAMA_*` e `LLAMA_ARG_*` e il file di registro. L'output standard e l'output di errore vengono aggiunti a quel file di registro. interlock non effettua l'accesso. Non esiste una variabile `OLLAMA_CUDA_BACKEND`. Se un backend deve essere forzato, la variabile è `OLLAMA_LLM_LIBRARY` e deve essere inclusa nella riga di avvio registrata.

Il record è un file JSON denominato da `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` e `SWITCHYARD_OLLAMA_EXE` funzionano ancora per questa versione e avvisano. I campi sono `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` e `backend_line`. Un avvio registrato su questo sistema è `cmd /c ollama serve` con `OLLAMA_HOST=127.0.0.1:11434` e `OLLAMA_MODELS=E:/AI-Models/Ollama`, con il registro che viene scritto in `E:/AI/ollama-cu134/system-11434.log`. La libreria caricata è `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. Il sha256 del file del fornitore avvia `6d3128e946bc9c95`. Un test registra la riga di avvio che gli è stata fornita e la riproduce. Un record mancante rifiuta il riavvio prima che qualcosa venga interrotto.

All'inizio di un test, l'insieme completo di `OLLAMA_*` e `LLAMA_ARG_*` del record viene confrontato con l'ambiente del processo attivo. Questo confronto è esatto. Quando l'ambiente del processo non può essere letto, la riga `server config` del registro del server viene utilizzata come fallback e anche questo confronto è esatto: un record che è solo una parte dei valori predefiniti registrati disabilita i riavvii. Una discrepanza significa che questo test non effettuerà riavvii. Il motivo viene scritto sul marcatore del test come `ollama_restart_disabled`. Nulla viene interrotto. `OLLAMA_HOST` corrisponde con o senza uno schema `http://` o `https://`.

Dopo che il server è tornato online, il test legge la libreria sha256 da `library_path`. L'attenzione flash, il tipo di cache KV, la riga `load_backend` e l'insieme completo di `OLLAMA_*` e `LLAMA_ARG_*` vengono letti dai byte del registro scritti dopo l'avvio. I byte precedenti a quell'offset sono il server precedente e una riga lì non corrisponde. Se la dimensione del registro non può essere letta, quell'offset è sconosciuto e il test si interrompe. Un registro mancante è vuoto e il suo offset è zero. Una riga `load_backend` che non è ancora stata scritta viene controllata dopo il primo caricamento del modello e prima che il campione venga mantenuto. Una riga mancante o diversa interrompe il test. La riga di configurazione del server del nuovo segmento deve coprire ogni variabile in base alla quale il test è stato misurato. I valori predefiniti che il processo non ha impostato possono anche essere registrati. Un server che torna online con una libreria diversa o in un ambiente diverso interrompe il test. La versione CUDA del driver da `nvidia-smi` non è il controllo: un riavvio non può modificarla. L'attenzione flash e il tipo di cache KV che il registro non fa eco fanno fallire la corrispondenza. Non vengono copiati in avanti dal record e considerati una lettura.

Se il server non è funzionante entro 60 secondi, o se la scheda non è tornata al suo valore di base, il test si interrompe e indica che una persona deve avviare Ollama. Al massimo due riavvii per test. Un riavvio fallito viene conteggiato rispetto a questo limite.

Ogni riavvio viene scritto sul marcatore del test e su `settings.ollama_restarts`: quando, quale porta, perché (`down` o `hung`), gli ID di processo vecchi e nuovi, il tempo necessario per tornare allo stato di funzionamento e la versione. La fase che era in corso è inconclusiva, non viene mai considerata superata. Viene riprovata solo quando la specifica imposta `retry` (il valore predefinito è nessuno). Un tentativo mantenuto imposta `settings.retried`. Il primo campione mantenuto dopo un riavvio imposta `settings.cold_after_restart` e viene escluso dalle percentuali di riscaldamento. Una misurazione effettuata durante un riavvio non viene mantenuta.

`--dry-run` stampa questa politica e non contatta nessuna delle due porte.

## Sviluppo

Python 3.12. `uv sync --frozen` crea l'ambiente del progetto dal file lock. Non condivide l'ambiente NPU. OpenVINO 2026.4.1 è l'extra `intel`. I test non ne hanno bisogno.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contributi

I bug vengono segnalati in GitHub Issues. Un rapporto sulla sicurezza segue [SECURITY.md](SECURITY.md) e non è un problema pubblico. Le modifiche vengono inviate come richieste di pull a `main`. I requisiti e i comandi da eseguire per primi sono in [CONTRIBUTING.md](CONTRIBUTING.md).

Alcune parti di questo repository sono state scritte con Grok (xAI). Grok, Kimi e Claude sono menzionati in [CONTRIBUTORS.md](CONTRIBUTORS.md).

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

