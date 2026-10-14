<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# interlock : routeur local pour appareils d’IA

> **Prototype, 0.1.x : un instrument de recherche.** L’interface de ligne de commande, le format de reçu et les spécifications de l’expérience peuvent changer sans préavis entre les versions 0.x. Interlock est uniquement à titre indicatif : il suggère un modèle, un moteur et un appareil, et il ne sert jamais à traiter les requêtes. Chaque nombre qui y figure a été mesuré sur une seule configuration (Intel Core Ultra 9 285K avec son NPU et son iGPU, ainsi qu’une RTX 5090). Considérez ses itinéraires comme les données de cette configuration, et non comme une référence générale.

L’enclenchement ferroviaire empêche qu’un train ne soit dirigé vers une voie incompatible. Cet outil fait de même pour les appareils : il envoie chaque tâche à un modèle, un moteur et un appareil.

Vous lui donnez une tâche (intégration, réévaluation, NLI, vérification, génération) et les contraintes importantes (qualité, latence, contexte, niveau de raisonnement). Il répond en indiquant un modèle, un moteur, un appareil et les paramètres, et il affiche le reçu qui justifie chaque valeur. Les plug-ins AUTO et HETERO d’OpenVINO peuvent choisir un appareil une fois que vous avez déjà sélectionné le modèle. Rien dans cette pile ne sélectionne le modèle. Cette lacune est le produit.

## Installation

À partir d’une copie :

```
uv sync
uv run interlock --help
```

À partir de PyPI, à partir de la version 0.1.0 :

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

Le nom de la distribution est `mcptoolshop-interlock`. L’importation et la commande sont `interlock`. Uniquement Python 3.12 (`requires-python` est `>=3.12,<3.13`). Les tests sont exécutés par CI sur Ubuntu. Une exécution de mesure en direct est prévue pour la configuration Windows sur laquelle ce tableau a été mesuré.

Ce à quoi ce programme est autorisé à accéder est décrit dans le [modèle de menace](#threat-model).

Le schéma de reçu, la table de routage et le module de test ont été créés en premier. `interlock route` lit cette table. Il n’utilise pas de temps de traitement de l’appareil.

## Ce que la table indique aujourd’hui

La table de routage contient 509 lignes de reçus validés. Voici quelques-unes des réponses qu’elle fournit déjà :

- **Un petit modèle fonctionne bien sur le processeur Intel.** Phi-3.5-mini effectue un décodage à environ 20 à 22 tokens/s sur le CPU et l’iGPU, et à environ 15 tokens/s sur le NPU, avec une invite de 1 024 tokens et le raisonnement désactivé. Cela laisse le 5090 libre pour des tâches plus importantes.
- **Un modèle de 8 Go ne convient pas au processeur Intel.** Avec la même invite de 1 024 tokens et le raisonnement désactivé, Qwen3-8B effectue un décodage à environ 10 à 12 tokens/s sur le CPU, l’iGPU et le NPU. Sur le 5090, il effectue un décodage à environ 208 tokens/s avec le raisonnement désactivé et à 211 tokens/s avec le raisonnement activé. Si vous demandez 15 tokens/s avec le 5090 en charge, le routeur refuse et indique l’expérience qui explique cette différence : `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Certaines formes ne fonctionnent pas du tout.** Le NPU a rejeté Qwen3-Embedding avec des entrées de 512 tokens, à la fois une par une et par lots de 8. Ces lignes sont conservées en tant que données de capacité, mais ne sont jamais classées comme des candidats lents.

Ces chiffres sont les taux de décodage enregistrés dans les reçus E3 et e3-ref. Les notes d’exécution se trouvent dans [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classe les résultats en fonction de la répétition la plus faible, et non de la médiane, de sorte que ses choix sont plus prudents que ces résumés.

## Ce qu’il ne fera pas

Interlock ne traite pas les requêtes et ne les transmet pas. Il n’entraîne pas les modèles. Il ne modifie pas offrig, npu-serve ou rnd. Ces éléments restent la propriété de leurs propriétaires. Une couche de service a besoin de ses propres spécifications.

## Appareils

Chaque appareil est nommé.

| Nom | Description |
|---|---|
| `NPU` | Intel AI Boost sur le Core Ultra 9 285K |
| `iGPU` | Le GPU Intel. Une énumération en direct doit indiquer le fournisseur `0x8086` et le type d’appareil `INTEGRATED`. Le nom complet est enregistré, et non la décision. |
| `CPU` | Le CPU du même processeur |
| `5090` | Le RTX 5090, accessible uniquement via Ollama à l’adresse `127.0.0.1:11434` |

`GPU.1` est le 5090 tel qu’OpenVINO le voit sur cette configuration. Le résolveur refuse cet index, même si le nom ou le fournisseur change, refuse un simple `GPU`, refuse `AUTO`, `HETERO` et `MULTI`, et refuse un GPU qui n’est pas du fournisseur `0x8086` et `INTEGRATED`. Une liste qui ne contient aucune de ces propriétés d’appareil exige toujours que le nom complet contienne « Intel ». Un appareil manquant est une erreur. Interlock ne passe pas à un appareil moins performant.

Le NPU est exclusif. Une étape du 5090 ne s’exécute que dans le cadre d’une autorisation Publisher. Le programme d’exécution enregistre cette autorisation. Les étapes qui nécessitent beaucoup de CPU attendent une fenêtre annoncée par le Publisher.

## Reçus

Une ligne de routage indique la tâche, le modèle (dépôt ou balise, révision ou hachage, quantification, licence), le moteur et sa version, l’appareil, les paramètres, ainsi que la qualité et la latence mesurées. Les mesures de qualité contiennent leur propre `n`, intervalle et hachage de l’ensemble de données. La latence distingue les valeurs à froid et à chaud. Une seule exécution est marquée comme provisoire. La ligne pointe vers le reçu source et la validation à partir de laquelle elle a été importée, ou vers une spécification d’expérience et un ID d’exécution.

Une exécution peut également enregistrer une ligne de capacité au lieu d’une vitesse. `device_lost` et `hang` ne contiennent pas de données de synchronisation. `native_binary_rejected` (l’appareil a compilé le modèle et a refusé le binaire natif lors de l’exécution) conserve uniquement le temps de compilation comme preuve, et est limité au modèle, à l’appareil et à la forme statique : seules les exécutions avec cette clé identique sont marquées comme non exécutées, et les autres formes de l’appareil continuent de s’exécuter. Les trois sont provisoires et ne sont jamais considérés comme des vitesses. Une exécution qui a été interrompue au milieu de son exécution est conservée dans son reçu et marquée `aborted: true`, avec les emplacements qui ont été réellement exécutés ; la table de routage refuse les lignes interrompues, et `table/gaps.md` répertorie l’exécution interrompue. Un thread de surveillance protège chaque exécution contre les éléments extérieurs au thread principal : une compilation native qui dépasse `compile_ceiling_s` (par défaut 900 s) est enregistrée comme une ligne de capacité `compile_timeout`, et la RAM à 88 % interrompt l’exécution ; les deux vident d’abord le reçu, puis terminent le processus avec le code de sortie 75 (dépassement du délai de compilation) ou 76 (arrêt de la RAM). Chaque exécution écrit également `receipts/<run_id>.run.json` avant de toucher un appareil (ignoré par Git, pas un reçu), de sorte qu’une exécution interrompue de l’extérieur laisse toujours une trace de son démarrage. Pour arrêter une exécution proprement de l’extérieur, créez `receipts/<run_id>.stop` : le programme de surveillance vide le reçu et se termine avec le code 77.

```
interlock check
```

Cela valide chaque reçu sous `receipts/` et `table/routing.jsonl`. Un arbre vide est acceptable. Un chemin d’accès au répertoire personnel dans un reçu ne l’est pas.

## Exécuter un cycle de vie

Une exécution se termine dans l’un des deux états suivants. Une exécution terminée laisse `receipts/<run-id>.jsonl` et, pour E4b, sa série à côté. Une exécution qui s’arrête en cours de route enregistre d’abord ce reçu, avec les appels qui se sont terminés et `aborted: true`, puis déplace le reçu, la série, le journal et le marqueur d’exécution vers `aborted/<run-id>-<sha>/` avec un `NOTE.json` (l’ID de l’exécution, le hachage du harnais, le début et la fin, l’exception et les fichiers). `aborted/` est ignoré par Git. Ce n’est pas un résultat : `interlock import` ne le lit pas, et ces exécutions n’atteignent jamais la table de routage.

`receipts/` ne doit pas contenir de répertoire `*-series/` qui ne contienne aucun reçu. L’exécution suivante refuse de démarrer tant qu’il y en a un, et nomme `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` affiche l’ID de chaque exécution et un état : `complete`, `aborted` ou `stray`. `quarantine` déplace un élément isolé vers `aborted/` avec une note indiquant qu’il a été mis de côté manuellement. Aucune de ces commandes ne démarre Ollama ou un appareil. `--quarantine` choisit le répertoire mis de côté ; par défaut, il s’agit de `aborted/` dans le répertoire racine de l’exécution.

En l’absence de `--run-id`, E4b utilise `e4b-block` ou `e4b-pilot`, puis `<base>-2`, `<base>-3`, et ainsi de suite. L’ID choisi est la première ligne de l’exécution et un champ du reçu. Un nom déjà utilisé par un reçu, un répertoire de série, un journal, une mémoire cache de compilation ou une exécution mise en quarantaine est refusé.

## Modèle de menace

interlock interagit avec les appareils locaux et le protocole HTTP local, et avec le réseau uniquement pour télécharger un modèle qu’une spécification d’expérience a déjà nommé, à partir de l’éditeur de ce modèle.

- Les reçus et la table de routage sont des fichiers locaux.
- Les appels Ollama, lorsqu’une exécution est réellement démarrée, sont dirigés vers `127.0.0.1:11434` (le 5090) ou `127.0.0.1:11490` (CPU). Les balises de modèle cloud sont refusées.
- Les exécutions OpenVINO, lorsqu’une exécution est réellement démarrée, utilisent `NPU`, `CPU` ou le iGPU Intel (fournisseur `0x8086` et `INTEGRATED`). Elles ne sélectionnent jamais `GPU.1`.
- Une étape 5090 sans `--grant` ne démarre pas. Une étape NPU ou iGPU sans `--npu-session` ne démarre pas.
- Le chemin 5090 vérifie que le signal de surveillance VRAM est âgé d’au plus 60 secondes et qu’aucun modèle n’est chargé. Le fichier de signal n’est pas ouvert. Seul son temps de modification est lu.
- Rien ici n’écoute sur un port.
- interlock n’envoie pas de télémétrie.
- interlock ne loue pas de GPU, ne fixe pas de limite de dépenses et n’ouvre pas la base de données d’offrig. Un prix distant n’entre que sous la forme d’un fichier instantané daté.

## Disposition

`devices` est la liste blanche. `guards` est l’autorisation, la session, le signal et la carte vide. `receipt` est le schéma et `interlock check`. `importers` reconstruit `table/routing.jsonl` à partir des reçus épinglés. `interlock table` affiche cette table, filtrée par tâche ou appareil. `table/gaps.md` est la file d’attente des expériences. Le harnais exécute une spécification validée. `interlock route` lit cette table et n’interagit pas avec un appareil.

Le résumé de la construction et les quatre notes de retour qu’il nomme sont instantanés dans `docs/handoff/`. La copie dans le magasin de mémoire du studio reste faisant foi.

## Routage

`interlock route` répond à partir de `table/routing.jsonl`. Il ne démarre pas un appareil, ne lit pas un registre et n’appelle pas offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` correspond à une sous-chaîne du dépôt ou de la balise. Les tirets et les deux-points sont ignorés, de sorte que `Qwen3-8B` correspond à `qwen3:8b`. Une courte sous-chaîne peut correspondre à un dépôt non prévu. Le modèle correspondant est affiché. `--thinking off` correspond uniquement à `off`. `--thinking on` correspond à tout niveau de réflexion enregistré, et la décision conserve ce niveau, de sorte qu’un bras OpenVINO `low` reste `low`. `low`, `medium` et `high` correspondent exactement lorsque la requête en nomme un. `--min-tok-s` est le seuil du taux de décodage minimal par slot de répétition. Un refus nomme ce seuil `min_tok_s`. `--budget-left-usd` est une entrée, comme `--busy`. Une ligne distante dont le pire des cas le dépasse est un concurrent. En l’absence de drapeau, une suggestion distante est marquée comme budget non vérifié. `--as-of` a par défaut la date du jour. La politique elle-même ne lit pas l’horloge.

Générer des classements en fonction du taux de décodage le plus faible par slot de répétition. Toutes les autres tâches sont classées en fonction de la latence de slot de répétition la plus élevée, ou en fonction du temps de revendication maximal lorsque la ligne n’a pas de slots de répétition. La marge est l’écart entre ces limites conservatrices, jamais une médiane. Les égalités laissent le 5090 libre. Une ligne datant de plus de 30 jours est obsolète. Une ligne marquée comme provisoire, ou avec moins de deux répétitions, est provisoire. Les deux restent éligibles et sont signalées.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost` et une ligne avec `parity_valid` à faux (`parity_invalid`) sont répertoriées dans la catégorie « exclues », avec ces codes. Elles ne sont pas des candidats. Un résultat de capacité est le code affiché lorsque la ligne est également invalide en termes de parité. Lorsqu’aucune ligne ne répond aux contraintes, la commande se termine avec le code 1 et nomme la contrainte qui a vidé l’ensemble. Elle ne revient pas en arrière. `--json` affiche un document valide par rapport à `src/interlock/schema/route-decision.v1.json`. `--max-ttft-ms` refuse à moins que chaque ligne comparée n’enregistre le même `settings.prefix_caching` (`on`, `off` ou `default`). Non défini signifie inconnu.

Une suggestion distante inclut un objet `offrig_plan`. interlock n’appelle pas `offrig_plan` ou `offrig_launch`, et n’effectue pas de dépenses. Voir `docs/offrig-integration.md`.

## Exécuter une expérience

Chaque expérience est pré-enregistrée : sa spécification dans `experiments/<id>/spec.yaml` est validée, avec la question, la décision qu’elle fournit, les bras, l’ensemble de données, les métriques et les règles d’arrêt, avant tout temps d’appareil. `interlock run` refuse une spécification qui a été modifiée après sa validation. Le blob de la spécification est vérifié par rapport à cette validation avant l’exécution (`require_unmodified`) et affiché par `--dry-run`. Il n’est pas écrit dans le reçu. Un reçu E4b enregistre le harnais `git_sha` et `clean_tree`. Un reçu de carte de remplissage enregistre `experiment_spec` et `run_id`, et lorsque le pré-vol a été exécuté, il enregistre également `git_sha` et `clean_tree`. Tout autre reçu de runner partagé n’enregistre que `experiment_spec` et `run_id`. Par convention, une règle n’est pas modifiée après que les résultats ont été observés ; une modification datée intervient avant l’exécution dont les résultats sont pris en compte, et une exécution mise de côté comme invalide est enregistrée comme telle.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` affiche le plan et les protections dont une véritable exécution aurait besoin, et ne démarre rien. Une véritable exécution a besoin des drapeaux que ses appareils demandent : `--grant` pour le 5090, `--npu-session` pour le NPU ou le iGPU, et `--cpu-window` pour une étape nécessitant beaucoup de CPU. Elle écrit un reçu dans `receipts/`. `interlock import` reconstruit ensuite `table/` à partir des reçus validés, et `interlock check` les valide.

## Quand Ollama se bloque

`interlock ollama-health` interroge `GET /api/version` sur `127.0.0.1:11434` (le serveur 5090) et `127.0.0.1:11490` (le serveur CPU uniquement). Un port est sain lorsque cette réponse est reçue dans les 5 secondes. La commande affiche une ligne par port et se termine avec le code 1 si l’un ou l’autre est hors service.

Un redémarrage réel de `interlock run` ne relance Ollama que sur `127.0.0.1:11434`, uniquement entre les étapes, et uniquement après qu’une étape a échoué en raison d’une erreur de transport, d’un délai d’attente qui représente un blocage réel, ou d’une erreur 5xx lorsque le serveur est hors service. Hors service signifie trois vérifications de l’état ayant échoué, avec un intervalle de deux secondes. Une requête qui a dépassé son propre délai n’est considérée comme un blocage que lorsque la vérification de l’état échoue ensuite, ou si `/api/ps` ne montre pas que le modèle progresse. Une longue requête dont la version répond toujours et dont le modèle est toujours répertorié n’est pas un blocage. Une requête de 118 000 jetons n’est pas interrompue pour être trop lente.

Le port `11490` est l’indexeur CPU d’offrig. La vérification de l’état continue de le signaler. Un `11490` hors service arrête l’exécution. Interlock n’arrête jamais ce processus, et aucune autorisation ne couvre son redémarrage.

Immédiatement avant que quoi que ce soit ne soit arrêté, le redémarrage vérifie à nouveau `--grant`, le contrôle de l’activité et `/api/ps`. Une liste qui mentionne un modèle provenant d’une autre session est refusée. Un `/api/ps` illisible n’empêche pas, en soi, le redémarrage d’un serveur qui est déjà hors service. Pendant un blocage, la vérification de l’état peut toujours répondre. Un `/api/ps` illisible refuse alors le redémarrage, car le modèle d’une autre session serait supprimé sans être affiché.

Le processus qui est arrêté est celui qui écoute sur `11434`. Rien n’est interrompu par son nom. L’application dans la barre des tâches reste active, sauf s’il s’agit de cet écouteur. `ollama serve` est relancé à partir de la ligne de départ enregistrée pour cette exécution : les arguments de la ligne de commande, chaque variable `OLLAMA_*` et `LLAMA_ARG_*`, et le fichier journal. Stdout et stderr sont ajoutés à ce journal. Interlock ne se connecte pas. Il n’y a pas de variable `OLLAMA_CUDA_BACKEND`. Si un module complémentaire doit être forcé, la variable est `OLLAMA_LLM_LIBRARY`, et elle doit figurer dans la ligne de départ enregistrée.

L’enregistrement est un fichier JSON nommé par `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` et `SWITCHYARD_OLLAMA_EXE` fonctionnent toujours pour cette version et émettent un avertissement. Les champs sont `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` et `backend_line`. Un démarrage enregistré sur cette configuration est `cmd /c ollama serve` avec `OLLAMA_HOST=127.0.0.1:11434` et `OLLAMA_MODELS=E:/AI-Models/Ollama`, et enregistre dans `E:/AI/ollama-cu134/system-11434.log`. La bibliothèque chargée est `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. Le hachage sha256 de ce fichier du développeur commence par `6d3128e946bc9c95`. Une exécution enregistre la ligne de départ qui lui a été fournie et la relance. Un enregistrement manquant refuse le redémarrage avant que quoi que ce soit ne soit arrêté.

Au début d’une exécution, l’ensemble complet des variables `OLLAMA_*` et `LLAMA_ARG_*` de l’enregistrement est comparé à l’environnement du processus en cours. Cette comparaison est exacte. Lorsque l’environnement du processus ne peut pas être lu, la ligne `server config` du journal du serveur est utilisée comme solution de repli, et cette comparaison est également exacte : un enregistrement qui ne fait partie que des valeurs par défaut enregistrées désactive les redémarrages. Une différence signifie que cette exécution n’effectuera aucun redémarrage. La raison est écrite sur le marqueur d’exécution sous la forme `ollama_restart_disabled`. Rien n’est arrêté. `OLLAMA_HOST` correspond avec ou sans schéma `http://` ou `https://`.

Une fois que le serveur est de nouveau opérationnel, l’exécution lit le hachage sha256 de la bibliothèque à partir de `library_path`. L’attention flash, le type de cache KV, la ligne `load_backend` et l’ensemble complet des variables `OLLAMA_*` et `LLAMA_ARG_*` sont lus à partir des octets du journal écrits après le lancement. Les octets avant ce décalage correspondent au serveur précédent, et une ligne qui s’y trouve ne correspond pas. Si la taille du journal ne peut pas être lue, ce décalage est inconnu et l’exécution s’arrête. Un journal manquant est vide, et son décalage est nul. Une ligne `load_backend` qui n’a pas encore été écrite est vérifiée après le premier chargement du modèle, et avant que cet échantillon ne soit conservé. Une ligne manquante ou différente arrête l’exécution. La ligne de configuration du serveur du nouveau segment doit couvrir toutes les variables pour lesquelles l’exécution a été mesurée. Les valeurs par défaut que le processus n’a pas définies peuvent également être enregistrées. Un serveur qui revient avec une bibliothèque différente, ou dans un environnement différent, arrête l’exécution. La version du pilote CUDA à partir de `nvidia-smi` n’est pas la vérification : un redémarrage ne peut pas la modifier. L’attention flash et le type de cache KV que le journal ne renvoie pas entraînent l’échec de la correspondance. Ils ne sont pas copiés à partir de l’enregistrement et appelés une lecture.

Si le serveur n’est pas opérationnel dans les 60 secondes, ou si la carte n’est pas revenue à sa valeur de référence, l’exécution s’arrête et indique qu’une personne doit démarrer Ollama. Au maximum, deux redémarrages par exécution. Un redémarrage ayant échoué est pris en compte dans cette limite.

Chaque redémarrage est écrit sur le marqueur d’exécution et sur `settings.ollama_restarts` : quand, quel port, pourquoi (`down` ou `hung`), les anciens et nouveaux PID, le temps nécessaire pour être opérationnel, et la version. L’étape qui était en cours n’est pas concluante, elle n’est jamais considérée comme réussie. Elle est tentée à nouveau uniquement lorsque la spécification définit `retry` (la valeur par défaut est aucune). Une tentative conservée définit `settings.retried`. Le premier échantillon conservé après un redémarrage définit `settings.cold_after_restart` et est exclu des pourcentiles de réchauffement. Une mesure prise pendant un redémarrage n’est pas conservée.

`--dry-run` affiche cette politique et ne contacte aucun des ports.

## Développement

Python 3.12. `uv sync --frozen` crée l’environnement du projet à partir du fichier lock. Il ne partage pas l’environnement NPU. OpenVINO 2026.4.1 est le module complémentaire `intel`. Les tests n’en ont pas besoin.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribution

Les bogues sont signalés sur GitHub Issues. Un rapport de sécurité suit le document [SECURITY.md](SECURITY.md) et n’est pas un problème public. Les modifications sont soumises sous forme de demandes d’extraction vers `main`. Les exigences et les commandes à exécuter en premier sont disponibles dans le document [CONTRIBUTING.md](CONTRIBUTING.md).

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

