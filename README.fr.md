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
  <a href="https://pypi.org/project/mcptoolshop-interlock/"><img src="https://img.shields.io/pypi/v/mcptoolshop-interlock" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/interlock"><img src="https://codecov.io/gh/mcp-tool-shop-org/interlock/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/interlock"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/interlock/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/15467"><img src="https://www.bestpractices.dev/projects/15467/badge" alt="OpenSSF Best Practices"></a>
</p>

# interlock : routeur local pour appareils d’IA

> **Version 1.0.1 : un instrument de recherche.** À partir de la version 1.0, les versions suivent une numérotation sémantique : une modification qui perturbe l’interface de ligne de commande, le format de réception ou le format des spécifications de l’expérience entraîne une nouvelle version principale. interlock est uniquement à titre indicatif : il suggère un modèle, un moteur et un appareil, et il ne sert jamais à acheminer le trafic. Chaque nombre local a été mesuré sur un seul appareil (Intel Core Ultra 9 285K avec son NPU et iGPU, ainsi qu’une RTX 5090), et les données `colab:<card>` distantes sont stockées dans un pool de preuves distinct avec l’identification complète de la carte (`provider`, `card`, `vram_mib`, `driver_version`, `cuda_version`, `pool`). Considérez ses itinéraires comme des données mesurées pour ce pool matériel spécifique, et non comme une référence générale.

L’enclenchement ferroviaire empêche qu’un train ne soit dirigé vers une voie incompatible. Cet outil fait de même pour les appareils : il envoie chaque tâche à un modèle, un moteur et un appareil.

Vous lui donnez une tâche (intégration, réévaluation, compréhension du langage naturel, vérification, génération) et les contraintes importantes (qualité, latence, contexte, niveau de raisonnement). Il répond en indiquant un modèle, un moteur, un appareil et les paramètres, et il affiche les données de référence pour chaque nombre. Les plugins AUTO et HETERO de OpenVINO peuvent choisir un appareil une fois que vous avez déjà choisi le modèle. Rien dans cette pile ne choisit le modèle. Cet écart est le produit.

## Installation

À partir d’un clone :

```
uv sync
uv run interlock --help
```

À partir de PyPI :

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

Le nom de la distribution est `mcptoolshop-interlock`. L’importation et la commande sont `interlock`. Uniquement Python 3.12 (`requires-python` est `>=3.12,<3.13`). Les tests sont exécutés par CI sur Ubuntu. Une exécution de mesure en direct sur l’appareil est écrite pour l’appareil Windows sur lequel ce tableau a été mesuré. Le chemin CUDA Linux (nouveau dans la version 1.0.0) s’exécute sur une machine virtuelle Colab distante.

Ce que ce programme est autorisé à utiliser est indiqué dans le [modèle de menace](#modèle-de-menace).

Le schéma de référence, le tableau de routage et le module de test ont été créés en premier. `interlock route` lit ce tableau. Il n’utilise pas de temps de traitement de l’appareil.

## Nouveautés de la version 1.0.0

La version 1.0.0 fait suite à la version 0.2.0. Il n’y a pas eu de version 0.3.0 distincte ; ses modifications sont intégrées à la version 1.0.0.

- **Cartes Colab distantes en tant qu’appareils.** Les lignes `colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4` et `colab:G4` contiennent l’identification de la carte (`provider`, `card`, `vram_mib`, la version du pilote et de CUDA, ainsi qu’une zone `pool`) et sont stockées dans leur propre pool, séparément des lignes de l’appareil. Voir [Appareils](#appareils).
- **Un chemin de mesure CUDA Linux** pour ces cartes (`interlock run` avec une spécification `platform: linux-cuda`), avec trois spécifications préenregistrées : `experiments/c2-fill-card/`, `experiments/c2-generate/` et `experiments/c2-throughput/`. Les formes ayant échoué (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) sont conservées en tant que lignes de capacité.
- **Chaque modèle C2 indique une source vérifiée :** une balise Ollama avec son hachage de manifeste de registre, ou un fichier GGUF avec son référentiel Hugging Face, son commit et sha256. Gemma 4 31B utilise les versions Q4_K_M, Q8_0 et BF16 de la bibliothèque Ollama ; il n’a pas de version publiée Q6_K, de sorte que cette cellule est préenregistrée en tant que `load_refused`. `interlock c2-sources <spec>` affiche la source et la commande d’extraction de chaque entrée, et `--verify-live` vérifie chaque source, hachage et étiquette de quantification par rapport à `registry.ollama.ai` et `huggingface.co`.
- **Les entrées de débit sont des longueurs de jetons exactes.** Chaque modèle de débit reçoit des entrées fixes de 128 jetons et 510 jetons, créées avec son propre tokenizer à une révision fixe (510 laisse de la place pour les jetons spéciaux dans une limite de 512 jetons), et validées dans `experiments/c2-throughput/inputs.json`. La réévaluation et la compréhension du langage naturel ne sont pas prises en charge sur C2 : Ollama standard ne prend en charge aucune de ces fonctions, de sorte que ces cellules sont préenregistrées en tant que `load_refused` avec cette raison.
- **Routage distant facultatif.** `interlock route` ignore les lignes distantes, sauf si la requête passe `--allow-remote` ou `--device colab:<card>`. Un gagnant distant est marqué `remote: true` avec son `remote_device`. Voir [Route](#itinéraire).
- **Importation d’une fenêtre distante :** `interlock import --remote-window <path>` conserve chaque emplacement de répétition, et `interlock table --device colab:<card>` (ou `colab:*`) répertorie les lignes distantes.

Le tableau ne contient pas encore de lignes distantes : aucune fenêtre distante n’a été importée. Voir [CHANGELOG.md](CHANGELOG.md) pour la liste complète.

## Ce que le tableau indique aujourd’hui

Le tableau de routage contient 509 lignes provenant de données de référence validées, toutes mesurées sur l’appareil. Voici quelques-unes des réponses qu’il fournit déjà :

- **Un petit modèle fonctionne bien sur le côté Intel.** Phi-3.5-mini décode à environ 20 à 22 jetons/s sur le CPU et le iGPU, et à environ 15 jetons/s sur le NPU, avec une invite de 1024 jetons et la désactivation du raisonnement. Cela laisse le 5090 libre pour des tâches plus importantes.
- **Un modèle de 8 Go n’a pas sa place sur le côté Intel.** Avec la même invite de 1024 jetons et la désactivation du raisonnement, Qwen3-8B décode à environ 10 à 12 jetons/s sur le CPU, iGPU et le NPU. Sur le 5090, il décode à environ 208 jetons/s avec le raisonnement désactivé et 211 jetons/s avec le raisonnement activé. Si vous demandez 15 jetons/s avec le 5090 occupé, le routeur refuse et indique l’expérience qui explique cet écart : `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Certaines formes ne fonctionnent pas du tout.** Le NPU a rejeté Qwen3-Embedding avec des entrées de 512 jetons, à la fois un par un et par lots de 8. Ces lignes sont conservées en tant que données de capacité, et ne sont jamais classées comme des candidats lents.

Ces chiffres sont les taux de décodage enregistrés dans les données de référence E3 et e3-ref. Les notes d’exécution se trouvent dans [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` classe en fonction de la pire répétition, et non de la médiane, de sorte que ses choix sont plus prudents que ces résumés.

## Ce qu’il ne fera pas

interlock ne sert pas le trafic et ne transmet pas les requêtes. Il ne forme pas les modèles. Il ne modifie pas offrig, npu-serve ou rnd. Ces éléments restent sous la responsabilité de leurs propriétaires. Une couche de service a besoin de ses propres spécifications.

## Appareils

Chaque appareil est nommé.

| Nom | Ce que c’est |
|---|---|
| `NPU` | Intel AI Boost sur le Core Ultra 9 285K |
| `iGPU` | Le GPU Intel. Une énumération en direct doit indiquer le fournisseur `0x8086` et le type d’appareil `INTEGRATED`. Le nom complet est enregistré, et non la décision. |
| `CPU` | Le CPU du même processeur |
| `5090` | Le RTX 5090, accessible uniquement via Ollama à `127.0.0.1:11434` |
| `colab:<card>` | Cartes CUDA Colab distantes (`A100-40`, `A100-80`, `L4`, `T4`, `G4`), mesurées sur Linux via NVML ou `nvidia-smi` |

`GPU.1` est le 5090 tel que OpenVINO l’affiche sur cette configuration. Le résolveur refuse cet index, même si le nom ou le fournisseur change, refuse un `GPU` simple, refuse `AUTO`, `HETERO` et `MULTI`, et refuse une carte graphique qui n’est pas du fournisseur `0x8086` et `INTEGRATED`. Une liste qui ne contient aucune propriété de périphérique nécessite toujours que le nom complet contienne « Intel ». Un périphérique manquant est une erreur. interlock ne se détache pas pour se fixer sur un autre périphérique moins performant.

Les lignes distantes contiennent l’identifiant `remote_device` (`provider: "colab"`, `card`, `vram_mib`, `driver_version`, `cuda_version` et une chaîne `pool` représentant une zone ; jamais un identifiant de compte). Les lignes distantes sont conservées séparément des lignes locales et ne sont jamais regroupées dans le pool de la configuration. `interlock table --device colab:<card>` (ou `colab:*`) liste les lignes distantes, et `interlock route` ne prend en compte que les cartes distantes lorsque l’option est activée (`--allow-remote` ou `--device colab:<card>`). Les formes distantes ayant échoué (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) sont enregistrées en tant que lignes de capacité avec leur raison mesurée, plutôt que d’être ignorées.

Le NPU est exclusif. Une étape 5090 ne s’exécute que dans le cadre d’une autorisation Publisher. Le processus enregistre cette autorisation. Les étapes gourmandes en CPU attendent une fenêtre annoncée par le Publisher.

## Reçus

Une ligne de routage indique la tâche, le modèle (dépôt ou balise, révision ou hachage, quantification, licence), le moteur et sa version, le périphérique, les paramètres, ainsi que la qualité et la latence mesurées. Les mesures de qualité contiennent leur propre `n`, intervalle et hachage de l’ensemble de données. La latence distingue les valeurs à froid et à chaud. Une seule exécution est marquée comme provisoire. La ligne pointe vers le reçu source et le commit à partir duquel elle a été importée, ou vers une spécification d’expérience et un identifiant d’exécution.

Une exécution peut également enregistrer une ligne de capacité au lieu d’une vitesse. `device_lost` et `hang` ne contiennent aucune information de synchronisation. `native_binary_rejected` (le périphérique a compilé le modèle et a refusé le binaire natif lors de l’exécution) conserve uniquement le temps de compilation comme preuve et est limité au modèle, au périphérique et à la forme statique : seules les exécutions avec cette clé identique sont marquées comme non exécutées, et les autres formes du périphérique continuent de s’exécuter. Les trois sont provisoires et ne sont jamais comptabilisées comme des vitesses. Une exécution qu’un arrêt a interrompue au milieu de son exécution est conservée dans son reçu, marquée `aborted: true`, avec les emplacements qui ont été réellement complétés ; la table de routage refuse les lignes interrompues, et `table/gaps.md` liste l’exécution interrompue. Un thread de surveillance protège chaque exécution depuis l’extérieur du thread principal : une compilation native qui dépasse `compile_ceiling_s` (par défaut 900 s) est enregistrée comme une ligne de capacité `compile_timeout`, et une utilisation de la RAM système de 88 % interrompt l’exécution ; les deux opérations vident d’abord le reçu, puis terminent le processus avec le code de sortie 75 (dépassement du délai de compilation) ou 76 (arrêt de la RAM). Chaque exécution écrit également `receipts/<run_id>.run.json` avant de toucher un périphérique (ignoré par Git, ce n’est pas un reçu), de sorte qu’une exécution interrompue depuis l’extérieur laisse toujours une trace de son démarrage. Pour arrêter une exécution proprement depuis l’extérieur, créez `receipts/<run_id>.stop` : le thread de surveillance vide le reçu et se termine avec le code 77.

```
interlock check
```

Cela valide chaque reçu sous `receipts/` et `table/routing.jsonl`. Un arbre vide est acceptable. Un chemin d’accès au répertoire personnel dans un reçu ne l’est pas.

## Cycle de vie d’une exécution

Une exécution se termine dans l’un des deux états suivants. Une exécution terminée laisse `receipts/<run-id>.jsonl` et, pour E4b, sa série à côté. Une exécution qui s’arrête à mi-parcours écrit d’abord ce reçu, avec les appels qui ont été terminés et `aborted: true`, puis déplace le reçu, la série, le journal et le marqueur d’exécution vers `aborted/<run-id>-<sha>/` avec un `NOTE.json` (l’identifiant de l’exécution, le hachage du harnais, le début et la fin, l’exception et les fichiers). `aborted/` est ignoré par Git. Ce n’est pas un résultat : `interlock import` ne le lit pas, et ces exécutions n’atteignent jamais la table de routage.

`receipts/` ne doit pas contenir de répertoire `*-series/` qui ne contient pas de reçu. L’exécution suivante refuse de démarrer tant qu’il y en a un, et indique `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` affiche chaque identifiant d’exécution et un état : `complete`, `aborted` ou `stray`. `quarantine` déplace une exécution isolée vers `aborted/` avec une note indiquant qu’elle a été mise de côté manuellement. Aucune de ces commandes ne démarre Ollama ou un périphérique. `--quarantine` choisit le répertoire mis de côté ; par défaut, il s’agit de `aborted/` sous le répertoire de l’exécution.

En l’absence de `--run-id`, E4b utilise `e4b-block` ou `e4b-pilot`, puis `<base>-2`, `<base>-3`, et ainsi de suite. L’identifiant choisi est la première ligne de l’exécution et un champ du reçu. Un nom déjà utilisé par un reçu, un répertoire de série, un journal, un cache de compilation ou une exécution mise en quarantaine est refusé.

## Modèle de menace

interlock touche les périphériques locaux et le protocole HTTP local, et le réseau uniquement pour télécharger un modèle qu’une spécification d’expérience a déjà nommé, à partir du fournisseur de ce modèle, ou pour vérifier les sources C2 lorsqu’on le demande.

- Les fichiers de reçus et la table de routage sont des fichiers locaux.
- Les appels à Ollama, lorsqu'une exécution est réellement lancée, se font vers `127.0.0.1:11434` (le 5090) ou `127.0.0.1:11490` (CPU). Les balises de modèles cloud sont refusées.
- OpenVINO exécutions, lorsqu'une exécution est réellement lancée, utilisent `NPU`, `CPU` ou l'Intel iGPU (fournisseur `0x8086` et `INTEGRATED`). Elles ne sélectionnent jamais `GPU.1`.
- Une étape 5090 sans `--grant` ne démarre pas. Une étape NPU ou iGPU sans `--npu-session` ne démarre pas.
- Le chemin 5090 vérifie que le signal de surveillance VRAM est âgé d'au plus 60 secondes et qu'aucun modèle n'est chargé. Le fichier de signal n'est pas ouvert. Seul son temps de modification est lu.
- Rien ici n'écoute sur un port.
- interlock n'envoie pas de télémétrie.
- Le chemin Linux CUDA (`platform: linux-cuda`, pour les fenêtres distantes `colab:<card>`) s'exécute sur la VM distante, et non sur le serveur. Il démarre son propre Ollama sur un port de boucle dédié (`127.0.0.1:11441` par défaut), refuse les ports partagés `11434` et `11490`, et refuse toute URL Ollama qui n'est pas une boucle : l'hôte doit être `127.0.0.1:<port>`, `[::1]:<port>` ou `localhost:<port>` lorsque `localhost` ne se résout qu'à une boucle, avec le port en chiffres ASCII de 1 à 65535. Il lit la carte via NVML ou `nvidia-smi`. Il ne télécharge pas de modèles et ne touche pas aux compteurs Windows.
- `interlock c2-sources --verify-live` est la seule commande qui accède à Internet pour C2. Elle envoie des requêtes HTTPS en lecture seule à `registry.ollama.ai` (manifestes et blocs de configuration) et `huggingface.co` (arborescences de fichiers et `tokenizer.json`), et rien d'autre. Elle ne télécharge aucun poids de modèle et n'envoie aucune information d'identification. Sans `--verify-live`, `c2-sources` ne lit que les fichiers locaux.
- Les résultats d'une fenêtre distante atteignent la table uniquement sous la forme d'un fichier de reçu que `interlock import --remote-window` lit. interlock ne se connecte pas à Colab. Un champ nommé pour un compte (`account`, `user`, `owner`, leurs `_id` formulaires, `ggufRepo`, `gguf_repo` ou `gguf_source`) doit contenir `sha256:<64-hex>`. Les lignes distantes refusent également un jeton de compte balisé, tel que `account:<id>` ou `user=<id>`, dans toute valeur ou clé de chaîne, et `remote_device` refuse les identifiants de compte, les adresses e-mail, les chemins et les identifiants de dépôt dans ses propres champs. Les mots courants tels que « orienté utilisateur » sont acceptés dans une valeur ou une clé.
- interlock ne loue pas de GPU, ne fixe pas de limite de dépenses et n'ouvre pas la base de données d'offrig. Un prix distant n'est pris en compte que sous la forme d'un fichier instantané daté.

## Disposition

`devices` est la liste d'autorisation. `guards` est l'autorisation, la session, le signal et la carte vide. `receipt` est le schéma et `interlock check`. `importers` reconstruit `table/routing.jsonl` à partir des reçus épinglés, et `importers/remote.py` lit le reçu d'une fenêtre distante. `interlock table` affiche cette table, filtrée par tâche ou par appareil. `table/gaps.md` est le backlog d'expériences. Le harnais exécute une spécification validée ; `harness/linux_cuda.py` est le chemin Linux CUDA pour les cartes distantes. `interlock route` lit cette table et ne touche pas à un appareil.

Le résumé de la tâche et les quatre notes de retour qu'il mentionne sont instantanés dans `docs/handoff/`. La copie dans le magasin de mémoire du studio reste la version faisant autorité.

## Itinéraire

`interlock route` répond à partir de `table/routing.jsonl`. Il ne démarre pas d'appareil, ne lit pas de registre et n'appelle pas offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` correspond à une sous-chaîne du dépôt ou de la balise. Les tirets et les deux-points sont ignorés, de sorte que `Qwen3-8B` correspond à `qwen3:8b`. Une courte sous-chaîne peut correspondre à un dépôt non prévu. Le modèle correspondant est affiché. `--thinking off` ne correspond qu'à `off`. `--thinking on` correspond à tout niveau de réflexion enregistré, et la décision conserve ce niveau, de sorte qu'une branche OpenVINO `low` reste `low`. `low`, `medium` et `high` correspondent exactement lorsque la requête en nomme un. `--min-tok-s` est la limite minimale du taux de décodage de la plage de répétition. Un refus indique cette limite `min_tok_s`. `--budget-left-usd` est une entrée, comme `--busy`. Une ligne distante dont le pire des cas dépasse cette limite est une candidate de secours. Sans indicateur, une suggestion distante est marquée comme non vérifiée en termes de budget. `--allow-remote` autorise les lignes distantes `colab:<card>` dans le pool de candidats (par défaut, désactivé), et `--device` filtre par un appareil exact (`colab:L4`, `5090`) ou un caractère générique de fournisseur (`colab:*`). `--as-of` a par défaut la date du jour. La politique elle-même ne lit pas l'heure.

Générer des classements en fonction du taux de décodage de la plage de répétition le plus faible. Toutes les autres tâches sont classées en fonction de la latence de la plage de répétition la plus élevée, ou en fonction de la valeur maximale de l'intervalle de temps lorsque la ligne n'a pas de plages de répétition. La marge est l'écart entre ces limites conservatrices, jamais une médiane. Les égalités laissent le 5090 libre. Une ligne datant de plus de 30 jours est obsolète. Une ligne marquée comme provisoire, ou avec moins de deux répétitions, est provisoire. Les deux restent éligibles et sont signalées.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost`, `oom`, `load_refused`, `timeout`, `ctx_exceeded` et une ligne avec `parity_valid` faux (`parity_invalid`) sont répertoriées sous la catégorie exclues, avec ces codes. Elles ne sont pas des candidats. Les lignes distantes sont filtrées avant la boucle de résultats, sauf si la requête y adhère (`--allow-remote` ou `--device colab:...`), de sorte qu'elles n'apparaissent jamais dans la liste des exclusions ou la raison du refus d'une requête qui ne concerne que le serveur. Un résultat de capacité est le code affiché lorsque la ligne est également invalide en termes de parité. Lorsqu'aucune ligne ne répond aux contraintes, la commande se termine avec le code 1 et indique la contrainte qui a vidé l'ensemble. Elle ne revient pas en arrière. `--json` affiche un document valide par rapport à `src/interlock/schema/route-decision.v1.json` (en marquant les gagnants distants avec `remote: true` et `remote_device`). `--max-ttft-ms` refuse sauf si chaque ligne comparée enregistre la même valeur `settings.prefix_caching` (`on`, `off` ou `default`). Non défini signifie inconnu.

Une suggestion distante inclut un objet `offrig_plan` lorsqu'elle est soutenue par une offre RunPod, ou `remote: true` et `remote_device` lorsqu'elle est soutenue par une ligne `colab:<card>`. interlock n'appelle pas `offrig_plan` ou `offrig_launch`, et n'effectue pas de dépenses. Voir `docs/offrig-integration.md`.

## Exécuter une expérience

Chaque expérience est pré-enregistrée : sa spécification, dans `experiments/<id>/spec.yaml`, est validée, avec la question, la décision à laquelle elle conduit, les branches, l'ensemble de données, les métriques et les règles d'arrêt, avant tout temps d'appareil. `interlock run` refuse une spécification qui a été modifiée après sa validation. Le bloc de la spécification est vérifié par rapport à cette validation avant l'exécution (`require_unmodified`) et affiché par `--dry-run`. Il n'est pas écrit dans le reçu. Un reçu E4b enregistre le harnais `git_sha` et `clean_tree`. Un reçu de remplissage de carte enregistre `experiment_spec` et `run_id`, et lorsque le pré-vol a été exécuté, il enregistre également `git_sha` et `clean_tree`. Tout autre reçu de runner partagé n'enregistre que `experiment_spec` et `run_id`. Par convention, une règle n'est pas modifiée après que les résultats ont été observés ; une modification datée intervient avant l'exécution dont les résultats sont pris en compte, et une exécution mise de côté comme invalide est enregistrée comme telle.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

Une spécification `platform: linux-cuda` (les spécifications C2 sous `experiments/c2-*/`) s'exécute sur une VM Colab distante dans une fenêtre réservée, en utilisant son propre Ollama sur un port de boucle dédié.

`--dry-run` affiche le plan et les paramètres nécessaires pour une exécution réelle, et ne lance rien. Une exécution réelle nécessite les indicateurs spécifiés par ses périphériques : `--grant` pour le 5090, `--npu-session` pour le NPU ou iGPU, et `--cpu-window` pour une étape nécessitant beaucoup de ressources CPU. Il écrit un enregistrement sous `receipts/`. `interlock import` (avec `--remote-window <path>` facultatif pour les enregistrements JSON de la fenêtre Colab distante) reconstruit ensuite `table/` à partir des enregistrements validés, et `interlock check` les valide.

## Quand Ollama se bloque

`interlock ollama-health` interroge `GET /api/version` sur `127.0.0.1:11434` (le serveur 5090) et `127.0.0.1:11490` (le serveur CPU uniquement). Un port est considéré comme fonctionnel lorsque cette réponse est reçue dans les 5 secondes. La commande affiche une ligne par port et quitte avec le code 1 si l’un des deux est hors service.

Une exécution réelle de `interlock run` redémarre Ollama uniquement sur `127.0.0.1:11434`, uniquement entre les étapes, et uniquement après qu’une étape a échoué en raison d’une erreur de transport, d’un délai d’attente qui indique un blocage réel, ou d’une erreur 5xx alors que ce serveur est hors service. Hors service signifie trois vérifications de l’état ayant échoué, avec un intervalle de deux secondes. Une requête qui a dépassé son propre délai n’est considérée comme un blocage que lorsque la vérification de l’état échoue ensuite, ou que `/api/ps` n’indique pas que le modèle progresse. Une requête longue dont la version répond toujours et dont le modèle est toujours répertorié n’est pas un blocage. Une requête de 118 000 jetons n’est pas interrompue pour être trop lente.

Le port `11490` est l’indexeur CPU d’offrig. La vérification de l’état continue de l’indiquer. Un `11490` hors service arrête l’exécution. interlock n’arrête jamais ce processus, et aucun paramètre ne couvre son redémarrage.

Immédiatement avant que quoi que ce soit ne soit arrêté, le redémarrage vérifie à nouveau `--grant`, le contrôle de l’activité (watchdog heartbeat), et `/api/ps`. Une liste qui mentionne un modèle provenant d’une autre session est refusée. Un `/api/ps` illisible n’empêche pas, en soi, le redémarrage d’un serveur qui est déjà hors service. Pendant un blocage, la vérification de l’état peut toujours répondre. Un `/api/ps` illisible refuse alors le redémarrage, car le modèle d’une autre session serait supprimé sans être affiché.

Le processus qui est arrêté est celui qui écoute sur `11434`. Rien n’est interrompu par son nom. L’application dans la barre des tâches reste active, sauf si c’est ce processus d’écoute. `ollama serve` est relancé à partir de la ligne de départ enregistrée pour cette exécution : les arguments de la ligne de commande (argv), chaque variable `OLLAMA_*` et `LLAMA_ARG_*`, et le fichier journal. La sortie standard (stdout) et la sortie d’erreur (stderr) sont ajoutées à ce fichier journal. interlock ne se connecte pas. Il n’y a pas de variable `OLLAMA_CUDA_BACKEND`. Si un module d’arrière-plan doit être forcé, la variable est `OLLAMA_LLM_LIBRARY`, et elle doit figurer dans la ligne de départ enregistrée.

L’enregistrement est un fichier JSON nommé par `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` et `SWITCHYARD_OLLAMA_EXE` fonctionnent toujours pour cette version et émettent un avertissement. Les champs sont `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` et `backend_line`. Un enregistrement de départ sur cette configuration est `cmd /c ollama serve` avec `OLLAMA_HOST=127.0.0.1:11434` et `OLLAMA_MODELS=E:/AI-Models/Ollama`, et enregistre dans `E:/AI/ollama-cu134/system-11434.log`. La bibliothèque chargée est `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. Le sha256 du fichier du développeur lance `6d3128e946bc9c95`. Une exécution enregistre la ligne de départ qui lui a été fournie et la relance. Un enregistrement manquant refuse le redémarrage avant que quoi que ce soit ne soit arrêté.

Au début d’une exécution, l’ensemble complet `OLLAMA_*` et `LLAMA_ARG_*` de l’enregistrement est comparé à l’environnement du processus en cours. Cette comparaison est exacte. Lorsque l’environnement du processus ne peut pas être lu, la ligne `server config` du journal du serveur est utilisée comme solution de repli, et cette comparaison est également exacte : un enregistrement qui ne fait partie que des valeurs par défaut enregistrées désactive les redémarrages. Une incompatibilité signifie que cette exécution ne lancera aucun redémarrage. La raison est écrite sur le marqueur d’exécution sous la forme `ollama_restart_disabled`. Rien n’est arrêté. `OLLAMA_HOST` correspond avec ou sans un schéma `http://` ou `https://`.

Une fois que le serveur est de nouveau opérationnel, l’exécution lit la bibliothèque sha256 à partir de `library_path`. L’attention flash, le type de cache KV, la ligne `load_backend` et l’ensemble complet `OLLAMA_*` et `LLAMA_ARG_*` sont lus à partir des octets du journal écrits après le lancement. Les octets avant ce décalage correspondent au serveur précédent, et une ligne qui s’y trouve ne correspond pas. Si la taille du journal ne peut pas être lue, ce décalage est inconnu et l’exécution s’arrête. Un journal manquant est vide, et son décalage est nul. Une ligne `load_backend` qui n’a pas encore été écrite est vérifiée après le premier chargement du modèle, et avant que cet échantillon ne soit conservé. Une ligne manquante ou différente arrête l’exécution. La ligne de configuration du serveur du nouveau segment doit couvrir toutes les variables pour lesquelles l’exécution a été mesurée. Les valeurs par défaut que le processus n’a pas définies peuvent également être enregistrées. Un serveur qui revient avec une bibliothèque différente, ou dans un environnement différent, arrête l’exécution. La version CUDA du pilote à partir de `nvidia-smi` n’est pas la vérification : un redémarrage ne peut pas la modifier. L’attention flash et le type de cache KV que le journal n’affiche pas entraînent l’échec de la correspondance. Ils ne sont pas copiés à partir de l’enregistrement et utilisés comme une lecture.

Si le serveur n’est pas opérationnel dans les 60 secondes, ou si la carte n’est pas revenue à sa valeur de référence, l’exécution s’arrête et indique qu’une personne doit démarrer Ollama. Au maximum, deux redémarrages par exécution. Un redémarrage ayant échoué est pris en compte dans cette limite.

Chaque redémarrage est écrit sur le marqueur d’exécution et sur `settings.ollama_restarts` : quand, quel port, pourquoi (`down` ou `hung`), les anciens et nouveaux PID, le temps nécessaire pour être opérationnel, et la version. L’étape qui était en cours est inconclusive, elle n’est jamais considérée comme réussie. Elle est tentée à nouveau uniquement lorsque la spécification définit `retry` (la valeur par défaut est aucune). Une tentative conservée définit `settings.retried`. Le premier échantillon conservé après un redémarrage définit `settings.cold_after_restart` et est exclu des pourcentiles de réchauffement. Une mesure prise pendant un redémarrage n’est pas conservée.

`--dry-run` affiche cette politique et ne contacte aucun des deux ports.

## Développement

Python 3.12. `uv sync --frozen` crée l’environnement du projet à partir du fichier lock. Il ne partage pas l’environnement NPU. OpenVINO 2026.4.1 est le `intel` supplémentaire. Les tests n’en ont pas besoin.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribution

Les bogues sont signalés dans GitHub Issues. Un rapport de sécurité suit [SECURITY.md](SECURITY.md) et n’est pas un problème public. Les modifications sont soumises sous forme de demandes d’extraction (pull requests) vers `main`. Les exigences et les commandes à exécuter en premier sont disponibles dans [CONTRIBUTING.md](CONTRIBUTING.md).

Certaines parties de ce dépôt ont été écrites avec Grok (xAI). Grok, Kimi et Claude sont mentionnés dans [CONTRIBUTORS.md](CONTRIBUTORS.md).

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

