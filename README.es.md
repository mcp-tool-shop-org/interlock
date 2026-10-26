<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# interlock: enrutador de dispositivos de IA local

> **Versión 1.0.9: un instrumento de investigación.** A partir de la versión 1.0, las versiones siguen la versión semántica: un cambio que interrumpa la CLI, el formato de recibo o el formato de especificación del experimento viene con una nueva versión principal. interlock es solo una recomendación: sugiere un modelo, un motor y un dispositivo, y nunca sirve tráfico. Cada número local se midió en un único equipo (Intel Core Ultra 9 285K con su NPU y iGPU, más una RTX 5090), y las filas `colab:<card>` remotas se encuentran en su propio conjunto de pruebas independiente con una identidad de tarjeta completa (`provider`, `card`, `vram_mib`, `driver_version`, `cuda_version`, `pool`). Considere sus rutas como evidencia medida para ese conjunto de hardware específico, no como un punto de referencia general. **Corrección en la versión 1.0.6:** el tiempo hasta el primer token y el rendimiento de concurrencia superior a 1 que generaba la escala de Linux CUDA registrados en la versión 1.0.5 y anteriores no eran medidas válidas (cada solicitud cronometrada se servía desde la caché de solicitudes de Ollama y las solicitudes concurrentes se ponían en cola a través de una sola ranura); interlock ya no los utiliza y las filas de la escala de la versión 3 los reemplazan.

El enclavamiento ferroviario evita que un tren se dirija a una vía conflictiva. Esta herramienta hace lo mismo para los dispositivos: envía cada tarea a un modelo, un motor y un dispositivo.

Le proporciona una tarea (incrustación, reordenamiento, NLI, verificación, generación) y las restricciones que importan (calidad, latencia, contexto, nivel de razonamiento). Responde con un modelo, un motor, un dispositivo y la configuración, y muestra el recibo detrás de cada número. Los complementos AUTO y HETERO de OpenVINO pueden seleccionar un dispositivo una vez que ya ha seleccionado el modelo. Nada en esa pila selecciona el modelo. Esa diferencia es el producto.

## Instalar

Desde un clon:

```
uv sync
uv run interlock --help
```

Desde PyPI:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

El nombre de la distribución es `mcptoolshop-interlock`. La importación y el comando son `interlock`. Python 3.13, 3.14 y 3.15 (`requires-python` es `>=3.13`; 3.13 es la versión que ejecuta la imagen predeterminada de Colab). CI ejecuta las pruebas en Ubuntu en las tres versiones, siendo 3.15 la versión principal. En 3.15, PyYAML se instala desde el código fuente hasta que publica los paquetes cp315 (genera un paquete puro de Python cuando libyaml no está presente). El paquete adicional `intel` requiere Python 3.13 o 3.14: OpenVINO 2026.4.1 aún no tiene paquetes cp315, por lo que en 3.15 el paquete adicional instala `transformers` pero no `openvino*` (un marcador de entorno); el núcleo funciona en 3.15. Se ha creado una ejecución de medición en vivo en el equipo para el equipo de Windows en el que se midió esta tabla. La ruta de Linux CUDA (nueva en la versión 1.0.0) se ejecuta en una máquina virtual remota de Colab.

Lo que este programa puede acceder está en el [modelo de amenazas](#modelo-de-amenazas).

El esquema de recibo, la tabla de enrutamiento y el conjunto de pruebas se crearon primero. `interlock route` lee esa tabla. No utiliza tiempo de dispositivo.

## Novedades en la versión 1.0.0

La versión 1.0.0 sigue a la 0.2.0. No hubo una versión 0.3.0 independiente; sus cambios se incluyen en la 1.0.0.

- **Tarjetas Colab remotas como dispositivos.** Las filas `colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4` y `colab:G4` contienen la identidad de la tarjeta (`provider`, `card`, `vram_mib`, versión del controlador y CUDA, y una zona `pool`) y se encuentran en su propio conjunto, aparte de las filas del equipo. Consulte [Dispositivos](#dispositivos).
- **Una ruta de medición CUDA para Linux** para esas tarjetas (`interlock run` con una `platform: linux-cuda` especificación), con tres especificaciones predefinidas: `experiments/c2-fill-card/`, `experiments/c2-generate/` y `experiments/c2-throughput/`. Las formas que fallaron (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) se conservan como filas de capacidad. También se conserva una celda que no se midió (`generation_aborted`, una generación de Ollama que se abortó, o `unmeasured`, un error no reconocido o una instrucción que no se pudo completar), pero no indica nada sobre lo que la tarjeta puede hacer. Cada fila de la escala de generación registra `settings.ladder_version` y la longitud de la instrucción que el modelo realmente evaluó.
- **Cada modelo C2 nombra una fuente verificada:** una etiqueta de Ollama con su resumen del manifiesto del registro, o un archivo GGUF con su repositorio de Hugging Face, confirmación y sha256. Gemma 4 31B utiliza las compilaciones Q4_K_M, Q8_0 y BF16 de la biblioteca de Ollama; no tiene una compilación publicada Q6_K, por lo que esa celda se pre-registra como `load_refused`. `interlock c2-sources <spec>` imprime la fuente y el comando de extracción de cada entrada, y `--verify-live` verifica cada fuente, resumen y etiqueta de cuantificación con `registry.ollama.ai` y `huggingface.co`.
- **Las entradas de rendimiento son longitudes exactas de tokens.** Cada modelo de rendimiento recibe entradas fijas de 128 tokens y 510 tokens, creadas con su propio tokenizador en una revisión fijada (510 deja espacio para los tokens especiales dentro de un límite de 512 tokens), confirmadas en `experiments/c2-throughput/inputs.json`. El reordenamiento y NLI están fuera del alcance en C2: Ollama estándar no ofrece ninguno de los dos, por lo que esas celdas se pre-registran como `load_refused` con esa razón.
- **Enrutamiento remoto opcional.** `interlock route` ignora las filas remotas a menos que la solicitud pase `--allow-remote` o `--device colab:<card>`. Se marca un ganador remoto con `remote: true` y su `remote_device`. Consulte [Ruta](#ruta).
- **Importación de una ventana remota:** `interlock import --remote-window <path>` conserva cada ranura de repetición, y `interlock table --device colab:<card>` (o `colab:*`) enumera las filas remotas.

La tabla aún no contiene filas remotas: no se ha importado ninguna ventana remota. Consulte [CHANGELOG.md](CHANGELOG.md) para obtener la lista completa.

## Lo que dice la tabla hoy

La tabla de enrutamiento contiene 509 filas de recibos confirmados, todos medidos en el equipo. Algunas de las respuestas que ya proporciona:

- **Un modelo pequeño funciona bien en el lado de Intel.** Phi-3.5-mini se decodifica a aproximadamente 20-22 tokens/s en la CPU y la iGPU, y a aproximadamente 15 tokens/s en la NPU, con una solicitud de 1024 tokens y el razonamiento desactivado. Esto deja la 5090 libre para tareas más grandes.
- **Un modelo de 8B no pertenece al lado de Intel.** Con la misma solicitud de 1024 tokens y el razonamiento desactivado, Qwen3-8B se decodifica a aproximadamente 10-12 tokens/s en la CPU, iGPU y la NPU. En la 5090 se decodifica a aproximadamente 208 tokens/s con el razonamiento desactivado y 211 con el razonamiento activado. Si se solicita 15 tokens/s con la 5090 ocupada, el enrutador se niega y nombra el experimento detrás de la diferencia: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Algunas formas no se ejecutan en absoluto.** La NPU rechazó Qwen3-Embedding con entradas de 512 tokens, tanto una a la vez como en lotes de 8. Estas filas se conservan como hallazgos de capacidad, y nunca se clasifican como candidatos lentos.

Estas cifras son las tasas de decodificación registradas en los recibos E3 y e3-ref. Las notas de la ejecución se encuentran en [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` clasifica según la peor repetición, no la mediana, por lo que sus elecciones son más cautelosas que estos resúmenes.

## Lo que no hará

interlock no sirve tráfico ni actúa como proxy para las solicitudes. No entrena modelos. No edita offrig, npu-serve o rnd. Estos permanecen con sus propietarios. Una capa de servicio necesita su propio resumen.

## Dispositivos

Cada dispositivo tiene un nombre.

| Nombre | Qué es |
|---|---|
| `NPU` | Intel AI Boost en el Core Ultra 9 285K |
| `iGPU` | La GPU de Intel. Una enumeración en vivo debe indicar el proveedor `0x8086` y el tipo de dispositivo `INTEGRATED`. Se registra el nombre completo, no la decisión. |
| `CPU` | La CPU del mismo procesador |
| `5090` | La RTX 5090, a la que solo se accede a través de Ollama en `127.0.0.1:11434` |
| `colab:<card>` | Tarjetas CUDA de Colab remotas (`A100-40`, `A100-80`, `L4`, `T4`, `G4`), medidas en Linux a través de NVML o `nvidia-smi` |

`GPU.1` es el 5090 tal como lo ve OpenVINO en esta configuración. El resolvedor rechaza ese índice, incluso si el nombre o el proveedor cambian, rechaza un `GPU` sin más, rechaza `AUTO`, `HETERO` y `MULTI`, y rechaza una GPU que no sea del proveedor `0x8086` y `INTEGRATED`. Una lista que no contenga ninguna propiedad del dispositivo aún requiere que el nombre completo contenga "Intel". Un dispositivo faltante es un error. interlock no se desprende de uno más débil.

Las filas remotas contienen la identidad `remote_device` (`provider: "colab"`, `card`, `vram_mib`, `driver_version`, `cuda_version` y una cadena de zona `pool`; nunca un ID de cuenta). Las filas remotas se mantienen separadas de las filas locales y nunca se agrupan en el grupo de filas. `interlock table --device colab:<card>` (o `colab:*`) enumera las filas remotas, y `interlock route` considera solo las tarjetas remotas cuando se activa la opción (`--allow-remote` o `--device colab:<card>`). Las formas remotas fallidas (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) se registran como filas de capacidad con la razón medida en lugar de omitirse. `interlock route` genera filas de escalera solo dentro de la `ladder_version` más reciente, y sus notas indican cuántas filas más antiguas no comparó.

La NPU es exclusiva. Un paso 5090 se ejecuta solo dentro de una concesión de Publisher. El ejecutor registra esa concesión. Las etapas que requieren mucha CPU esperan una ventana que el Publisher anuncia.

## Recibos

Una fila de enrutamiento nombra la tarea, el modelo (repositorio o etiqueta, revisión o resumen, cuantificación, licencia), el motor y su versión, el dispositivo, la configuración y la calidad y la latencia que midió. Las métricas de calidad contienen su propio `n`, intervalo y hash del conjunto de datos. La latencia mantiene separados los valores "fríos" y "calientes". Una sola ejecución se marca como provisional. La fila apunta al recibo de origen y al commit del que se importó, o a una especificación de experimento y un ID de ejecución.

Una ejecución también puede registrar una fila de capacidad en lugar de una velocidad. `device_lost` y `hang` no contienen información de tiempo. `native_binary_rejected` (el dispositivo compiló el modelo y rechazó el binario nativo en la ejecución) conserva solo el tiempo de compilación como evidencia y tiene un alcance limitado al modelo, dispositivo y forma estática: solo las ejecuciones con esa clave idéntica se marcan como no ejecutadas, y las otras formas del dispositivo aún se ejecutan. Las tres son provisionales y nunca se cuentan como velocidades. Una ejecución que fue interrumpida a mitad de la programación se mantiene en su recibo marcada como `aborted: true`, con las ranuras que realmente se completaron; la tabla de enrutamiento rechaza las filas interrumpidas, y `table/gaps.md` enumera la ejecución interrumpida. Un hilo de vigilancia protege cada ejecución desde fuera del hilo principal: una compilación nativa que se ejecuta más allá de `compile_ceiling_s` (el valor predeterminado es 900 s) se registra como una fila de capacidad `compile_timeout`, y la RAM del sistema al 88% interrumpe la ejecución; ambos primero vacían el recibo y luego finalizan el proceso con el código de salida 75 (tiempo de espera de la compilación) o 76 (interrupción de la RAM). Cada ejecución también escribe `receipts/<run_id>.run.json` antes de acceder a un dispositivo (ignorada por Git, no es un recibo), por lo que una ejecución interrumpida desde el exterior aún deja un registro de que comenzó. Para detener una ejecución de forma limpia desde el exterior, cree `receipts/<run_id>.stop`: el vigilante vacía el recibo y sale con el código 77.

```
interlock check
```

Esto valida cada recibo bajo `receipts/` y `table/routing.jsonl`. Un árbol vacío está bien. Una ruta de directorio de inicio en un recibo no lo está.

## Ciclo de vida de la ejecución

Una ejecución finaliza en uno de dos estados. Una ejecución finalizada deja `receipts/<run-id>.jsonl` y, para E4b, su serie junto a ella. Una ejecución que se detiene a mitad de camino primero escribe ese recibo, con las llamadas que se completaron y `aborted: true`, y luego mueve el recibo, la serie, el registro y el marcador de ejecución a `aborted/<run-id>-<sha>/` con un `NOTE.json` (el ID de ejecución, el hash del arnés, el inicio y el final, la excepción y los archivos). `aborted/` se ignora en Git. No es un resultado: `interlock import` no lo lee, y esas ejecuciones nunca llegan a la tabla de enrutamiento.

`receipts/` no debe contener un directorio `*-series/` que no tenga un recibo. La siguiente ejecución se niega a comenzar mientras haya uno allí y nombra `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` imprime cada ID de ejecución y un estado: `complete`, `aborted` o `stray`. `quarantine` mueve una ejecución errante a `aborted/` con una nota que indica que se reservó manualmente. Ninguno de los comandos inicia Ollama o un dispositivo. `--quarantine` elige el directorio reservado; el valor predeterminado es `aborted/` debajo del directorio raíz de la ejecución.

Sin `--run-id`, E4b usa `e4b-block` o `e4b-pilot`, luego `<base>-2`, `<base>-3` y así sucesivamente. El ID elegido es la primera línea de la ejecución y un campo en el recibo. Se rechaza un nombre que ya esté siendo utilizado por un recibo, un directorio de serie, un registro, una caché de compilación o una ejecución en cuarentena.

## Modelo de amenazas

interlock accede a dispositivos locales y a HTTP local, y a la red solo para descargar un modelo que una especificación de experimento ya haya nombrado, del editor de ese modelo, o para verificar las fuentes C2 cuando se le solicite.

- Los archivos de registro y la tabla de enrutamiento son archivos locales.
- Las llamadas a Ollama, cuando realmente se inicia una ejecución, se dirigen a `127.0.0.1:11434` (el 5090) o `127.0.0.1:11490` (CPU). Se rechazan las etiquetas de modelos en la nube.
- Las OpenVINO ejecuciones, cuando realmente se inicia una ejecución, utilizan `NPU`, `CPU` o la Intel iGPU (proveedor `0x8086` y `INTEGRATED`). Nunca seleccionan `GPU.1`.
- Un paso 5090 sin `--grant` no se inicia. Un paso NPU o iGPU sin `--npu-session` no se inicia.
- El camino 5090 verifica que el latido del watchdog de VRAM tenga como máximo 60 segundos y que no se haya cargado ningún modelo. El archivo de latido no se abre. Solo se lee su hora de modificación.
- Nada aquí escucha en un puerto.
- interlock no envía telemetría.
- El camino Linux CUDA (`platform: linux-cuda`, para ventanas remotas `colab:<card>`) se ejecuta en la VM remota, no en el equipo. Inicia su propio Ollama en un puerto de bucle dedicado (`127.0.0.1:11441` por defecto), rechaza los puertos compartidos `11434` y `11490`, y rechaza cualquier URL de Ollama que no sea de bucle: el host debe ser `127.0.0.1:<port>`, `[::1]:<port>` o `localhost:<port>` cuando `localhost` solo se resuelve a bucle, con el puerto en dígitos ASCII del 1 al 65535. Lee la tarjeta a través de NVML o `nvidia-smi`. No descarga modelos y no toca los contadores de Windows.
- `interlock c2-sources --verify-live` es el único comando que accede a Internet para C2. Envía solicitudes HTTPS de solo lectura a `registry.ollama.ai` (manifiestos y fragmentos de configuración) y `huggingface.co` (árboles de archivos y `tokenizer.json`), y nada más. No descarga ningún peso del modelo y no envía ninguna credencial. Sin `--verify-live`, `c2-sources` solo lee archivos locales.
- Los resultados de una ventana remota llegan a la tabla solo como un archivo de registro que `interlock import --remote-window` lee. interlock no se conecta a Colab. Un campo con el nombre de una cuenta (`account`, `user`, `owner`, sus `_id` formularios, `ggufRepo`, `gguf_repo` o `gguf_source`) debe contener `sha256:<64-hex>`. Las filas remotas también rechazan un token de cuenta etiquetado, como `account:<id>` o `user=<id>`, en cualquier valor o clave de cadena, y `remote_device` rechaza los ID de cuenta, correos electrónicos, rutas y controladores de repositorio en sus propios campos. Las palabras comunes, como "orientado al usuario", se aceptan en un valor o una clave.
- interlock no alquila una GPU, establece un límite de gasto ni abre la base de datos de offrig. Un precio remoto entra solo como un archivo de instantánea con fecha.

## Diseño

`devices` es la lista de permitidos. `guards` es la concesión, la sesión, el latido y la tarjeta vacía. `receipt` es el esquema y `interlock check`. `importers` reconstruye `table/routing.jsonl` a partir de los registros anclados, y `importers/remote.py` lee el registro de una ventana remota. `interlock table` imprime esa tabla, filtrada por tarea o dispositivo. `table/gaps.md` es la cola de experimentos. El arnés ejecuta una especificación confirmada; `harness/linux_cuda.py` es el camino Linux CUDA para tarjetas remotas. `interlock route` lee esa tabla y no toca un dispositivo.

El resumen del diseño y las cuatro notas de retroalimentación que menciona se almacenan en `docs/handoff/`. La copia en el almacén de memoria del estudio permanece como la versión autorizada.

## Ruta

`interlock route` responde desde `table/routing.jsonl`. No inicia un dispositivo, lee un libro mayor ni llama a offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` coincide con una subcadena del repositorio o la etiqueta. Los guiones y los dos puntos se ignoran, por lo que `Qwen3-8B` coincide con `qwen3:8b`. Una subcadena corta puede coincidir con un repositorio no deseado. El modelo coincidente se imprime. `--thinking off` solo coincide con `off`. `--thinking on` coincide con cualquier nivel de razonamiento registrado, y la decisión mantiene ese nivel, por lo que un brazo OpenVINO `low` permanece `low`. `low`, `medium` y `high` coinciden exactamente cuando la solicitud nombra uno. `--min-tok-s` es el límite mínimo de la tasa de decodificación de la ranura de repetición. Un rechazo nombra ese límite `min_tok_s`. `--budget-left-usd` es una entrada, como `--busy`. Una fila remota cuyo peor caso lo excede es una finalista. Sin ninguna marca, se marca una sugerencia remota como presupuesto no verificado. `--allow-remote` permite que las filas remotas `colab:<card>` entren en el grupo de candidatos (desactivado por defecto), y `--device` filtra por un dispositivo exacto (`colab:L4`, `5090`) o un comodín de proveedor (`colab:*`). `--as-of` tiene como valor predeterminado el día de hoy. La política en sí no lee el reloj.

Generar clasificaciones según la menor tasa de decodificación de ranuras de repetición (tok/s). Todas las demás tareas se clasifican según la mayor latencia de ranuras de repetición, o según el máximo tiempo de espera cuando la fila no tiene ranuras de repetición. El margen es la diferencia entre esos límites conservadores, nunca una mediana. Los empates dejan el 5090 libre. Una fila con más de 30 días de antigüedad está obsoleta. Una fila marcada como provisional, o con menos de dos repeticiones, es provisional. Ambas siguen siendo elegibles y están marcadas.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost`, `oom`, `load_refused`, `timeout`, `ctx_exceeded`, los tipos no medidos (`sampler_error`, `generation_aborted`, `unmeasured`) y una fila con `parity_valid` valores falsos (`parity_invalid`) se enumeran en la sección de excluidos, junto con esos códigos. No son candidatos. Las filas remotas se filtran antes del ciclo de resultados, a menos que la solicitud lo permita (`--allow-remote` o `--device colab:...`), por lo que nunca aparecen en la lista de excluidos o en la razón del rechazo de una solicitud que solo incluye datos locales. Un resultado de capacidad es el código que se muestra cuando la fila también es inválida en cuanto a la paridad. Cuando ninguna fila cumple las restricciones, el comando se cierra con el código 1 y menciona la restricción que vació el conjunto. No se recurre a una alternativa. `--json` imprime un documento válido según `src/interlock/schema/route-decision.v1.json` (marcando los ganadores remotos con `remote: true` y `remote_device`). `--max-ttft-ms` rechaza la solicitud a menos que cada fila comparada registre el mismo `settings.prefix_caching` (`on`, `off` o `default`). No establecido significa desconocido.

Una sugerencia remota incluye un objeto `offrig_plan` cuando se respalda con una oferta RunPod, o `remote: true` y `remote_device` cuando se respalda con una fila `colab:<card>`. interlock no llama a `offrig_plan` ni a `offrig_launch`, y no gasta. Consulte `docs/offrig-integration.md`.

## Ejecutar un experimento

Cada experimento está preregistrado: su especificación en `experiments/<id>/spec.yaml` está confirmada, con la pregunta, la decisión que alimenta, los brazos, el conjunto de datos, las métricas y las reglas de detención, antes de cualquier tiempo de dispositivo. `interlock run` rechaza una especificación que se editó después de su confirmación. El fragmento de la especificación se verifica con respecto a esa confirmación antes de la ejecución (`require_unmodified`) y se imprime mediante `--dry-run`. No se escribe en el registro. Un registro E4b registra el arnés `git_sha` y `clean_tree`. Un registro de tarjeta de relleno registra `experiment_spec` y `run_id`, y cuando se ejecutó la prueba previa, también registra `git_sha` y `clean_tree`. Cualquier otro registro de ejecutor compartido registra solo `experiment_spec` y `run_id`. Por práctica, una regla no se cambia después de ver los resultados; una enmienda con fecha se realiza antes de la ejecución cuyos resultados cuentan, y una ejecución que se descarta como inválida se registra como tal.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

Una especificación `platform: linux-cuda` (las C2 especificaciones en `experiments/c2-*/`) se ejecuta en una VM de Colab remota dentro de una ventana reservada, contra su propio Ollama en un puerto de bucle dedicado.

`--dry-run` muestra el plan y los parámetros que necesitaría una ejecución real, pero no inicia nada. Una ejecución real necesita los indicadores que requieren sus dispositivos: `--grant` para la 5090, `--npu-session` para la NPU o iGPU, y `--cpu-window` para una fase que requiera un uso intensivo de la CPU. Escribe un registro en `receipts/`. `interlock import --remote-window <path>` añade entonces el registro de una ventana remota de Colab a `table/`, y `interlock check` valida los registros. Una reconstrucción completa a partir de los repositorios de origen fijados requiere `--source` (véase más abajo).

## Rutas y el sistema de vigilancia

No se especifica la ruta de ningún equipo en el código. Cada ubicación que se indica a continuación es primero una variable de entorno, luego `[paths]` en `interlock.toml` en el directorio de trabajo, y se utiliza un valor tal como lo proporcionaste:

- `INTERLOCK_SCRATCH_ROOT` / `scratch_root`: el directorio de trabajo temporal que offrig abre como su proyecto para `offrig verify calibrate`. Valor predeterminado: `interlock-scratch`, dentro del directorio temporal del sistema.
- `INTERLOCK_WEIGHTS_HOT` / `weights_hot`: la caché de pesos que utiliza una rama de la prueba con `hf_cache: hot` para leer sus pesos; la ausencia de una copia se considera un rechazo. Valor predeterminado: `hot/` dentro de `HF_HOME`, o dentro de la caché de Hugging Face del usuario.
- `WATCHDOG_HEARTBEAT` / `heartbeat`: el archivo de señal de actividad del monitor de la VRAM, que el guardián de la señal de actividad lee antes de cada paso en la 5090. No hay un valor predeterminado: si no se establece, la ejecución se rechaza y muestra el error `WATCHDOG_HEARTBEAT`. Si no se ejecuta ningún monitor, se pasa el valor `--no-watchdog`; el registro `notes` del informe lo indica.

```toml
[paths]
scratch_root = "scratch"
weights_hot = "hf-hot"
heartbeat = "watchdog/HEARTBEAT"
```

## Importación de registros

La mayoría de los usuarios solo realizan fusiones. `interlock import --remote-window <file>` añade las filas de un registro de la ventana de Colab (el archivo JSONL de la fila de enrutamiento que `interlock run` escribe, o el archivo JSON `{run_id, date, remote_device, records}` más antiguo) al archivo `table/routing.jsonl` existente. No necesita ningún otro repositorio y rechaza una fila que ya esté en la tabla.

Una reconstrucción completa (`interlock import` sin `--remote-window`, o con `--offers`) regenera las filas fijas a partir de tres repositorios de origen, cada uno en un commit específico: `rnd` (`08ca419f647a3ed69ee645610eaa1c7f35f40f91`), `npu` (`275e51101f7557bf50898608812cd61d25d5ff17`, en la rama `npu-buildout` de rnd y etiquetado como `data/npu-buildout-275e511`, por lo que es un clon de rnd) y `aspire` (`3b05eab928894311993ffa3af2075fef39cccce6`). Indique dónde se encuentra cada clon con `--source name=path` (repetible), o una sola vez en `interlock.toml`, junto a la tabla:

```toml
[import.sources]
rnd = "../rnd"
npu = "../rnd"
aspire = "../aspire-si"
```

La opción tiene prioridad sobre el archivo y no existe una ruta predeterminada. El commit no es configurable: se rechaza un clon que no lo contenga. La importación escribe el nombre, el identificador y la ruta de cada fuente (tal como se proporcionó) al final de `table/gaps.md`, nunca en una fila. Una reconstrucción reemplaza solo las filas que genera; todas las demás filas de la tabla permanecen, y se rechaza una reconstrucción que eliminaría una fila que le pertenece, a menos que se proporcione `--allow-shrink`.

## Cuando Ollama se bloquea

`interlock ollama-health` pregunta a `GET /api/version` en `127.0.0.1:11434` (el servidor 5090) y a `127.0.0.1:11490` (el servidor solo de CPU). Un puerto está en buen estado cuando esa respuesta llega en un plazo de 5 segundos. El comando imprime una línea por puerto y sale con el código 1 si alguno de los dos no está funcionando.

Un `interlock run` real reinicia Ollama solo en `127.0.0.1:11434`, solo entre etapas, y solo después de que una etapa haya fallado con un error de transporte, un tiempo de espera que es un bloqueo real, o un error 5xx mientras ese servidor no está funcionando. "No funciona" significa tres comprobaciones de estado fallidas, con dos segundos de diferencia. Una solicitud que superó su propio plazo es un bloqueo solo cuando el estado falla después, o `/api/ps` no muestra que el modelo está progresando. Un mensaje largo cuya versión aún responde, y cuyo modelo aún está en la lista, no es un bloqueo. Un mensaje de 118 000 tokens no se detiene por ser lento.

El puerto `11490` es el indexador de CPU de offrig. El sistema de vigilancia aún informa sobre él. Un `11490` que no funciona detiene la ejecución. interlock nunca detiene ese proceso, y ninguna concesión cubre su reinicio.

Inmediatamente antes de que se detenga algo, el reinicio vuelve a verificar `--grant`, el "latido" del sistema de vigilancia y `/api/ps`. Se rechaza una lista que nombra un modelo de otra sesión. Un `/api/ps` ilegible no bloquea, por sí solo, el reinicio de un servidor que ya no está funcionando. Durante un bloqueo, el sistema de vigilancia aún puede responder. Un `/api/ps` ilegible rechaza entonces el reinicio, porque el modelo de otra sesión se eliminaría sin que se viera.

El proceso que se detiene es el que escucha en `11434`. Nada se detiene por nombre. La aplicación de la bandeja permanece activa a menos que sea ese receptor. `ollama serve` se reproduce desde la línea de inicio registrada para esta ejecución: el argv, cada variable `OLLAMA_*` y `LLAMA_ARG_*`, y el archivo de registro. Stdout y stderr se agregan a ese registro. interlock no inicia sesión. No hay una variable `OLLAMA_CUDA_BACKEND`. Si es necesario forzar un backend, la variable es `OLLAMA_LLM_LIBRARY`, y pertenece a la línea de inicio registrada.

El registro es un archivo JSON con el nombre `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` y `SWITCHYARD_OLLAMA_EXE` aún funcionan para esta versión y advierten. Los campos son `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` y `backend_line`. Un inicio registrado en este sistema es `cmd /c ollama serve` con `OLLAMA_HOST=127.0.0.1:11434` y `OLLAMA_MODELS=E:/AI-Models/Ollama`, registrando en `E:/AI/ollama-cu134/system-11434.log`. La biblioteca cargada es `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. El sha256 del editor de ese archivo inicia `6d3128e946bc9c95`. Una ejecución registra la línea de inicio que se le proporcionó y reproduce esa. Un registro faltante rechaza el reinicio antes de que se detenga algo.

Al comienzo de una ejecución, el conjunto completo de `OLLAMA_*` y `LLAMA_ARG_*` del registro se compara con el entorno del proceso activo. Esa comparación es exacta. Cuando no se puede leer el entorno del proceso, la línea de registro `server config` del servidor es la alternativa, y esa comparación también es exacta: un registro que es solo parte de los valores predeterminados registrados deshabilita los reinicios. Una discrepancia significa que esta ejecución no realiza ningún reinicio. La razón se escribe en el marcador de ejecución como `ollama_restart_disabled`. Nada se detiene. `OLLAMA_HOST` coincide con o sin un esquema `http://` o `https://`.

Una vez que el servidor vuelve a estar en línea, el proceso lee la biblioteca sha256 desde `library_path`. Se lee la atención Flash, el tipo de caché KV, la línea `load_backend` y el conjunto completo `OLLAMA_*` y `LLAMA_ARG_*` a partir de los bytes del registro escritos después del inicio. Los bytes anteriores a ese desplazamiento corresponden al servidor anterior, y una línea allí no coincide. Si no se puede leer el tamaño del registro, ese desplazamiento es desconocido y el proceso se detiene. Un registro faltante está vacío y su desplazamiento es cero. Se verifica una línea `load_backend` que aún no se ha escrito después de la primera carga del modelo y antes de que se conserve esa muestra. Una línea faltante o diferente detiene el proceso. La línea de configuración del servidor del nuevo segmento debe cubrir todas las variables bajo las cuales se midió el proceso. Los valores predeterminados que el proceso no estableció también se pueden registrar. Un servidor que vuelve a estar en línea con una biblioteca diferente o en un entorno diferente detiene el proceso. La versión del controlador CUDA de `nvidia-smi` no es el criterio de verificación: un reinicio no puede cambiarla. La atención Flash y el tipo de caché KV que el registro no refleja provocan que falle la coincidencia. No se copian desde el registro y se consideran una lectura.

Si el servidor no está operativo en 60 segundos, o si la tarjeta no ha vuelto a su estado inicial, el proceso se detiene y se indica que una persona debe iniciar Ollama. Como máximo, se permiten dos reinicios por proceso. Un reinicio fallido se cuenta dentro de ese límite.

Cada reinicio se registra en el marcador del proceso y en `settings.ollama_restarts`: cuándo, qué puerto, por qué (`down` o `hung`), los PID antiguos y nuevos, el tiempo hasta que esté operativo y la versión. El paso que estaba en curso es inconcluso, nunca se completa con éxito. Solo se vuelve a intentar cuando la especificación establece `retry` (el valor predeterminado es ninguno). Un reintento conservado establece `settings.retried`. La primera muestra conservada después de un reinicio establece `settings.cold_after_restart` y se excluye de los percentiles de calentamiento. Una medición realizada durante un reinicio no se conserva.

`--dry-run` imprime esta política y no se comunica con ninguno de los puertos.

## Desarrollo

Python 3.15 (`.python-version`); también funcionan las versiones 3.13 y 3.14. `uv sync --frozen` crea el entorno del proyecto a partir del archivo de bloqueo. No comparte el entorno de la NPU. OpenVINO 2026.4.1 es el `intel` componente adicional. No es necesario para las pruebas.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribuciones

Los errores se reportan en GitHub Issues. Un informe de seguridad sigue [SECURITY.md](SECURITY.md) y no es un problema público. Los cambios se envían como solicitudes de incorporación (pull requests) contra `main`. Los requisitos y los comandos para ejecutar primero se encuentran en [CONTRIBUTING.md](CONTRIBUTING.md).

Partes de este repositorio fueron escritas con Grok (xAI). Se reconoce a Grok, Kimi y Claude en [CONTRIBUTORS.md](CONTRIBUTORS.md).

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

