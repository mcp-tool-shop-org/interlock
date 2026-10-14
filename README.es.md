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
  <a href="https://codecov.io/gh/mcp-tool-shop-org/interlock"><img src="https://codecov.io/gh/mcp-tool-shop-org/interlock/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/interlock"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/interlock/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/15467"><img src="https://www.bestpractices.dev/projects/15467/badge" alt="OpenSSF Best Practices"></a>
</p>

# interlock: enrutador de dispositivos de IA local

> **Prototipo, 0.1.x: un instrumento de investigación.** La CLI, el formato de los resultados y las especificaciones del experimento pueden cambiar sin previo aviso entre las versiones 0.x. Interlock es solo una herramienta de consulta: sugiere un modelo, un motor y un dispositivo, y nunca procesa tráfico. Cada número se midió en una sola configuración (Intel Core Ultra 9 285K con su NPU y iGPU, además de una RTX 5090). Considere sus rutas como evidencia de esa configuración, no como una referencia general.

El enclavamiento ferroviario evita que un tren se dirija a una vía conflictiva. Esta herramienta hace lo mismo para los dispositivos: envía cada tarea a un modelo, un motor y un dispositivo.

Le proporciona una tarea (incrustación, reordenamiento, NLI, verificación, generación) y las restricciones que importan (calidad, latencia, contexto, nivel de razonamiento). Responde con un modelo, un motor, un dispositivo y la configuración, y muestra los resultados detallados de cada medición. Los complementos AUTO y HETERO de OpenVINO pueden seleccionar un dispositivo una vez que ya ha seleccionado el modelo. Ningún componente de esa pila selecciona el modelo. Esa diferencia es el producto.

## Instalación

Desde una copia:

```
uv sync
uv run interlock --help
```

Desde PyPI, a partir de la versión 0.1.0:

```
pip install mcptoolshop-interlock
```

```
uv tool install mcptoolshop-interlock
```

El nombre de la distribución es `mcptoolshop-interlock`. La importación y el comando son `interlock`. Solo Python 3.12 (`requires-python` es `>=3.12,<3.13`). CI ejecuta las pruebas en Ubuntu. Se ha escrito una ejecución de medición en vivo para la configuración de Windows en la que se midieron los datos de esta tabla.

Lo que este programa puede acceder se indica en el [modelo de amenazas](#threat-model).

El esquema de resultados, la tabla de enrutamiento y el conjunto de pruebas se crearon primero. `interlock route` lee esa tabla. No utiliza tiempo de procesamiento del dispositivo.

## Lo que indica la tabla hoy

La tabla de enrutamiento contiene 509 filas de resultados registrados. Algunas de las respuestas que ya proporciona:

- **Un modelo pequeño funciona bien en el lado de Intel.** Phi-3.5-mini genera aproximadamente 20-22 tokens/s en la CPU y la iGPU, y aproximadamente 15 tokens/s en la NPU, con una solicitud de 1024 tokens y el razonamiento desactivado. Esto deja la 5090 libre para tareas más grandes.
- **Un modelo de 8B no pertenece al lado de Intel.** Con la misma solicitud de 1024 tokens y el razonamiento desactivado, Qwen3-8B genera aproximadamente 10-12 tokens/s en la CPU, la iGPU y la NPU. En la 5090, genera aproximadamente 208 tokens/s con el razonamiento desactivado y 211 con el razonamiento activado. Si solicita 15 tokens/s con la 5090 ocupada, el enrutador se niega y muestra el experimento que explica la diferencia: `interlock route --task generate --model qwen3-8b --min-tok-s 15 --busy 5090`.
- **Algunas configuraciones no se ejecutan en absoluto.** La NPU rechazó Qwen3-Embedding con entradas de 512 tokens, tanto de una en una como en lotes de 8. Estas filas se conservan como hallazgos de capacidad, pero nunca se clasifican como candidatos lentos.

Estas cifras son las tasas de generación registradas en los resultados E3 y e3-ref. Las notas de la ejecución se encuentran en [experiments/e3/run-notes-2026-10-10-11.md](experiments/e3/run-notes-2026-10-10-11.md). `interlock route` clasifica según la peor repetición, no la mediana, por lo que sus elecciones son más conservadoras que estos resúmenes.

## Lo que no hará

Interlock no procesa tráfico ni actúa como proxy para las solicitudes. No entrena modelos. No edita offrig, npu-serve ni rnd. Estos permanecen con sus propietarios. Una capa de servicio necesita su propia descripción.

## Dispositivos

Cada dispositivo tiene un nombre.

| Nombre | Qué es |
|---|---|
| `NPU` | Intel AI Boost en el Core Ultra 9 285K |
| `iGPU` | La GPU de Intel. Una enumeración en vivo debe indicar el proveedor `0x8086` y el tipo de dispositivo `INTEGRATED`. Se registra el nombre completo, no la decisión. |
| `CPU` | La CPU del mismo procesador |
| `5090` | La RTX 5090, a la que solo se accede a través de Ollama en `127.0.0.1:11434` |

`GPU.1` es la 5090 tal como la ve OpenVINO en esta configuración. El solucionador rechaza ese índice incluso si el nombre o el proveedor cambian, rechaza un `GPU` sin más, rechaza `AUTO`, `HETERO` y `MULTI`, y rechaza una GPU que no sea del proveedor `0x8086` y `INTEGRATED`. Una lista sin ninguna de estas propiedades del dispositivo aún requiere que el nombre completo contenga "Intel". Un dispositivo faltante es un error. Interlock no recurre a uno más débil.

La NPU es exclusiva. Un paso de la 5090 solo se ejecuta dentro de una concesión de Publisher. El ejecutor registra esa concesión. Las etapas que requieren mucha CPU esperan una ventana que anuncia el Publisher.

## Resultados

Una fila de enrutamiento indica la tarea, el modelo (repositorio o etiqueta, revisión o resumen, cuantificación, licencia), el motor y su versión, el dispositivo, la configuración y la calidad y la latencia que midió. Las métricas de calidad llevan su propio `n`, intervalo y hash del conjunto de datos. La latencia distingue entre frío y caliente. Una sola ejecución se marca como provisional. La fila apunta al resultado fuente y al commit desde el que se importó, o a una especificación de experimento y un ID de ejecución.

Una ejecución también puede registrar una fila de capacidad en lugar de una velocidad. `device_lost` y `hang` no llevan información de tiempo. `native_binary_rejected` (el dispositivo compiló el modelo y rechazó el binario nativo en la ejecución) solo conserva el tiempo de compilación como evidencia y tiene un alcance limitado al modelo, el dispositivo y la forma estática: solo las ejecuciones con esa clave idéntica se marcan como no ejecutadas, y las otras formas del dispositivo aún se ejecutan. Las tres son provisionales y nunca se cuentan como velocidades. Una ejecución que se detuvo a mitad de la programación se conserva en sus resultados marcada como `aborted: true`, con las ranuras que realmente se completaron; la tabla de enrutamiento rechaza las filas interrumpidas, y `table/gaps.md` enumera la ejecución interrumpida. Un hilo de observador protege cada ejecución de elementos externos al hilo principal: una compilación nativa que se ejecuta durante más de `compile_ceiling_s` (el valor predeterminado es 900 s) se registra como una fila de capacidad `compile_timeout`, y la RAM al 88% interrumpe la ejecución; ambas limpian primero los resultados y terminan el proceso con el código de salida 75 (tiempo de espera de la compilación) o 76 (interrupción de la RAM). Cada ejecución también escribe `receipts/<run_id>.run.json` antes de acceder a un dispositivo (se ignora en git, no es un resultado), por lo que una ejecución interrumpida desde el exterior aún deja un registro de que comenzó. Para detener una ejecución de forma limpia desde el exterior, cree `receipts/<run_id>.stop`: el observador limpia los resultados y sale con el código 77.

```
interlock check
```

Esto valida cada resultado según `receipts/` y `table/routing.jsonl`. Un árbol vacío está bien. Una ruta en el directorio de inicio en un resultado no lo está.

## Ejecutar ciclo de vida

Una ejecución finaliza en uno de dos estados. Una ejecución finalizada deja `receipts/<run-id>.jsonl` y, para E4b, su serie junto a ella. Una ejecución que se detiene a la mitad escribe primero ese registro, con las llamadas que se completaron y `aborted: true`, luego mueve el registro, la serie, el registro y el marcador de ejecución a `aborted/<run-id>-<sha>/` con un `NOTE.json` (el ID de la ejecución, el hash del conjunto de pruebas, el inicio y el final, la excepción y los archivos). `aborted/` se ignora en git. No es un resultado: `interlock import` no lo lee, y esas ejecuciones nunca llegan a la tabla de enrutamiento.

`receipts/` no debe contener un directorio `*-series/` que no tenga ningún registro. La siguiente ejecución se negará a iniciarse mientras haya uno, y lo nombrará `interlock runs quarantine <run-id>`.

```
interlock runs list
interlock runs quarantine <run-id>
```

`list` imprime el ID de cada ejecución y un estado: `complete`, `aborted` o `stray`. `quarantine` mueve un elemento sobrante a `aborted/` con una nota que indica que se reservó manualmente. Ninguno de los comandos inicia Ollama ni un dispositivo. `--quarantine` elige el directorio reservado; el valor predeterminado es `aborted/` dentro del directorio raíz de la ejecución.

Sin `--run-id`, E4b utiliza `e4b-block` o `e4b-pilot`, luego `<base>-2`, `<base>-3`, y así sucesivamente. El ID elegido es la primera línea de la ejecución y un campo en el registro. Se rechaza un nombre ya utilizado por un registro, un directorio de serie, un registro, una caché de compilación o una ejecución en cuarentena.

## Modelo de amenazas

interlock interactúa con dispositivos locales y HTTP local, y con la red solo para descargar un modelo que ya haya nombrado la especificación de un experimento, del editor de ese modelo.

- Los registros y la tabla de enrutamiento son archivos locales.
- Las llamadas de Ollama, cuando realmente se inicia una ejecución, van a `127.0.0.1:11434` (el 5090) o `127.0.0.1:11490` (CPU). Se rechazan las etiquetas de modelos en la nube.
- Las ejecuciones de OpenVINO, cuando realmente se inicia una ejecución, utilizan `NPU`, `CPU` o la iGPU de Intel (proveedor `0x8086` y `INTEGRATED`). Nunca seleccionan `GPU.1`.
- Un paso 5090 sin `--grant` no se inicia. Un paso NPU o iGPU sin `--npu-session` no se inicia.
- La ruta 5090 verifica que el latido del reloj de VRAM tenga como máximo 60 segundos y que no se haya cargado ningún modelo. El archivo de latido no se abre. Solo se lee su hora de modificación.
- Nada aquí escucha en un puerto.
- interlock no envía telemetría.
- interlock no alquila una GPU, establece un límite de gasto ni abre la base de datos de offrig. Un precio remoto entra solo como un archivo de instantánea con fecha.

## Diseño

`devices` es la lista de permitidos. `guards` es la concesión, la sesión, el latido y la tarjeta vacía. `receipt` es el esquema y `interlock check`. `importers` reconstruye `table/routing.jsonl` a partir de los registros fijados. `interlock table` imprime esa tabla, filtrada por tarea o dispositivo. `table/gaps.md` es la cola de experimentos. El conjunto de pruebas ejecuta una especificación confirmada. `interlock route` lee esa tabla y no interactúa con un dispositivo.

El resumen de la compilación y las cuatro notas de retroalimentación que nombra se guardan en `docs/handoff/`. La copia en el almacén de memoria del estudio sigue siendo la versión autorizada.

## Ruta

`interlock route` responde desde `table/routing.jsonl`. No inicia un dispositivo, lee un libro mayor ni llama a offrig.

```
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` coincide con una subcadena del repositorio o la etiqueta. Los guiones y los dos puntos se ignoran, por lo que `Qwen3-8B` coincide con `qwen3:8b`. Una subcadena corta puede coincidir con un repositorio no deseado. Se imprime el modelo coincidente. `--thinking off` solo coincide con `off`. `--thinking on` coincide con cualquier nivel de razonamiento registrado, y la decisión mantiene ese nivel, por lo que un brazo OpenVINO `low` permanece en `low`. `low`, `medium` y `high` coinciden exactamente cuando la solicitud nombra uno. `--min-tok-s` es el límite mínimo de la tasa de decodificación de la ranura de repetición. Un rechazo nombra ese límite `min_tok_s`. `--budget-left-usd` es una entrada, como `--busy`. Una fila remota cuyo peor caso lo excede es un finalista. Sin ninguna marca, se marca una sugerencia remota como presupuesto no verificado. `--as-of` tiene como valor predeterminado el día de hoy. La política en sí no lee el reloj.

Generar clasificaciones según la menor tasa de decodificación de tokens por ranura de repetición. Todas las demás tareas se clasifican según la mayor latencia de la ranura de repetición, o según el máximo tiempo de espera cuando la fila no tiene ranuras de repetición. El margen es la diferencia entre esos límites conservadores, nunca una mediana. Los empates dejan el 5090 libre. Una fila con más de 30 días de antigüedad está obsoleta. Una fila marcada como provisional, o con menos de dos repeticiones, es provisional. Ambas siguen siendo elegibles y están marcadas.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost` y una fila con `parity_valid` falso (`parity_invalid`) se enumeran en la sección excluidos, con esos códigos. No son candidatos. Un resultado de capacidad es el código que se muestra cuando la fila también es inválida en términos de paridad. Cuando ninguna fila cumple con las restricciones, el comando sale con el código 1 y nombra la restricción que vació el conjunto. No se recurre a una alternativa. `--json` imprime un documento válido con respecto a `src/interlock/schema/route-decision.v1.json`. `--max-ttft-ms` se niega a menos que cada fila comparada registre el mismo `settings.prefix_caching` (`on`, `off` o `default`). No establecido significa desconocido.

Una sugerencia remota incluye un objeto `offrig_plan`. interlock no llama a `offrig_plan` ni a `offrig_launch`, y no gasta. Consulte `docs/offrig-integration.md`.

## Ejecutar un experimento

Cada experimento está preregistrado: su especificación en `experiments/<id>/spec.yaml` se confirma, con la pregunta, la decisión que alimenta, los brazos, el conjunto de datos, las métricas y las reglas de detención, antes de cualquier tiempo de dispositivo. `interlock run` rechaza una especificación que se editó después de su confirmación. El blob de la especificación se verifica con respecto a esa confirmación antes de la ejecución (`require_unmodified`) y se imprime por `--dry-run`. No se escribe en el registro. Un registro de E4b registra el conjunto de pruebas `git_sha` y `clean_tree`. Un registro de tarjeta de relleno registra `experiment_spec` y `run_id`, y cuando se ejecutó la fase de preverificación, también registra `git_sha` y `clean_tree`. Cualquier otro registro de ejecutor compartido registra solo `experiment_spec` y `run_id`. Por práctica, una regla no se cambia después de ver los resultados; una enmienda con fecha se realiza antes de la ejecución cuyos resultados se tienen en cuenta, y una ejecución reservada como inválida se registra como tal.

```
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

`--dry-run` imprime el plan y las protecciones que necesitaría una ejecución real, y no inicia nada. Una ejecución real necesita las marcas que solicitan sus dispositivos: `--grant` para el 5090, `--npu-session` para la NPU o la iGPU, y `--cpu-window` para una fase con gran carga en la CPU. Escribe un registro en `receipts/`. `interlock import` luego reconstruye `table/` a partir de los registros confirmados, y `interlock check` los valida.

## Cuando Ollama se bloquea

`interlock ollama-health` pregunta a `GET /api/version` en `127.0.0.1:11434` (el servidor 5090) y `127.0.0.1:11490` (el servidor solo para CPU). Un puerto está en buen estado cuando se recibe esa respuesta en un plazo de 5 segundos. El comando imprime una línea por puerto y sale con el código 1 si alguno de los dos no funciona.

Un reinicio real de `interlock run` solo reinicia Ollama en `127.0.0.1:11434`, solo entre pasos y solo después de que un paso haya fallado debido a un error de transporte, un tiempo de espera que realmente bloquea el sistema o un error 5xx mientras ese servidor está inactivo. Inactivo significa tres comprobaciones de estado fallidas, con dos segundos de diferencia. Una solicitud que superó su propio plazo solo se considera un bloqueo cuando el estado falla posteriormente, o `/api/ps` no muestra que el modelo está progresando. Un mensaje largo cuya versión aún responde y cuyo modelo aún está en la lista no se considera un bloqueo. Un mensaje de 118.000 tokens no se detiene por ser lento.

El puerto `11490` es el indexador de CPU de offrig. El estado aún lo informa. Un `11490` inactivo detiene la ejecución. Interlock nunca detiene ese proceso y ninguna autorización cubre su reinicio.

Inmediatamente antes de que se detenga algo, el reinicio vuelve a comprobar `--grant`, el latido del watchdog y `/api/ps`. Se rechaza una lista que nombra un modelo de otra sesión. Un `/api/ps` ilegible no impide, por sí solo, el reinicio de un servidor que ya está inactivo. Durante un bloqueo, el estado aún puede responder. Un `/api/ps` ilegible rechaza entonces el reinicio, porque el modelo de otra sesión se descartaría sin que se viera.

El proceso que se detiene es el que escucha en `11434`. Nada se detiene por nombre. La aplicación de la bandeja del sistema permanece activa a menos que sea ese receptor. `ollama serve` se reproduce desde la línea de inicio registrada para esta ejecución: el argv, cada variable `OLLAMA_*` y `LLAMA_ARG_*`, y el archivo de registro. Stdout y stderr se añaden a ese registro. Interlock no inicia sesión. No hay ninguna variable `OLLAMA_CUDA_BACKEND`. Si es necesario forzar un backend, la variable es `OLLAMA_LLM_LIBRARY` y debe incluirse en la línea de inicio registrada.

El registro es un archivo JSON con el nombre `INTERLOCK_OLLAMA_SERVE`. `SWITCHYARD_OLLAMA_SERVE`, `SWITCHYARD_OLLAMA_LOG` y `SWITCHYARD_OLLAMA_EXE` siguen funcionando para esta versión y emiten una advertencia. Los campos son `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256` y `backend_line`. Un inicio registrado en este equipo es `cmd /c ollama serve` con `OLLAMA_HOST=127.0.0.1:11434` y `OLLAMA_MODELS=E:/AI-Models/Ollama`, registrando en `E:/AI/ollama-cu134/system-11434.log`. La biblioteca cargada es `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v13\ggml-cuda.dll`. El hash sha256 del editor de ese archivo comienza con `6d3128e946bc9c95`. Una ejecución registra la línea de inicio que se le proporcionó y la reproduce. Un registro faltante rechaza el reinicio antes de que se detenga algo.

Al inicio de una ejecución, el conjunto completo de `OLLAMA_*` y `LLAMA_ARG_*` del registro se compara con el entorno del proceso activo. Esa comparación es exacta. Cuando no se puede leer el entorno del proceso, la línea de registro `server config` del servidor es la alternativa, y esa comparación también es exacta: un registro que solo forma parte de los valores predeterminados registrados deshabilita los reinicios. Una discrepancia significa que esta ejecución no realiza ningún reinicio. La razón se escribe en el marcador de ejecución como `ollama_restart_disabled`. Nada se detiene. `OLLAMA_HOST` coincide con o sin un esquema `http://` o `https://`.

Después de que el servidor vuelve a estar en línea, la ejecución lee el hash sha256 de la biblioteca desde `library_path`. La atención flash, el tipo de caché KV, la línea `load_backend` y el conjunto completo de `OLLAMA_*` y `LLAMA_ARG_*` se leen de los bytes del registro escritos después del inicio. Los bytes anteriores a ese desplazamiento corresponden al servidor anterior, y una línea allí no coincide. Si no se puede leer el tamaño del registro, ese desplazamiento es desconocido y la ejecución se detiene. Un registro faltante está vacío y su desplazamiento es cero. Se comprueba una línea `load_backend` que aún no se ha escrito después de la primera carga del modelo y antes de que se guarde esa muestra. Una línea faltante o diferente detiene la ejecución. La línea de configuración del servidor del nuevo segmento debe cubrir todas las variables con las que se midió la ejecución. Los valores predeterminados que el proceso no estableció también pueden registrarse. Un servidor que vuelve a estar en línea con una biblioteca diferente o en un entorno diferente detiene la ejecución. La versión del controlador CUDA de `nvidia-smi` no es la comprobación: un reinicio no puede cambiarla. La atención flash y el tipo de caché KV que el registro no refleja hacen que falle la coincidencia. No se copian desde el registro y se consideran una lectura.

Si el servidor no está en buen estado en 60 segundos, o la tarjeta no ha vuelto a su estado base, la ejecución se detiene y se indica que una persona debe iniciar Ollama. Como máximo, dos reinicios por ejecución. Un reinicio fallido se cuenta en ese límite.

Cada reinicio se escribe en el marcador de ejecución y en `settings.ollama_restarts`: cuándo, qué puerto, por qué (`down` o `hung`), los PID antiguos y nuevos, el tiempo hasta que esté en buen estado y la versión. El paso que estaba en curso es inconcluso, nunca se considera un éxito. Solo se vuelve a intentar cuando la especificación establece `retry` (el valor predeterminado es ninguno). Un reintento guardado establece `settings.retried`. La primera muestra guardada después de un reinicio establece `settings.cold_after_restart` y se excluye de los percentiles cálidos. Una medición tomada durante un reinicio no se guarda.

`--dry-run` imprime esta política y no se comunica con ninguno de los puertos.

## Desarrollo

Python 3.12. `uv sync --frozen` crea el entorno del proyecto a partir del archivo de bloqueo. No comparte el entorno de la NPU. OpenVINO 2026.4.1 es el extra `intel`. Las pruebas no lo necesitan.

```
uv sync --frozen
uv run interlock check
uv run pytest
```

## Contribuyendo

Los errores se envían a GitHub Issues. Un informe de seguridad sigue [SECURITY.md](SECURITY.md) y no es un problema público. Los cambios se envían como solicitudes de extracción contra `main`. La barra y los comandos para ejecutar primero están en [CONTRIBUTING.md](CONTRIBUTING.md).

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>

