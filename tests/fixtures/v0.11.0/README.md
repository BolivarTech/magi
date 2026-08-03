# Fixtures de `magi.toml` de v0.11.0

Archivos de configuración **reales** de magi-rs v0.11.0, usados por
`src/config/migrate.rs` para probar la pasada de migración a v0.12.0 (REQ-A21c, SC-A21d).

## Por qué son reales y no escritos a mano

v0.12.0 rompe **todo** `magi.toml` de v0.11.0, y la decisión del usuario (2026-08-01) fue no
proveer bandera de escape: el mensaje de migración es la **única** defensa. Si al mensaje se le
escapa un patrón, el usuario queda con un binario que no arranca y sin downgrade limpio.

Un fixture escrito a mano prueba que el mensaje **se emite**. Solo uno real prueba que
**alcanza** — que nombra todas las incompatibilidades de ese archivo y que lo que propone
efectivamente parsea en v0.12.0.

## Por qué están commiteados y no se generan en tiempo de test

Generarlos durante la suite ataría los tests a tener v0.11.0 instalado. No lo hay en CI, ni en un
clon nuevo, ni dentro de un año. Un test que no puede correr no defiende nada, y éste es el único
que defiende la migración.

## Procedencia

Generados el **2026-08-02** con el binario publicado de v0.11.0:

```bash
cargo install magi-rs --version 0.11.0 --root <tmp>/magi-v11 --locked
cd <tmp>/genfix && <tmp>/magi-v11/bin/magi-rs.exe --init-config
```

| Archivo | Origen |
|---|---|
| `default.toml` | Salida **verbatim** de `magi-rs --init-config` (4769 bytes). Canónico. |
| `with-models.toml` | `default.toml` con los tres modelos por mage cambiados a valores no built-in. |
| `full.toml` | `default.toml` con cinco knobs avanzados de `[memory]` y `[embedding]` descomentados. |
| `with-credentials.toml` | `default.toml` con `[openai].base_url = "https://user:s3cr3t@host/v1"`, más un comentario TOML al final de esa línea con el marcador del escáner de secretos (ver abajo). |

Las tres variantes se derivan de `default.toml` agregando o modificando **solo claves que el
schema de v0.11.0 acepta** — verificado contra `git show v0.11.0:src/config.rs` y
`git show v0.11.0:src/memory/config.rs`. v0.11.0 parsea con `deny_unknown_fields`, así que una
clave inventada produciría un archivo que v0.11.0 habría rechazado, y el fixture no probaría nada.

## Cómo se verificó cada uno

Cada fixture se colocó como `.magi/magi.toml` en un workspace temporal y se ejecutó el binario de
v0.11.0 contra él, buscando en la salida el warning que ese binario emite cuando la configuración
no parsea (`"is invalid and was ignored"`, de `MagiConfig::load`):

```bash
<tmp>/magi-v11/bin/magi-rs.exe init
cp <fixture> .magi/magi.toml
MAGI_PASSPHRASE="…" <tmp>/magi-v11/bin/magi-rs.exe query -i "hi" --timeout 2 2>&1 \
  | grep -ci "is invalid and was ignored"
```

Cero coincidencias significa que v0.11.0 aceptó el archivo. Los cuatro dieron cero.

**El método se validó antes de confiar en él.** Se corrió primero con un archivo deliberadamente
inválido (`bogus_unknown_key = 1`), que sí produjo el warning. Sin ese control, "no apareció el
warning" sería indistinguible de "el warning nunca aparece", y los cuatro habrían pasado por
construcción.

## `with-credentials.toml` contiene un secreto a propósito

Lleva `s3cr3t` embebido en la `base_url`. Es exactamente lo que `tests/no_hardcoded_secrets.rs`
busca, así que necesita una exención explícita en ese escáner.

La exención es **por línea, no por directorio**: el marcador `allow-secret-scan` va como
comentario TOML al final de la línea de `base_url`, y `tests/fixtures` se **agregó** a los
directorios que el escáner recorre. Excluir el directorio habría sido más simple y peor — los
fixtures son justo la clase de archivo donde alguien pega una credencial real sin querer
mientras los genera, así que la superficie tiene que quedar vigilada salvo en la línea donde el
secreto es deliberado.

El comentario **no altera lo que v0.11.0 parsea** —es un comentario TOML— y el archivo se
re-verificó con el binario de v0.11.0 **después** de agregarlo, no antes.

## Regenerarlos

Repetir los comandos de arriba. `default.toml` debe salir byte-idéntico mientras se use el mismo
v0.11.0 publicado; las variantes se re-derivan de él con los cambios de la tabla.
