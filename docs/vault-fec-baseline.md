# ConcatenatedFec — línea base de performance (REQ-V36)

> **Historial de desempeño, NO un gate de decisión.** La elección de FEC está fijada por diseño
> (D-V02): la resiliencia de canal es parte de la tesis de endurecimiento. Este número existe para
> comparar en releases futuros, **no** para reabrir la decisión. Medido con
> `cargo run --release --bin bench_vault_crypto` sobre `magi-rs` 0.8.0 (rustc 1.97, Windows MSVC).

## Medición (2026-07-15)

Corpus: **1000 récords × 220 B** de plaintext (tamaño representativo de un mensaje de conversación).

| Métrica | Valor |
|---|---|
| Plaintext total | 220 000 B |
| Encoded total (en disco) | 1 364 000 B |
| **Expansión en disco** | **≈ 6.2×** |
| **Decrypt por récord** | **≈ 5.3 ms** |
| Decrypt total (1000 récords) | ≈ 5.3 s |

## Interpretación honesta

**La expansión real (≈6.2×) es mayor que el ≈2.3× que la spec estimó.** El ≈2.3× aplica al *payload
grande* (RS 1.14× × Viterbi 2×); para **mensajes chicos** (220 B) los overheads fijos dominan —
nonce + tag AEAD (~28 B), padding del bloque RS(255,223), la duplicación de Viterbi y el base64
externo se pagan sobre un payload pequeño, así que la razón sube.

**El decrypt (~5.3 ms/récord) es el costo de Viterbi.** El crypto in-tree anterior era **RS-only (sin
Viterbi)**; `ConcatenatedFec` **agrega** la etapa Viterbi (corrección de errores de bit + coding
gain), que es computacionalmente pesada. La migración es, por diseño, **más lenta y más grande** a
cambio de **más resiliencia de canal**.

## Impacto en el hot path (`recall()` / carga de historial)

- **Modo `selective` (default):** `recall()` descifra ~`top_k` (12) memorias/turno ⇒ **~64 ms/turno**
  de FEC-decrypt. Despreciable frente a la latencia de un turno LLM (segundos).
- **Modo `load_all` (control del benchmark, no default):** cargar N mensajes cuesta N × 5.3 ms ⇒
  **~5.3 s por 1000 mensajes**. Es la razón adicional por la que `selective` es el default.
- **Carga de una sesión** (`get_messages`): decenas de mensajes ⇒ cientos de ms. Perceptible pero
  acotado.

## Palancas si el costo resultara inaceptable (fuera de alcance de MS1)

La FEC es una **estrategia inyectable** (`ErrorCorrection`). Si en el futuro el costo pesara, el crate
0.3.0 ofrece `fec::ReedSolomonCodec` (RS-only, ~1.14×, sin Viterbi — como el crypto viejo) o `NoFec`
(1×, solo AEAD). **Cambiar la FEC cambia el formato en disco** (D-V02) ⇒ requeriría migración/reset,
por eso no es un cambio trivial. Registrado como palanca conocida, no como acción de MS1.
