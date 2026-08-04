// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Subsistema MAGI de magi-rs: resolución de modo, gate de complejidad y probe.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

pub mod endpoint;
pub mod gate;
pub mod kind;
pub mod mode;
pub mod probe;
pub mod report_anchors;

use std::time::Duration;

/// Extremos del rango admisible del techo por mage (§4.9 de la spec).
// `pub`, no privadas: las consume `validate_agent_timeout` desde `config.rs` (bin) y el
// barrido de invariante desde `tests/` — dos crates distintos, así que privadas no compilan
// en ninguno de los dos. El rango de §4.9 es contrato, no detalle interno.
pub const AGENT_TIMEOUT_MIN_SECS: u64 = 30;
/// Ver [`AGENT_TIMEOUT_MIN_SECS`].
pub const AGENT_TIMEOUT_MAX_SECS: u64 = 120;

/// Techo POR MAGE y POR INTENTO (REQ-A04, verificado contra `orchestrator.rs`).
///
/// 90 s: suficiente para una generación legítima de un modelo cloud con cold-load,
/// y deja el peor caso por mage (2 intentos) en 180 s. El default de magi-core (300)
/// es demasiado alto: vuelve inalcanzable la cadena de retry.
pub const AGENT_TIMEOUT_SECS: u64 = 90;

/// Fracción del techo para el presupuesto total de reintentos.
///
/// 0.6 + 0.3 = 0.9 < 1.0: deja 10 % de margen para que el abandono sea TIPADO
/// (`OperationBudgetExhausted`) y no un corte opaco del techo externo.
const OPERATION_BUDGET_FRACTION: f64 = 0.6;
/// Fracción del techo para el timeout de UNA petición HTTP. Ver [`OPERATION_BUDGET_FRACTION`].
const CLIENT_TIMEOUT_FRACTION: f64 = 0.3;

/// Pisos absolutos: por debajo, ninguna petición real completa.
const MIN_OPERATION_BUDGET: Duration = Duration::from_secs(10);
/// Ver [`MIN_OPERATION_BUDGET`].
const MIN_CLIENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Techo más chico para el que la escala derivada **todavía cumple** REQ-A04.
///
/// # El agujero que esta constante cierra
///
/// Los dos pisos de arriba son `.max()`, o sea que **ganan** cuando la fracción queda por
/// debajo. Con un techo de 10 s la derivación da `max(6,10) + max(3,5) = 15 > 10`: el
/// invariante que REQ-A04 declara *"imposible de romper por construcción"* queda roto —
/// **y era alcanzable desde `magi.toml`**, porque `agent_timeout_secs` no se validaba.
///
/// Peor: el barrido del invariante corría de 30 a 120, así que **nunca cruzaba el punto de
/// quiebre**. El test no fallaba porque no miraba. Un guardián que solo recorre el rango
/// feliz certifica el rango feliz, no el invariante.
///
/// La suma de los pisos ES el punto de quiebre, así que se **deriva** de ellos en vez de
/// escribirse a mano: mover un piso sin mover esto reabriría el agujero en silencio.
///
/// `pub` por la misma razón que [`AGENT_TIMEOUT_MIN_SECS`]: el rustdoc de
/// `MagiConfig::validate_agent_timeout` lo enlaza desde el **bin**, y un intra-doc link a un
/// símbolo privado de otro crate no resuelve. Es el punto de quiebre documentado de la
/// derivación, o sea contrato del módulo — no un detalle interno.
pub const AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS: u64 =
    MIN_OPERATION_BUDGET.as_secs() + MIN_CLIENT_TIMEOUT.as_secs();

/// Presupuesto total de reintentos, DERIVADO del techo (REQ-A04).
///
/// La derivación es lo que hace imposible configurar una escala inválida: no existe
/// combinación que rompa `operation_budget + client_timeout <= techo`.
#[must_use]
pub fn derive_operation_budget(ceiling_secs: u64) -> Duration {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let derived = Duration::from_secs((ceiling_secs as f64 * OPERATION_BUDGET_FRACTION) as u64);
    derived.max(MIN_OPERATION_BUDGET)
}

/// Timeout de UNA petición HTTP, DERIVADO del techo. Ver [`derive_operation_budget`].
#[must_use]
pub fn derive_client_timeout(ceiling_secs: u64) -> Duration {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let derived = Duration::from_secs((ceiling_secs as f64 * CLIENT_TIMEOUT_FRACTION) as u64);
    derived.max(MIN_CLIENT_TIMEOUT)
}

/// Techo de la llamada de clasificación (REQ-A07c).
///
/// 6 s: es UNA etiqueta, no una generación. Un techo generoso acá anula el beneficio
/// de la ruta barata. En providers lentos expira y cae a `Analysis`/`Default` — es
/// best-effort declarado, y `default_mode` es la salida sin latencia.
pub const CLASSIFY_TIMEOUT_SECS: u64 = 6;

/// Techo de UNA sonda del probe (REQ-A24). Por sonda, NO plazo compartido.
///
/// 5 s: es una petición HTTP a un endpoint típicamente local, y el arranque NO depende
/// de su resultado. Se dimensiona por cuánto es tolerable esperar de más al arrancar,
/// no por el peor caso. Muy por debajo de los 30 s de `DEFAULT_PREFLIGHT_TIMEOUT`.
pub const PROBE_TIMEOUT_SECS: u64 = 5;

/// Cap de entrada DE MAGI-RS, previo a magi-core (REQ-A11b).
///
/// 256 KiB. El criterio es COSTO, no capacidad: el payload va a los tres mages, así que
/// se paga por tres. Elegido a mano dentro del rango 256 KiB–1 MiB, holgadamente bajo
/// los 4 MiB de `max_input_len` para que el de magi-core nunca muerda.
pub const MAX_QUERY_BYTES: usize = 256 * 1024;

/// Fracción de la ventana medida para derivar `input_warn_tokens` (REQ-A24b).
///
/// 0.75: el aviso debe llegar ANTES de acercarse al límite, así que un umbral EN el
/// límite no avisa nada. Nunca 1.0 — desactivaría el guardarraíl en modelos grandes.
pub const WARN_WINDOW_FRACTION: f64 = 0.75;

/// Piso del rango cerrado de una ventana aceptable del probe (REQ-A16b).
///
/// Fuera de rango degrada a *no medido*, NUNCA se recorta al extremo: un valor recortado
/// se usa como si fuera real. El máximo cubre los modelos de contexto grande conocidos.
pub const PROBE_WINDOW_MIN: usize = 2_048;
/// Techo del rango. Ver [`PROBE_WINDOW_MIN`].
pub const PROBE_WINDOW_MAX: usize = 2_000_000;

/// Ratio que dispara el notice de composición staleness × ventana (SC-A24i).
///
/// 0.8: con el cap **convertido a tokens** por encima del 80 % de la ventana medida, el
/// margen es tan chico que un cambio a un modelo menor lo cruza y el aviso de tamaño se
/// apaga solo.
pub const STALE_NOTICE_RATIO: f64 = 0.8;

// CALIBRACIÓN, verificada contra los defaults que shipeamos — no un número suelto.
//
// El notice compara `bytes_to_tokens_est(MAX_QUERY_BYTES)` contra la ventana MEDIDA. Con el
// valor anterior (512 KiB ⇒ ~131 k tokens estimados) y un mage de ventana 128 k, el ratio
// daba 1.0 y **el notice salía en CADA arranque de la configuración por defecto** — un aviso
// que aparece siempre deja de leerse, que es peor que no tenerlo.
//
// Con 256 KiB (~65 k tokens) contra 128 k el ratio es 0.50, holgadamente bajo el umbral: el
// notice vuelve a significar lo que dice, "esta configuración está apretada".
//
// 256 KiB sigue adentro del rango de §4.9 (256 KB – 1 MB), sigue siendo "un diff de review
// real" y **abarata el cap ×3** que es el criterio de costo de REQ-A11b.

/// Caracteres por token del estimador compartido del proyecto.
///
/// **Existe porque `max_query_bytes` y la ventana medida están en UNIDADES DISTINTAS** —
/// bytes contra tokens— y compararlos directo no compara nada: el notice de SC-A24i
/// saldría o no por accidente aritmético. Es el mismo valor que ya usa `[memory]`, y es
/// una **aproximación declarada**, no una medición: el notice lo nombra.
pub const CHARS_PER_TOKEN_EST: f64 = 4.0;

/// Convierte un cap en bytes a tokens estimados, para poder compararlo con una ventana.
///
/// Redondea **hacia arriba**, que es la dirección segura: sobreestimar el tamaño del
/// payload hace que el notice salga de más, no de menos.
#[must_use]
pub fn bytes_to_tokens_est(bytes: usize) -> usize {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let tokens = (bytes as f64 / CHARS_PER_TOKEN_EST).ceil() as usize;
    tokens
}

/// Holgura del `--timeout` headless, en porcentaje del término mayor de la fórmula (§4.9).
pub const HEADLESS_TIMEOUT_HOLGURA_PCT: u64 = 20;

/// Mínimo de wall-clock para una corrida que lanza consult, **derivado en RUNTIME del
/// techo CONFIGURADO** (REQ-A04).
///
/// **NO es una `const`, y esa es la corrección.** Una constante se calcula sobre
/// [`AGENT_TIMEOUT_SECS`] —el default built-in— mientras `[magi].agent_timeout_secs` es
/// **configurable**: un operador que lo suba a 120 dejaría el mínimo calculado sobre 90 y
/// la relación se rompería **en runtime**, que es exactamente el modo de fallo que REQ-A04
/// existe para eliminar. Derivar por construcción no sirve si se deriva del valor
/// equivocado.
///
/// El valor efectivo lo entrega `MagiConfig::effective_agent_timeout_secs()` (contrato), que
/// es lo que resuelve la precedencia entre lo declarado y el default. Las dos capas internas
/// se DERIVAN de ese número, nunca se configuran (REQ-A04).
///
/// La fórmula es `clasificación + 2 × techo + holgura`. **NO se multiplica por 3**: los
/// mages corren en paralelo (verificado, SC-A04e), así que el peor caso es el del mage más
/// lento, no la suma de los tres.
#[must_use]
pub fn headless_consult_timeout_secs(configured_ceiling: u64) -> u64 {
    let dominant = 2 * configured_ceiling; // el término mayor de la fórmula
    let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
    // §4.9: la holgura es 10–30 % del TÉRMINO MAYOR, no del total — sobre el total se
    // infla proporcionalmente al término chico, que no es el que domina el riesgo.
    minimum + dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100
}

/// Decisión de wall-clock de una corrida, con su aviso si corresponde (SC-A04d).
pub struct TimeoutDecision {
    /// Segundos efectivos: lo que el operador pidió, o el default derivado.
    pub effective_secs: u64,
    /// Aviso cuando lo pedido queda bajo el mínimo de la fórmula.
    pub warning: Option<String>,
    /// Va al JSON de la corrida (REQ-A11d).
    pub below_formula: bool,
}

/// Resuelve el wall-clock de la corrida. **Obedece siempre lo explícito**, y avisa cuando
/// ese valor hace imposible completar un consult con reintento de schema.
#[must_use]
pub fn resolve_run_timeout(asked: Option<u64>, configured_ceiling: u64) -> TimeoutDecision {
    let minimum = headless_consult_timeout_secs(configured_ceiling);
    let Some(secs) = asked else {
        return TimeoutDecision {
            effective_secs: minimum,
            warning: None,
            below_formula: false,
        };
    };
    let below = secs < minimum;
    TimeoutDecision {
        effective_secs: secs,
        warning: below.then(|| {
            format!(
                "warning: --timeout {secs}s está por debajo de los {minimum}s que exige la \
                 escala para `agent_timeout_secs = {configured_ceiling}`; un consult que \
                 necesite su reintento de schema NO va a completar. Se usa el valor pedido \
                 igual."
            )
        }),
        below_formula: below,
    }
}

/// Umbral built-in del gate para `CodeReview` (REQ-A20).
///
/// 200 caracteres. Procedencia: el ejemplo del rustdoc de
/// `MagiBuilder::with_complexity_gate`. **NO está calibrado empíricamente** — es el punto
/// de partida del autor de la librería, no una medición. La telemetría de REQ-A20 permite
/// ajustarlo con datos.
pub const GATE_CODE_REVIEW: usize = 200;
/// Umbral built-in del gate para `Design`. Ver [`GATE_CODE_REVIEW`]: misma procedencia,
/// mismo estado de calibración.
pub const GATE_DESIGN: usize = 500;
/// Umbral built-in del gate para `Analysis` (REQ-A20).
///
/// **NO hereda el "no vacío" del ejemplo de magi-core, y esa desviación es el punto:**
/// `Analysis` es el default de toda invocación sin modo, así que un umbral de 1 apagaría el
/// gate en el camino autónomo más común. El gate de magi-core protege a cualquier
/// consumidor; el nuestro solo ve ruteo autónomo, donde vetar es el trabajo.
///
/// # Por qué 200 y no 150
///
/// La primera versión puso 150, que lo dejaba como **el umbral más bajo de los tres** — o
/// sea el modo que MENOS se veta. Eso invierte el argumento de arriba: el razonamiento dice
/// *"es el camino que el gate más necesita cubrir"* y el número lo hacía el más permisivo.
/// Un umbral no puede contradecir el rustdoc que lo justifica.
///
/// Queda **igual que [`GATE_CODE_REVIEW`], no por encima**. Empatarlo con el más exigente
/// sería la otra sobrecorrección: `Analysis` es la lente más ancha —cae acá toda pregunta
/// general— y un umbral tipo [`GATE_DESIGN`] vetaría consultas legítimas por ser cortas.
/// `Design` sigue siendo el más alto porque una deliberación de arquitectura que se puede
/// plantear en 300 caracteres casi nunca necesita tres perspectivas.
///
/// Como los otros dos: **no está calibrado empíricamente**. La telemetría de SC-A20h existe
/// para que la próxima elección tenga datos en vez de otra corazonada.
pub const GATE_ANALYSIS: usize = 200;

/// Los dos saltos de línea que separan la marca del texto conservado.
///
/// Constante y no un `2` suelto (B4): el número sale de la forma en que el llamador pega la
/// marca, y escribirlo a mano lo desacopla de esa forma en silencio.
pub const TRUNCATION_SEPARATOR_LEN: usize = 2;

/// Cap de SALIDA del reporte por defecto, en las tres rutas (REQ-A11b).
///
/// Nace en Fase 0 y no en Fase 6 por la misma razón que los otros tres símbolos del
/// recorte: `effective_tool_result_cap` lo consume en **Fase 1**.
///
/// El criterio del número es el mismo que el del cap de entrada —COSTO— pero la cuenta es
/// al revés: la entrada se paga una vez por tres mages, la salida se paga una vez por cada
/// turno restante de la sesión, porque vive en el historial.
pub const TOOL_RESULT_CAP_BYTES: usize = 64 * 1024;

/// Marca que se agrega a un reporte recortado. **Un recorte silencioso es indistinguible de
/// un reporte completo**, y esa es toda la razón de que exista.
pub const TRUNCATION_MARK: &str = "[reporte recortado por límite de tamaño]";

/// Bytes que la marca agrega, para que cada nivel descuente su propio presupuesto.
///
/// Se DERIVA de la constante en vez de escribirse a mano: cambiar el texto de la marca sin
/// mover este número volvería a desbordar el cap, en silencio y solo en el borde.
#[must_use]
pub fn mark_overhead() -> usize {
    TRUNCATION_MARK.len() + TRUNCATION_SEPARATOR_LEN
}

/// Cap de salida mínimo viable: por debajo, ni la marca de recorte entra.
///
/// Con un cap menor que [`mark_overhead`], los tres niveles hacen `checked_sub` → `None` y
/// el recorte **no aplica nada**: el reporte sale entero, o sea que el cap configurado se
/// ignora en silencio. Un límite que deja de aplicarse cuando lo apretás es peor que no
/// tenerlo.
#[must_use]
pub fn min_viable_output_cap() -> usize {
    mark_overhead() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A04 / REQ-A04: la escala se cumple POR CONSTRUCCIÓN para cualquier techo del rango.
    #[test]
    fn derived_scale_satisfies_invariant_across_the_whole_admissible_range() {
        // Arranca en el PISO ABSOLUTO, no en el mínimo de §4.9: el punto de quiebre está
        // por debajo del rango configurable, y un barrido que no lo cruza no prueba nada.
        for ceiling in AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS..=AGENT_TIMEOUT_MAX_SECS {
            let budget = derive_operation_budget(ceiling);
            let client = derive_client_timeout(ceiling);
            assert!(
                budget + client <= Duration::from_secs(ceiling),
                "techo {ceiling}s: {budget:?} + {client:?} excede el techo",
            );
            assert!(
                budget >= MIN_OPERATION_BUDGET,
                "techo {ceiling}s cae bajo el piso"
            );
            assert!(
                client >= MIN_CLIENT_TIMEOUT,
                "techo {ceiling}s cae bajo el piso"
            );
        }
    }

    /// REQ-A04: el --timeout headless cubre clasificación + 2 intentos + holgura.
    ///
    /// Se DERIVA, no se hardcodea: un literal se desincroniza en silencio en cuanto alguien
    /// mueve `AGENT_TIMEOUT_SECS`, y este test lo detectaría recién en el commit siguiente.
    #[test]
    fn headless_timeout_default_covers_classification_and_two_attempts() {
        let dominant = 2 * AGENT_TIMEOUT_SECS;
        let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
        let holgura = dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100;
        assert!(
            headless_consult_timeout_secs(AGENT_TIMEOUT_SECS) >= minimum + holgura,
            "el default headless no cubre {minimum}s + holgura",
        );
    }

    /// SC-A04c — el `--timeout` headless respeta la fórmula PARA EL TECHO CONFIGURADO.
    ///
    /// Es el mismo bug que la función reemplaza, nombrado por su escenario: una `const` se
    /// ata al default built-in, no al valor que el operador puso en `[magi]`.
    #[test]
    fn a_raised_ceiling_raises_the_headless_minimum_too() {
        for ceiling in AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS {
            let dominant = 2 * ceiling;
            let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
            let holgura = dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100;
            assert!(
                headless_consult_timeout_secs(ceiling) >= minimum + holgura,
                "techo {ceiling}s: el mínimo headless no cubre la fórmula"
            );
        }
        assert!(
            headless_consult_timeout_secs(120) > headless_consult_timeout_secs(90),
            "subir `agent_timeout_secs` DEBE subir el mínimo; una const no lo haría"
        );
    }

    /// SC-A04d: un `--timeout` explícito por debajo del mínimo se OBEDECE, con aviso.
    ///
    /// La bandera es un tope de wall-clock del operador, no un invariante de seguridad:
    /// quien pide `--timeout 5` quiere cortar a los 5 segundos, y forzarlo a respetar la
    /// fórmula sería desobedecer una orden clara. Pero un valor bajo el mínimo garantiza
    /// que **ningún consult con reintento de schema completa**, y eso no es obvio desde la
    /// línea de comandos.
    #[test]
    fn an_explicit_timeout_below_the_formula_is_obeyed_and_warned_about() {
        let asked = 5_u64;
        let decision = resolve_run_timeout(Some(asked), AGENT_TIMEOUT_SECS);
        assert_eq!(
            decision.effective_secs, asked,
            "se obedece la orden del operador"
        );
        let warning = decision
            .warning
            .expect("un valor bajo el mínimo debe avisar");
        assert!(
            warning.contains(&headless_consult_timeout_secs(AGENT_TIMEOUT_SECS).to_string()),
            "el aviso nombra el mínimo que la fórmula pedía"
        );
        assert!(
            decision.below_formula,
            "y viaja al JSON: quien usa la bandera corre en pipeline, o sea que menos lee stderr"
        );

        assert!(
            resolve_run_timeout(None, AGENT_TIMEOUT_SECS)
                .warning
                .is_none(),
            "el default no avisa de sí mismo"
        );
        assert!(resolve_run_timeout(Some(1_000), AGENT_TIMEOUT_SECS)
            .warning
            .is_none());
    }

    /// §4.9: cada valor cae dentro de su rango admisible.
    #[test]
    fn plan_values_fall_inside_their_documented_ranges() {
        // Las dos mitades de la defensa, juntas para que se lean como una sola cosa:
        // la validación de carga impide entrar por debajo del rango, y el piso absoluto
        // queda holgadamente por debajo de ese rango — o sea que la derivación nunca ve
        // un techo donde los pisos ganen.
        //
        // `const` y no `assert!` suelto: las tres comparaciones de este test que son entre
        // constantes se evalúan EN COMPILACIÓN, así que violarlas rompe el build en vez de
        // un test. Es la garantía más fuerte disponible y es lo que clippy pide con
        // `assertions_on_constants` — la alternativa era un `#[allow]`, que solo apaga el
        // aviso y deja la comprobación donde estaba.
        const {
            assert!(
                AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS < AGENT_TIMEOUT_MIN_SECS,
                "el rango configurable DEBE quedar por encima del punto de quiebre; si un \
                 día no lo estuviera, la validación de carga dejaría pasar un techo que \
                 rompe el invariante y ningún otro test lo notaría"
            );
        }
        assert!(
            (AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS).contains(&AGENT_TIMEOUT_SECS),
            "el rango sale de §4.9, no de literales repetidos: con `30..=120` escrito a \
             mano acá Y en el barrido de arriba, mover el rango deja los dos en \
             desacuerdo, y el que falla es el que nadie mira"
        );
        assert!((3..=10).contains(&CLASSIFY_TIMEOUT_SECS));
        assert!((3..=10).contains(&PROBE_TIMEOUT_SECS));
        assert!((256 * 1024..=1024 * 1024).contains(&MAX_QUERY_BYTES));
        const {
            assert!(WARN_WINDOW_FRACTION > 0.0 && WARN_WINDOW_FRACTION < 1.0);
        }
        assert!((10..=30).contains(&HEADLESS_TIMEOUT_HOLGURA_PCT));
        const {
            assert!(PROBE_WINDOW_MIN < PROBE_WINDOW_MAX);
        }
        for t in [GATE_CODE_REVIEW, GATE_DESIGN, GATE_ANALYSIS] {
            assert!(t > 1, "un umbral de 1 apaga el gate en ese modo (REQ-A20)");
        }
    }

    /// SC-A24i: el estimador redondea HACIA ARRIBA, que es la dirección segura.
    ///
    /// Sobreestimar el payload hace que el notice salga de más, nunca de menos — y de menos
    /// es el modo de fallo que importa, porque apaga un aviso en silencio.
    #[test]
    fn the_token_estimator_rounds_up_and_handles_the_empty_case() {
        assert_eq!(
            bytes_to_tokens_est(0),
            0,
            "un payload vacío no estima tokens"
        );
        assert_eq!(
            bytes_to_tokens_est(1),
            1,
            "un byte suelto redondea a un token, no a cero"
        );
        assert_eq!(bytes_to_tokens_est(4), 1, "el caso exacto no infla");
        assert_eq!(
            bytes_to_tokens_est(5),
            2,
            "hacia arriba en cuanto sobra un byte"
        );
    }

    /// El overhead se DERIVA del texto de la marca en vez de escribirse a mano.
    ///
    /// Un número escrito a mano se desincroniza al editar la marca, y el desborde
    /// resultante aparece solo en el borde del cap — o sea casi nunca, y sin diagnóstico.
    #[test]
    fn the_truncation_overhead_is_derived_from_the_mark_text() {
        assert_eq!(
            mark_overhead(),
            TRUNCATION_MARK.len() + TRUNCATION_SEPARATOR_LEN
        );
        assert!(
            mark_overhead() > TRUNCATION_SEPARATOR_LEN,
            "la marca aporta su propio texto"
        );
    }

    /// Un cap por debajo del overhead haría que el recorte deje de aplicarse EN SILENCIO.
    ///
    /// Con `cap <= mark_overhead()` los tres niveles hacen `checked_sub` → `None` y el
    /// reporte sale entero: un límite que se ignora cuando lo apretás es peor que ninguno.
    #[test]
    fn the_minimum_viable_cap_leaves_room_for_the_mark_itself() {
        assert!(
            min_viable_output_cap() > mark_overhead(),
            "por debajo del overhead el recorte no aplica nada y el cap se ignora en silencio"
        );
        assert!(
            TOOL_RESULT_CAP_BYTES > min_viable_output_cap(),
            "el default built-in debe quedar holgadamente por encima del mínimo viable"
        );
    }
}
