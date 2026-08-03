// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-03

//! Gate de complejidad: predicado puro que decide si un consult **autorruteado** por el
//! agente amerita despachar el consenso de tres perspectivas (REQ-A20).
//!
//! Tres propiedades que gobiernan este módulo y que son fáciles de invertir por error:
//!
//! - **Solo el ruteo AUTÓNOMO se evalúa.** `/consult` en la TUI y `magi consult` explícitos
//!   NUNCA se vetan — el gate ve la decisión del agente de invocar el tool por su cuenta,
//!   nada más. Esa distinción es de **call site** (dónde se invoca `evaluate`), no de un
//!   flag que este módulo pueda ver: `gate.rs` no sabe quién llamó.
//! - **Un veto NO es un error.** [`GateVerdict::Veto`] es un resultado normal del predicado,
//!   no un `Err`: el embudo del agente lo traduce en un `ToolResult` explicando por qué no
//!   se despachó, y el turno sigue.
//! - **La ausencia de configuración NO apaga el gate.** Tabla `[magi.complexity]` ausente ⇒
//!   los built-ins de [`super::GATE_CODE_REVIEW`]/[`super::GATE_DESIGN`]/
//!   [`super::GATE_ANALYSIS`] siguen aplicando. Un modo declarado en `0` es la única vía
//!   explícita de apagar **ese** modo — no los otros dos.

use magi_core::schema::Mode;

use super::{GATE_ANALYSIS, GATE_CODE_REVIEW, GATE_DESIGN};

/// Umbrales efectivos del gate, uno por modo (REQ-A20b).
///
/// Existe como tipo propio —y no como tres `usize` sueltos— porque el punto de la
/// granularidad por modo es que **apagar uno no apague los otros**: con parámetros
/// posicionales del mismo tipo, un swap silencioso en un call site rompe exactamente esa
/// propiedad sin romper la compilación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateThresholds {
    /// Umbral de `code-review`, en caracteres.
    pub code_review: usize,
    /// Umbral de `design`, en caracteres.
    pub design: usize,
    /// Umbral de `analysis` — el modo por defecto, y por eso el que NO puede quedar
    /// efectivamente apagado por accidente (SC-A20j).
    pub analysis: usize,
}

impl GateThresholds {
    /// Los built-ins de §4.9, sin config de por medio.
    #[must_use]
    pub const fn builtin() -> Self {
        Self {
            code_review: GATE_CODE_REVIEW,
            design: GATE_DESIGN,
            analysis: GATE_ANALYSIS,
        }
    }

    /// Resuelve la tabla `[magi.complexity]` contra los built-ins.
    ///
    /// **Tabla ausente ⇒ built-ins** (el gate sigue vivo: un feature de seguridad que se
    /// apaga solo por omitir una sección es un feature apagado). **Clave ausente DENTRO de
    /// una tabla presente ⇒ su built-in, no cero**: `Option::unwrap_or` por clave, nunca
    /// `Default` sobre la struct entera, que colapsaría los tres a `0` y desactivaría el
    /// gate completo con solo declarar `[magi.complexity]` vacía.
    ///
    /// **Toma piezas sueltas, NO `&ComplexityConfig`.** Este módulo vive en el lib y
    /// `ComplexityConfig` en el bin (`src/config.rs`): tomar la struct ataría un módulo
    /// puro a la forma del TOML y lo volvería incompilable desde el lib. Desarmar la tabla
    /// es trabajo de `config.rs`'s `gate_thresholds_from`, que ya la tiene en la mano.
    #[must_use]
    pub fn from_overrides(o: GateOverrides) -> Self {
        let GateOverrides {
            code_review,
            design,
            analysis,
        } = o;
        let b = Self::builtin();
        Self {
            code_review: code_review.unwrap_or(b.code_review),
            design: design.unwrap_or(b.design),
            analysis: analysis.unwrap_or(b.analysis),
        }
    }

    /// Umbral del modo pedido. `0` significa "este modo nunca se veta" (lo interpreta
    /// [`evaluate`], no esta función).
    #[must_use]
    pub const fn for_mode(&self, mode: &Mode) -> usize {
        match mode {
            Mode::CodeReview => self.code_review,
            Mode::Design => self.design,
            Mode::Analysis => self.analysis,
        }
    }
}

/// Overrides de `[magi.complexity]`, con NOMBRE por campo.
///
/// Tres posicionales del mismo tipo (`Option<usize>`, `Option<usize>`, `Option<usize>`)
/// son exactamente el swap silencioso que el rustdoc de [`GateThresholds`] condena.
#[derive(Debug, Clone, Copy, Default)]
pub struct GateOverrides {
    /// Override de `code_review`; ausente ⇒ su built-in.
    pub code_review: Option<usize>,
    /// Override de `design`; ausente ⇒ su built-in.
    pub design: Option<usize>,
    /// Override de `analysis`; ausente ⇒ su built-in.
    pub analysis: Option<usize>,
}

/// Resultado de evaluar el gate (REQ-A20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// El contenido amerita el consenso: se despacha.
    Dispatch,
    /// Por debajo del umbral de su modo: NO se lanza ninguna llamada al modelo.
    Veto {
        /// Modo con el que se evaluó, para el registro de telemetría (SC-A20h).
        mode: Mode,
    },
}

/// Evalúa si un consult **autorruteado** amerita despacharse.
///
/// **100 % puro:** sin async, sin I/O, sin llamadas al modelo. Se evalúa en el embudo del
/// agente, no dentro de `ConsultTool::execute` — ver REQ-A20 para por qué no se usa
/// `MagiBuilder::with_complexity_gate`.
///
/// Mide **caracteres**, no bytes (`content.chars().count()`, O(n) sobre el contenido, sin
/// bucles anidados): un umbral en bytes trataría distinto al mismo texto en otro idioma.
#[must_use]
pub fn evaluate(content: &str, mode: &Mode, thresholds: &GateThresholds) -> GateVerdict {
    let threshold = thresholds.for_mode(mode);
    if threshold == 0 || content.chars().count() >= threshold {
        GateVerdict::Dispatch
    } else {
        GateVerdict::Veto { mode: *mode }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A20b: umbral por modo, y el borde es el documentado (inclusive).
    #[test]
    fn thresholds_are_per_mode_and_the_boundary_is_inclusive() {
        let t = GateThresholds::builtin();
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW), &Mode::CodeReview, &t),
            GateVerdict::Dispatch,
            "justo EN el umbral pasa"
        );
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW - 1), &Mode::CodeReview, &t),
            GateVerdict::Veto {
                mode: Mode::CodeReview
            }
        );
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW), &Mode::Design, &t),
            GateVerdict::Veto { mode: Mode::Design },
            "Design exige más"
        );
    }

    /// SC-A20j: el gate CUBRE `Analysis`, el modo por defecto de toda invocación sin uno
    /// declarado — un umbral de 1 lo apagaría justo en el camino autónomo más común.
    #[test]
    fn the_gate_covers_analysis_the_default_mode() {
        let t = GateThresholds::builtin();
        // `const` y no `assert!` suelto: es una comparación entre dos constantes, así que
        // clippy (`assertions_on_constants`) exige evaluarla en compilación — misma forma
        // que ya usa `mod.rs`'s `plan_values_fall_inside_their_documented_ranges`.
        const {
            assert!(
                GATE_ANALYSIS > 1,
                "un umbral de 1 apagaría el gate justo en el camino autónomo más común"
            );
        }
        assert_eq!(
            evaluate("trivial", &Mode::Analysis, &t),
            GateVerdict::Veto {
                mode: Mode::Analysis
            }
        );
    }

    /// SC-A20d: sin tabla, aplican los built-ins; `0` apaga SOLO ese modo.
    #[test]
    fn an_absent_table_keeps_the_gate_alive_and_zero_disables_one_mode() {
        let t = GateThresholds::from_overrides(GateOverrides::default());
        assert_eq!(
            t,
            GateThresholds::builtin(),
            "tabla ausente ⇒ built-ins: el gate no se apaga por omitir una sección"
        );

        let t = GateThresholds::from_overrides(GateOverrides {
            code_review: Some(0),
            ..GateOverrides::default()
        });
        assert_eq!(evaluate("x", &Mode::CodeReview, &t), GateVerdict::Dispatch);
        assert_eq!(
            evaluate("x", &Mode::Design, &t),
            GateVerdict::Veto { mode: Mode::Design },
            "apagar un modo no apaga los otros: sin granularidad, la única salida sería \
             poner los tres en cero, y un mecanismo que solo sabe apagarse termina apagado"
        );
    }

    /// SC-A20h: `NoGateTelemetry` es el sink por defecto — un no-op verificable,
    /// no una promesa de trait sin implementación de referencia.
    #[test]
    fn no_gate_telemetry_is_a_silent_no_op() {
        let sink = NoGateTelemetry;
        sink.on_gate_evaluation(&Mode::Analysis, 42, GATE_ANALYSIS, true);
        sink.on_gate_evaluation(&Mode::Design, 0, GATE_DESIGN, false);
    }

    // NOTA HONESTA: la propiedad de REQ-A20 "el gate ve el ruteo AUTÓNOMO
    // (`ToolUse`) y NO la inyección forzada (`authorize_and_execute_tool`)" es
    // DELIBERADAMENTE intestable desde este módulo. Los dos call sites viven en
    // `agent/mod.rs`, no acá — `gate.rs` solo expone `evaluate`, que no sabe ni
    // puede saber quién la invocó. Un test escrito acá solo puede SIMULAR los
    // dos call sites con literales, y una simulación de eso es exactamente lo
    // que hace falso el test: hace pasar "el gate corre en el sitio correcto"
    // por "escribí `true` en un lado y llamé `evaluate` en el otro", que no
    // ejercita ninguna distinción real. El test real —el que efectivamente
    // dispara `authorize_and_execute_tool` y el bucle de `ToolUse` contra un
    // `Agent` de verdad— es obligación de Task 3.2, que es quien cablea el gate
    // dentro de esos dos call sites.
}
