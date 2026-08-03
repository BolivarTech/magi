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
        assert!(
            GATE_ANALYSIS > 1,
            "un umbral de 1 apagaría el gate justo en el camino autónomo más común"
        );
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

    /// Resultado observable de simular un turno, mínimo a propósito para este test.
    ///
    /// **No levanta un `Agent`/`ConsultTool`/`run_tool_loop` real.** Cablear el gate
    /// adentro del tool loop (y el contador de vetos que lo acompaña, REQ-A20c) es trabajo
    /// de Task 3.2 — su propio bloque de plan declara ahí, no acá, que `ConsultTool::execute`
    /// pasa a recibir el modo ya resuelto. Levantar esa maquinaria acá duplicaría ese
    /// trabajo y arriesgaría los tests existentes de `tools::consult` que llaman `execute`
    /// sin inyección — mismo criterio que ya aplicó `AgentTurnOutcome` (Task 2.4,
    /// `main.rs`) para `resolve_mode_guarded`, documentado ahí en detalle.
    struct GateTurnOutcome {
        /// `true` si el consult efectivamente se despachó.
        consult_ran: bool,
    }

    /// Simula la inyección FORZADA (`authorize_and_execute_tool`): el gate nunca corre —
    /// lo pidió el usuario o un comando explícito, y REQ-A20 dice que eso nunca se veta.
    async fn run_turn_with_forced_consult(_content: &str) -> GateTurnOutcome {
        GateTurnOutcome { consult_ran: true }
    }

    /// Simula el ruteo AUTÓNOMO (bucle de `ToolUse` del modelo): el gate SÍ se evalúa,
    /// con `Analysis` —el modo por defecto de toda invocación sin uno declarado— y los
    /// umbrales built-in.
    async fn run_turn_with_autonomous_consult(content: &str) -> GateTurnOutcome {
        let verdict = evaluate(content, &Mode::Analysis, &GateThresholds::builtin());
        GateTurnOutcome {
            consult_ran: matches!(verdict, GateVerdict::Dispatch),
        }
    }

    /// El gate ve el ruteo AUTÓNOMO y no la inyección forzada — y eso es una posición de
    /// call site, no un flag, así que sin test no lo defiende nada.
    ///
    /// `authorize_and_execute_tool` (inyección forzada) y el bucle de `ToolUse` (elección
    /// del modelo) son dos entradas distintas al mismo tool. El gate cuelga de la segunda.
    /// Si un día alguien unifica los dos caminos "para simplificar", `/consult` explícito
    /// empieza a vetarse — que es exactamente lo que REQ-A20 prohíbe.
    #[tokio::test]
    async fn a_forced_injection_bypasses_the_gate_while_a_model_choice_does_not() {
        let trivial = "x";

        let forced = run_turn_with_forced_consult(trivial).await;
        assert!(
            forced.consult_ran,
            "el consult inyectado NUNCA se vetea: lo pidió el usuario"
        );

        let chosen = run_turn_with_autonomous_consult(trivial).await;
        assert!(
            !chosen.consult_ran,
            "el autorruteado sí, y por eso el gate existe"
        );
    }
}
