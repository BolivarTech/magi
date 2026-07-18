// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Resolución de parámetros de una corrida headless: precedencia por campo y el
//! clamp de costo del operador (REQ-H12/H12b).
//!
//! [`resolve`] combina tres fuentes con precedencia **flag CLI > envelope >
//! defaults (toml/built-in)** para las preferencias (`model`/`provider`/
//! `consult`), y aplica dos límites de **seguridad**:
//! - `max_tool_calls` es un **techo de costo**: el valor pedido se clampea a
//!   `operator_ceiling` (el envelope puede pedir menos, nunca más). El techo lo
//!   computa el *caller* (bin), no esta función.
//! - `system` es un **límite de prompt-injection**: el `system` del envelope se
//!   ignora salvo que el operador lo habilite con `allow_system_override`.
//!
//! La función es **pura y sin dependencias del bin** (toma structs lib-locales,
//! no `MagiConfig`/`Args`), por lo que se testea en aislamiento (R-H05).
//!
//! Visibilidad `pub(crate)`: [`Resolved`] embebe [`SystemPolicy`]/[`AppliedCaps`]
//! de `types.rs`, que T0 mantiene `pub(crate)` (aún no congelados como API
//! pública). Para no exponer un tipo más privado a través de uno público
//! (`private_interfaces`), toda la superficie de este módulo comparte esa
//! visibilidad; MS2 la ensancha (o la cablea desde el lib) cuando conecte el
//! `Agent`. El `allow(dead_code)` de módulo cubre ese hueco temporal (mismo
//! scaffolding intencional que `types.rs`, no un símbolo huérfano fabricado).
#![allow(dead_code)]

use super::input::Envelope;
use super::types::{AppliedCaps, SystemPolicy};

/// Defaults derivados de `magi.toml` (más el built-in) que el bin rellena.
///
/// Los campos ya-resueltos (`model`/`provider`/`system`) son `String` porque el
/// bin colapsa el fallback toml→built-in antes de construir este struct; los
/// opcionales quedan `None` cuando ni el toml ni el built-in fijan un valor.
#[derive(Debug, Clone)]
pub(crate) struct ConfigDefaults {
    /// Modelo LLM por defecto (toml o built-in ya colapsados).
    pub model: String,
    /// Proveedor LLM por defecto (toml o built-in ya colapsados).
    pub provider: String,
    /// Tope de llamadas a tools del toml, si lo fija; participa del techo.
    pub max_tool_calls: Option<u32>,
    /// Preferencia de `consult` del toml, si la fija.
    pub consult: Option<bool>,
    /// System-prompt del operador (rige salvo override habilitado).
    pub system: String,
}

/// Overrides provenientes de flags CLI del operador (ganan sobre el envelope).
///
/// Todo `None` significa "el operador no pasó ese flag": la resolución cae al
/// envelope y luego a los defaults.
#[derive(Debug, Clone, Default)]
pub(crate) struct CliOverrides {
    /// Modelo forzado por flag, si se pasó.
    pub model: Option<String>,
    /// Proveedor forzado por flag, si se pasó.
    pub provider: Option<String>,
    /// Tope de llamadas a tools forzado por flag, si se pasó.
    pub max_tool_calls: Option<u32>,
    /// Preferencia de `consult` forzada por flag, si se pasó.
    pub consult: Option<bool>,
}

/// Parámetros efectivos de la corrida tras aplicar precedencia y clamps.
///
/// `applied_caps` hace **visible** el resultado de los límites de seguridad
/// (clamp de costo, override de system) para que el caller no falle en silencio.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    /// Modelo LLM efectivo.
    pub model: String,
    /// Proveedor LLM efectivo.
    pub provider: String,
    /// System-prompt efectivo junto con su origen (operador vs. caller).
    pub system: SystemPolicy,
    /// Tope efectivo de llamadas a tools (ya clampeado al techo del operador).
    pub max_tool_calls: u32,
    /// Preferencia de `consult` efectiva; `None` deja decidir al agente.
    pub consult: Option<bool>,
    /// Límites efectivos aplicados (clamp de costo, override de system).
    pub applied_caps: AppliedCaps,
}

/// Resuelve los parámetros efectivos de la corrida por precedencia y clamp.
///
/// Precedencia por campo para las preferencias: `overrides` (flag CLI) > `env`
/// (envelope) > `defaults` (toml/built-in). `max_tool_calls` sigue la misma
/// cadena para el *valor pedido*, luego se clampea a `operator_ceiling`
/// (`min`), marcando `max_tool_calls_clamped` si el pedido lo excedía; si nadie
/// pide un valor, el efectivo es el techo. El `system` del envelope solo se
/// honra (`SystemPolicy::CallerOverride`) si `allow_system_override` es `true`;
/// de lo contrario rige el del operador (`SystemPolicy::Operator`).
///
/// El `operator_ceiling` lo computa el caller (no se recalcula aquí);
/// `timeout_secs` queda `None` (lo fija MS2).
///
/// # Examples
///
/// Un envelope que pide `max_tool_calls: Some(999)` con `operator_ceiling = 15`
/// produce un [`Resolved`] con `max_tool_calls == 15` y
/// `applied_caps.max_tool_calls_clamped == true` (el pedido se recorta al techo).
/// Ver los tests del módulo para casos ejecutables (el tipo es `pub(crate)`, de
/// ahí que el ejemplo sea ilustrativo y no un doctest).
pub(crate) fn resolve(
    env: Envelope,
    defaults: &ConfigDefaults,
    overrides: &CliOverrides,
    operator_ceiling: u32,
    allow_system_override: bool,
) -> Resolved {
    // Preferencias: overrides (flag) > env (envelope) > defaults (toml/built-in).
    let model = overrides
        .model
        .clone()
        .or(env.model)
        .unwrap_or_else(|| defaults.model.clone());
    let provider = overrides
        .provider
        .clone()
        .or(env.provider)
        .unwrap_or_else(|| defaults.provider.clone());
    let consult = overrides.consult.or(env.consult).or(defaults.consult);

    // Costo (SEGURIDAD): el pedido sigue la precedencia; el efectivo se clampea
    // al techo del operador. Sin pedido ⇒ el efectivo es el techo, sin clamp.
    let requested_max = overrides
        .max_tool_calls
        .or(env.max_tool_calls)
        .or(defaults.max_tool_calls);
    let (max_tool_calls, max_tool_calls_clamped) = match requested_max {
        Some(requested) => (
            requested.min(operator_ceiling),
            requested > operator_ceiling,
        ),
        None => (operator_ceiling, false),
    };

    // System (SEGURIDAD): el del envelope solo se honra con el flag del operador.
    let (system, system_override_applied) = match (allow_system_override, env.system) {
        (true, Some(caller)) => (SystemPolicy::CallerOverride(caller), true),
        _ => (SystemPolicy::Operator(defaults.system.clone()), false),
    };

    Resolved {
        model,
        provider,
        system,
        max_tool_calls,
        consult,
        applied_caps: AppliedCaps {
            max_tool_calls,
            max_tool_calls_clamped,
            timeout_secs: None,
            system_override_applied,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope con `prompt` fijo y el resto de campos vacíos, para tests.
    fn base_env() -> Envelope {
        Envelope {
            prompt: "p".to_string(),
            system: None,
            model: None,
            provider: None,
            max_tool_calls: None,
            consult: None,
        }
    }

    /// Defaults con valores distinguibles del operador, para tests.
    fn base_defaults() -> ConfigDefaults {
        ConfigDefaults {
            model: "def-model".to_string(),
            provider: "def-provider".to_string(),
            max_tool_calls: None,
            consult: None,
            system: "operator-system".to_string(),
        }
    }

    /// Un envelope que pide MÁS que el techo del operador se clampea al techo y
    /// marca el clamp (REQ-H12b: el caller no puede subir el presupuesto).
    #[test]
    fn test_envelope_max_tool_calls_clamped_to_operator_ceiling() {
        let mut env = base_env();
        env.max_tool_calls = Some(999);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 15);
        assert!(r.applied_caps.max_tool_calls_clamped);
        assert_eq!(r.applied_caps.max_tool_calls, 15);
    }

    /// El flag CLI gana sobre el valor del envelope para las preferencias.
    #[test]
    fn test_cli_flag_wins_over_envelope() {
        let mut env = base_env();
        env.model = Some("A".to_string());
        env.provider = Some("prov-env".to_string());
        env.consult = Some(false);

        let overrides = CliOverrides {
            model: Some("B".to_string()),
            provider: Some("prov-flag".to_string()),
            consult: Some(true),
            ..CliOverrides::default()
        };

        let r = resolve(env, &base_defaults(), &overrides, 15, false);

        assert_eq!(r.model, "B");
        assert_eq!(r.provider, "prov-flag");
        assert_eq!(r.consult, Some(true));
    }

    /// Sin flag ni envelope, las preferencias caen a los defaults; con envelope
    /// (y sin flag), el envelope gana sobre los defaults.
    #[test]
    fn test_precedence_envelope_over_defaults_and_fallthrough() {
        // Nada pedido ⇒ defaults.
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            false,
        );
        assert_eq!(r.model, "def-model");
        assert_eq!(r.provider, "def-provider");
        assert_eq!(r.consult, None);

        // Envelope pedido, sin flag ⇒ envelope gana sobre defaults.
        let mut env = base_env();
        env.model = Some("env-model".to_string());
        let r2 = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);
        assert_eq!(r2.model, "env-model");
    }

    /// El `system` del envelope se IGNORA sin el flag del operador: rige el del
    /// operador y `system_override_applied` queda `false` (REQ-H12b/H37).
    #[test]
    fn test_envelope_system_ignored_without_override_flag() {
        let mut env = base_env();
        env.system = Some("caller-sys".to_string());

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert!(matches!(r.system, SystemPolicy::Operator(ref s) if s == "operator-system"));
        assert!(!r.applied_caps.system_override_applied);
    }

    /// El `system` del envelope se HONRA con el flag del operador: se convierte
    /// en `CallerOverride` y `system_override_applied` queda `true`.
    #[test]
    fn test_envelope_system_honored_with_override_flag() {
        let mut env = base_env();
        env.system = Some("caller-sys".to_string());

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, true);

        assert!(matches!(r.system, SystemPolicy::CallerOverride(ref s) if s == "caller-sys"));
        assert!(r.applied_caps.system_override_applied);
    }

    /// El flag habilitado sin un `system` en el envelope no inventa un override:
    /// rige el del operador y no se marca aplicado.
    #[test]
    fn test_override_flag_without_envelope_system_uses_operator() {
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            true,
        );

        assert!(matches!(r.system, SystemPolicy::Operator(ref s) if s == "operator-system"));
        assert!(!r.applied_caps.system_override_applied);
    }

    /// Un envelope que pide MENOS que el techo no se clampea; el efectivo es el
    /// pedido (borde de la regla de costo).
    #[test]
    fn test_envelope_max_tool_calls_below_ceiling_not_clamped() {
        let mut env = base_env();
        env.max_tool_calls = Some(5);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 5);
        assert!(!r.applied_caps.max_tool_calls_clamped);
    }

    /// Sin pedido de `max_tool_calls` en ninguna fuente, el efectivo cae al
    /// techo del operador y no se marca clamp.
    #[test]
    fn test_max_tool_calls_falls_through_to_operator_ceiling() {
        let r = resolve(
            base_env(),
            &base_defaults(),
            &CliOverrides::default(),
            15,
            false,
        );

        assert_eq!(r.max_tool_calls, 15);
        assert!(!r.applied_caps.max_tool_calls_clamped);
        assert_eq!(r.applied_caps.timeout_secs, None);
    }

    /// Un pedido EXACTAMENTE igual al techo no se clampea (`>` estricto).
    #[test]
    fn test_max_tool_calls_equal_to_ceiling_not_clamped() {
        let mut env = base_env();
        env.max_tool_calls = Some(15);

        let r = resolve(env, &base_defaults(), &CliOverrides::default(), 15, false);

        assert_eq!(r.max_tool_calls, 15);
        assert!(!r.applied_caps.max_tool_calls_clamped);
    }
}
