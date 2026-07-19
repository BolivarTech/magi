// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Política de autorización de tools por tier headless (REQ-H06/H07/H08/H09).
//!
//! [`Policy`] traduce el tier headless (`default`/`--auto`/`--full-auto`) a una
//! decisión de aprobación por tool: **fail-closed** — un tool no reconocido
//! nunca se auto-aprueba, en ningún tier. El módulo es lógica **pura**, sin
//! dependencia del `Agent` ni de los tools (`headless` es un módulo de la
//! biblioteca [`crate`], mientras `src/tools/` vive solo en el binario) — el
//! runner (tarea posterior de MS2) cablea esta decisión al `approval_tx` del
//! agente.
//!
//! **Esta política nunca toca las barreras DURAS** (`bash::is_command_allowed`,
//! la prohibición de metacaracteres, `PathGuard::validate`): esas se aplican
//! dentro de cada tool y permanecen activas sin importar el tier (REQ-H09). Lo
//! que esta política decide es exclusivamente la aprobación **suave** por tier.

use super::limits::{FULL_AUTO_MAX_TOOL_CALLS, NORMAL_MAX_TOOL_CALLS};

/// Nombres de los tools READ-ONLY — única fuente de verdad del set (REQ-H06,
/// DRY). Verificado contra el registro real de `main.rs` (`ListTool`/
/// `FileReadTool`/`GrepTool`, cuyo `Tool::name()` devuelve exactamente estos
/// tres literales — ver `src/tools/{ls,read,grep}.rs`).
pub const READ_ONLY_TOOLS: &[&str] = &["ls", "view", "grep"];

/// Nombres de los tools que mutan estado o ejecutan procesos/LLM adicionales —
/// aprobados solo en `Auto`/`FullAuto` (REQ-H07). Verificado contra el registro
/// real de `main.rs` (`FileWriteTool`/`BashTool`/`ConsultTool`/`ProjectFactTool`,
/// cuyo `Tool::name()` devuelve estos cuatro literales — ver
/// `src/tools/{write,bash,consult,knowledge}.rs`).
const READ_WRITE_TOOLS: &[&str] = &["edit", "bash", "consult", "project_knowledge"];

/// Tier de autorización de tools de una corrida headless.
///
/// Determina exclusivamente la matriz de aprobación **suave**; las barreras
/// duras de cada tool son idénticas en los tres tiers (REQ-H09).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Solo los tools de [`READ_ONLY_TOOLS`] se auto-aprueban (REQ-H06).
    Default,
    /// Todos los tools registrados se auto-aprueban, barreras duras intactas
    /// (REQ-H07).
    Auto,
    /// Como `Auto`, además eleva `max_tool_calls` y silencia las guardas
    /// suaves del agente (REQ-H08).
    FullAuto,
}

/// Política de autorización efectiva de una corrida headless.
///
/// `max_tool_calls`/`timeout` viajan aquí para que el runner (tarea posterior
/// de MS2) los consuma junto con la decisión de aprobación; ninguno de los dos
/// participa en la lógica de [`Policy::approves`], que depende solo del tier.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Tier activo; determina la matriz de aprobación y los avisos emitidos.
    tier: Tier,
    /// Tope de llamadas a tools ya resuelto por el caller (sin clamp aquí).
    max_tool_calls: u32,
    /// Timeout de wall-clock en segundos, si se fijó (REQ-H36; lo aplica T4).
    timeout: Option<u64>,
}

impl Policy {
    /// Construye una política para `tier` con los límites ya resueltos.
    ///
    /// `max_tool_calls`/`timeout` se toman tal cual del caller — esta función
    /// no aplica ningún clamp de costo (eso ya ocurrió en la resolución de
    /// parámetros, `resolution::resolve`, tarea previa de MS1).
    #[must_use]
    pub fn new(tier: Tier, max_tool_calls: u32, timeout: Option<u64>) -> Self {
        Self {
            tier,
            max_tool_calls,
            timeout,
        }
    }

    /// Tier activo de esta política.
    #[must_use]
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Tope de llamadas a tools de esta corrida.
    #[must_use]
    pub fn max_tool_calls(&self) -> u32 {
        self.max_tool_calls
    }

    /// Timeout de wall-clock en segundos, si se fijó uno.
    #[must_use]
    pub fn timeout(&self) -> Option<u64> {
        self.timeout
    }

    /// Decide si `tool_name` se auto-aprueba bajo el tier de esta política.
    ///
    /// **Fail-closed:** un nombre que no pertenece ni a [`READ_ONLY_TOOLS`] ni
    /// a los tools de lectura-escritura conocidos nunca se aprueba, en ningún
    /// tier — así un tool futuro registrado en `main.rs` sin clasificar aquí
    /// queda denegado por defecto en vez de auto-aprobado por omisión.
    ///
    /// Esta función **no** evalúa ni relaja ninguna barrera dura: la
    /// aprobación aquí es una condición necesaria pero no suficiente — el tool
    /// igual puede fallar dentro de sí mismo (`bash` allowlist, `PathGuard`).
    #[must_use]
    pub fn approves(&self, tool_name: &str) -> bool {
        let is_read_only = READ_ONLY_TOOLS.contains(&tool_name);
        match self.tier {
            Tier::Default => is_read_only,
            Tier::Auto | Tier::FullAuto => is_read_only || READ_WRITE_TOOLS.contains(&tool_name),
        }
    }

    /// Avisos a emitir (stderr + log) para esta política, al inicio de la
    /// corrida.
    ///
    /// No vacío únicamente bajo `FullAuto` (REQ-H08): la elevación de
    /// privilegios (cap elevado + guardas suaves silenciadas) nunca es
    /// silenciosa. `Default`/`Auto` no elevan nada y no emiten aviso.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        match self.tier {
            Tier::FullAuto => vec![format!(
                "WARNING: --full-auto is active — max_tool_calls is elevated \
                 to {FULL_AUTO_MAX_TOOL_CALLS} (from the normal \
                 {NORMAL_MAX_TOOL_CALLS}), and the repetitive-call soft guard \
                 is silenced. Hard barriers (bash allowlist, metacharacter \
                 ban, PathGuard) remain fully enforced in every tier."
            )],
            Tier::Default | Tier::Auto => Vec::new(),
        }
    }

    /// `true` si el runner debe desactivar las guardas SUAVES del `Agent`.
    ///
    /// Únicamente `FullAuto` silencia (a) la detección de 3 llamadas
    /// idénticas consecutivas y (b) el cap normal de `max_tool_calls`
    /// (reemplazado por el elevado) — REQ-H08. Esta política solo **declara**
    /// la intención: no tiene una referencia al `Agent` para aplicarla, eso
    /// lo cablea el runner. Ninguna barrera dura se ve afectada por esta señal.
    #[must_use]
    pub fn silences_soft_guards(&self) -> bool {
        matches!(self.tier, Tier::FullAuto)
    }
}

/// Fuzz entrypoint del target `fuzz_policy` (MS2 Task 10 / REQ-H35): mapea
/// bytes arbitrarios a `(tier, nombre_de_tool)` y ejercita toda la superficie
/// pública de [`Policy`].
///
/// El primer byte selecciona el tier (`0` ⇒ [`Tier::Default`], `1` ⇒
/// [`Tier::Auto`], cualquier otro ⇒ [`Tier::FullAuto`]) y el resto de los bytes
/// es el nombre del tool, convertido con `String::from_utf8_lossy` — de modo
/// que la entrada cubre nombres no-UTF8. Invariantes verificadas sobre TODA
/// entrada: **nunca panic** (la matriz es lógica pura, total), y **fail-closed**
/// — una aprobación implica que el nombre pertenece al set conocido de tools en
/// cualquier tier (un nombre desconocido jamás devuelve `true`).
///
/// `#[doc(hidden)] pub` espeja la convención de los `fuzz_*_entrypoint` del
/// vault y de [`output`](super::output): expone la frontera al crate `fuzz/`
/// sin ensanchar la API pública documentada del módulo.
///
/// # Panics
///
/// Panica (bajo `debug_assertions`, que `cargo-fuzz` activa) solo si la
/// invariante fail-closed se viola — ese es el bug genuino que el fuzzer busca,
/// no un abort espurio.
#[doc(hidden)]
pub fn fuzz_policy_entrypoint(data: &[u8]) {
    // Primer byte ⇒ tier; resto ⇒ nombre del tool (lossy, cubre no-UTF8). El
    // fallback `(&0, &[])` cubre la entrada vacía sin indexar (fail-closed).
    let (&tier_byte, name_bytes) = data.split_first().unwrap_or((&0, &[]));
    let tier = match tier_byte {
        0 => Tier::Default,
        1 => Tier::Auto,
        _ => Tier::FullAuto,
    };

    // `max_tool_calls`/`timeout` derivados de la cola para ejercitar los
    // accesores con valores variados; no participan en la lógica de aprobación.
    let max_tool_calls = u32::try_from(name_bytes.len()).unwrap_or(u32::MAX);
    let timeout = name_bytes.first().map(|&b| u64::from(b));

    let name = String::from_utf8_lossy(name_bytes);
    let policy = Policy::new(tier, max_tool_calls, timeout);

    // Toda la superficie pública debe ser total (nunca panic) sobre la entrada.
    let approved = policy.approves(&name);
    let _ = policy.silences_soft_guards();
    let _ = policy.warnings();
    let _ = policy.tier();
    let _ = policy.max_tool_calls();
    let _ = policy.timeout();

    // Fail-closed: una aprobación implica un nombre de tool conocido, en
    // cualquier tier — un nombre desconocido jamás se auto-aprueba (REQ-H09).
    let name_ref: &str = name.as_ref();
    let is_known = READ_ONLY_TOOLS.contains(&name_ref) || READ_WRITE_TOOLS.contains(&name_ref);
    debug_assert!(
        !approved || is_known,
        "fail-closed violated: approved unknown tool name {name_ref:?} in tier {tier:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Todos los tools registrados en `main.rs`, tal como los devuelve su
    /// `Tool::name()` real (verificado en `src/tools/{ls,read,write,grep,
    /// bash,consult,knowledge}.rs` y el registro de `main.rs` en la fecha de
    /// este archivo). Es la lista de referencia del test de mantenimiento de
    /// abajo: un tool nuevo que se registre en `main.rs` sin agregarse aquí
    /// (y a [`READ_ONLY_TOOLS`]/`READ_WRITE_TOOLS`) queda sin cubrir por este
    /// guard hasta que un mantenedor actualice ambas listas — el módulo
    /// `headless` es puro y no puede importar `crate::tools` para verificarlo
    /// dinámicamente (ver rustdoc de módulo).
    const REAL_REGISTERED_TOOL_NAMES: &[&str] = &[
        "ls",
        "view",
        "edit",
        "grep",
        "bash",
        "consult",
        "project_knowledge",
    ];

    /// La matriz de aprobación por tier es exhaustiva sobre los tools
    /// conocidos y fail-closed sobre cualquier nombre desconocido, en TODOS
    /// los tiers (REQ-H06/H07/H09; MS2.md Task 1 Step 1 verbatim).
    fn policy(tier: Tier) -> Policy {
        Policy::new(tier, NORMAL_MAX_TOOL_CALLS, None)
    }

    #[test]
    fn test_tier_approval_matrix_is_exhaustive_and_fail_closed() {
        let default = policy(Tier::Default);
        for ro in READ_ONLY_TOOLS {
            assert!(
                default.approves(ro),
                "{ro} debe auto-aprobarse en default (read-only)"
            );
        }
        for rw in READ_WRITE_TOOLS {
            assert!(!default.approves(rw), "{rw} NO debe aprobarse en default");
        }
        assert!(
            !default.approves("tool_que_no_existe"),
            "fail-closed: un tool desconocido nunca se aprueba en default"
        );

        for tier in [Tier::Auto, Tier::FullAuto] {
            let p = policy(tier);
            for known in READ_ONLY_TOOLS.iter().chain(READ_WRITE_TOOLS.iter()) {
                assert!(
                    p.approves(known),
                    "{known} debe aprobarse en {tier:?} (todos los registrados)"
                );
            }
            assert!(
                !p.approves("tool_que_no_existe"),
                "fail-closed: un tool desconocido nunca se aprueba, ni en {tier:?}"
            );
        }
    }

    /// El conjunto conocido de [`READ_ONLY_TOOLS`] + `READ_WRITE_TOOLS` debe
    /// coincidir exactamente (mismo tamaño y mismos elementos) con el registro
    /// real de `main.rs`, para que un tool agregado ahí y olvidado en la
    /// clasificación de esta política falle este test en vez de quedar
    /// silenciosamente denegado o aprobado por omisión.
    #[test]
    fn test_known_tool_set_matches_real_tool_registry() {
        let mut known: Vec<&str> = READ_ONLY_TOOLS
            .iter()
            .copied()
            .chain(READ_WRITE_TOOLS.iter().copied())
            .collect();
        known.sort_unstable();

        let mut real: Vec<&str> = REAL_REGISTERED_TOOL_NAMES.to_vec();
        real.sort_unstable();

        assert_eq!(
            known, real,
            "READ_ONLY_TOOLS + READ_WRITE_TOOLS debe coincidir con el registro \
             real de main.rs — actualizar ambas listas al registrar un tool nuevo"
        );
    }

    /// `silences_soft_guards` es `true` únicamente bajo `FullAuto`.
    #[test]
    fn test_silences_soft_guards_true_only_for_full_auto() {
        assert!(!policy(Tier::Default).silences_soft_guards());
        assert!(!policy(Tier::Auto).silences_soft_guards());
        assert!(policy(Tier::FullAuto).silences_soft_guards());
    }

    /// `warnings()` está vacío en `Default`/`Auto` y contiene el aviso de
    /// elevación de límites en `FullAuto` (REQ-H08).
    #[test]
    fn test_warnings_nonempty_only_for_full_auto_and_mentions_elevation() {
        assert!(policy(Tier::Default).warnings().is_empty());
        assert!(policy(Tier::Auto).warnings().is_empty());

        let warnings = policy(Tier::FullAuto).warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("--full-auto"));
        assert!(warnings[0].contains(&FULL_AUTO_MAX_TOOL_CALLS.to_string()));
        assert!(warnings[0].contains("soft guard"));
    }

    /// Borde: `approves` con una cadena vacía (nunca un nombre de tool real)
    /// se deniega en todos los tiers — fail-closed también sobre input vacío.
    #[test]
    fn test_approves_denies_empty_tool_name_in_every_tier() {
        for tier in [Tier::Default, Tier::Auto, Tier::FullAuto] {
            assert!(!policy(tier).approves(""));
        }
    }

    /// Los accesores exponen tal cual los valores pasados a `new` (sin clamp),
    /// incluido el caso borde de `timeout: None`.
    #[test]
    fn test_new_accessors_expose_constructor_values_unmodified() {
        let p = Policy::new(Tier::Auto, 42, Some(900));
        assert_eq!(p.tier(), Tier::Auto);
        assert_eq!(p.max_tool_calls(), 42);
        assert_eq!(p.timeout(), Some(900));

        let no_timeout = Policy::new(Tier::Default, 15, None);
        assert_eq!(no_timeout.timeout(), None);
    }

    /// Unit-smoke del fuzz entrypoint `fuzz_policy` (REQ-H35): entradas
    /// degeneradas (vacía, tier fuera de rango, cola no-UTF8, tool desconocido)
    /// nunca panican y respetan el fail-closed. Es la versión local que SÍ
    /// corre en cada §0.1, complementando la corrida coverage-guided de CI.
    #[test]
    fn test_fuzz_policy_entrypoint_never_panics_on_arbitrary_input() {
        let cases: &[&[u8]] = &[
            b"",
            b"\x00",
            b"\x01",
            b"\x00ls",
            b"\x01bash",
            b"\xffedit",
            b"\x00tool_que_no_existe",
            &[0x02, 0xff, 0xfe, 0xfd],
        ];
        for case in cases {
            fuzz_policy_entrypoint(case);
        }
    }
}
