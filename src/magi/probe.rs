// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-04

//! Medición de ventanas de contexto y digest del modelo, por composición sobre
//! `ProviderProbe` (REQ-A24).
//!
//! # Por composición, no por migración
//!
//! `ProviderProbe` es un trait **separado** de `LlmProvider`: se construye un
//! `OllamaProvider` de magi-core **solo** para llamarle `.window()` y `.digest()`, nunca
//! para generar. magi-rs sigue completando con su propio `Provider` — D-A07 y R-A02 quedan
//! intactos. Solo `ollama` es medible (`ProviderKind::is_probeable`); `openai-compat` y
//! `anthropic` no ofrecen introspección y degradan a [`Measurement::NotMeasurable`].
//!
//! # El cap de tamaño del cuerpo (REQ-A16b / SC-A16c) — satisfecho POR COMPOSICIÓN
//!
//! REQ-A16b exige que un cuerpo de respuesta del probe se lea bajo un cap que corte
//! **mientras lee**, no que se verifique sobre un buffer ya alojado entero. Este módulo NO
//! implementa ese cap, y no es un descuido: `OllamaProvider::window()`/`.digest()` hacen su
//! propia petición HTTP dentro de magi-core y le devuelven a este módulo un
//! `Result<Option<T>, ProviderError>` **ya resuelto** — el cuerpo crudo nunca cruza la
//! frontera de la composición, así que no hay nada que magi-rs pueda capar sin reimplementar
//! el transporte entero (lo que R-A02 prohíbe).
//!
//! magi-core ya lo hace: `read_probe_body` corta a `MAX_SHOW_BODY_BYTES` (1 MiB) **durante**
//! la lectura, no después. La propiedad está verificada, no solo leída, por
//! `tests/magi_core_contract.rs::magi_core_rejects_an_endless_probe_body_instead_of_accumulating_it`,
//! que golpea un servidor real con un cuerpo sin fin y confirma que el lector corta por
//! tamaño en vez de agotar memoria o esperar un timeout. Si esa dependencia dejara de
//! capar, ese test lo dice antes que un endpoint hostil en producción.
//!
//! Lo que sí es responsabilidad de este módulo, y está implementado acá, es **validar los
//! VALORES ya resueltos**: una ventana fuera de `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]` o un
//! digest que no son exactamente 64 hex en minúscula degradan a *no medido*, nunca se usan
//! tal cual ni se recortan al extremo del rango.
//!
//! # `ProbeError` no se define en este archivo
//!
//! El encabezado de la tarea lista `ProbeError` como símbolo a definir acá, pero ningún
//! camino de este diseño necesita un tipo de error: `probe_for` devuelve [`ProbeSeat`] (no
//! `Result`), `probe_models` devuelve un `BTreeMap` (no `Result`), y `derive_warn_tokens`
//! devuelve `Option<usize>` (no `Result`) — el probe **falla abierto en todas partes**, así
//! que no hay un canal de error que propagar. Fabricar un tipo sin ningún caller solo para
//! completar la lista habría violado la regla de no inventar símbolos sin consumidor; se
//! documenta acá en vez de crearlo en silencio.

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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use magi_core::providers::ollama::OllamaProvider;
use magi_core::rotation::ProviderProbe;

use crate::magi::endpoint::ResolvedEndpoint;
use crate::magi::kind::ProviderKind;
use crate::magi::{PROBE_TIMEOUT_SECS, PROBE_WINDOW_MAX, PROBE_WINDOW_MIN, WARN_WINDOW_FRACTION};
use crate::redact::{redact_foreign_error, SafeErrorText};

/// Longitud exacta de un digest SHA-256 en hexadecimal (REQ-A16b).
///
/// SHA-256 produce 32 bytes; en hexadecimal eso son EXACTAMENTE 64 caracteres. No es una
/// elección de diseño: es el tamaño de una huella SHA-256, y magi-core valida por el mismo
/// número en `parse_tags_digest`. Documentado en vez de repetido como literal (B4).
const DIGEST_HEX_LEN: usize = 64;

/// Resultado de medir un modelo (REQ-A24c). **Tres estados, no dos.**
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Measurement {
    /// Medido: ventana en tokens (dentro de `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]`) y digest
    /// si se pudo resolver y validar.
    Measured {
        /// Ventana de contexto, ya validada contra el rango de REQ-A16b.
        window: usize,
        /// SHA256 del manifiesto — 64 hex en minúscula — o `None` si no se pudo resolver o
        /// no pasó la validación de formato.
        digest: Option<String>,
    },
    /// El endpoint no ofrece introspección (`openai-compat`, `anthropic`). Definitivo y
    /// esperado (SC-A24b) — **no es un fallo**.
    NotMeasurable,
    /// El endpoint SÍ ofrece introspección pero esta vez no contestó a tiempo, devolvió algo
    /// fuera de rango, o la sonda no se pudo construir.
    ///
    /// Es el caso común del **primer arranque** con un daemon Ollama frío. Sin distinguirlo
    /// de [`Self::NotMeasurable`], un "ventana desconocida" en el estreno se leería como
    /// *"esto no funciona"* cuando en realidad es *"todavía no cargó"*.
    NotMeasuredThisTime,
}

/// Qué salió de intentar armar una sonda para un `(endpoint, modelo)`. **Tres estados, no
/// dos**: un `kind` no medible y una sonda que no se pudo construir tienen consecuencias
/// distintas, y colapsarlos afirmaría algo falso sobre la capacidad del servidor cuando lo
/// que falló fue nuestra configuración.
pub enum ProbeSeat {
    /// Lista para sondear.
    Ready(Arc<dyn ProviderProbe>),
    /// El `kind` no ofrece probe. Definitivo (SC-A24b) — **no es un fallo**.
    NotProbeable,
    /// El `kind` SÍ es medible pero la sonda no se pudo construir (URL malformada, cliente
    /// HTTP). Arreglable: es un problema de nuestra config, no una afirmación sobre el
    /// servidor. El texto ya viene redactado — nunca puede contener la credencial resuelta
    /// del endpoint que se intentó sondear (B11).
    Unbuildable(SafeErrorText),
}

/// Fábrica de sondas — la costura de inyección que R-A04 exige (sin red real en los tests
/// de este módulo, salvo el que verifica la construcción real contra un servidor de prueba).
///
/// `probe_models` no construye un `OllamaProvider` adentro: lo pide a través de este trait,
/// así que un test puede sustituir la sonda por un doble determinista.
pub trait ProbeFactory: Send + Sync {
    /// Arma la sonda para un `(endpoint, modelo)`.
    ///
    /// Toma `&ResolvedEndpoint`, no `&str`: es el newtype cuyo único constructor es
    /// `EndpointTemplate::resolve`, así que una `base_url` con placeholders sin sustituir no
    /// puede llegar hasta acá por construcción — la resolución ocurre en el arranque, después
    /// de abrir el vault y antes de sondear o de construir el trío.
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat;
}

/// Producción: `OllamaProvider` **solo como sonda**, nunca para completions (D-A07).
///
/// La `base_url` se le pasa tal cual, con su `/v1` si lo trae: `OllamaProvider::new` acepta
/// las dos formas (con y sin `/v1`) y normaliza internamente — las peticiones de probe salen
/// siempre contra la RAÍZ del daemon (`{root}/api/show`, `{root}/api/tags`), nunca bajo
/// `/v1`. Verificado por el test
/// `the_real_factory_probes_the_daemon_root_not_the_v1_prefix` de este módulo (no es un
/// intra-doc link porque vive en `#[cfg(test)]`, fuera del árbol que `cargo doc` recorre)
/// contra un servidor de prueba real, no solo leído contra el código de magi-core.
pub struct OllamaProbeFactory;

impl ProbeFactory for OllamaProbeFactory {
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat {
        // Dos respuestas distintas, nunca colapsadas en un `Option`: un kind no medible y una
        // URL malformada bajo un kind medible tienen causas y remedios distintos.
        if !kind.is_probeable() {
            return ProbeSeat::NotProbeable;
        }
        match OllamaProvider::new(base_url.as_str(), model) {
            Ok(provider) => ProbeSeat::Ready(Arc::new(provider)),
            // `redact_foreign_error`, no `e.to_string()` crudo: `base_url` es la URL YA
            // RESUELTA (REQ-A16c), y un error de otro crate que la interpolara filtraría la
            // credencial real a quien sea que termine mostrando esta razón (B11).
            Err(e) => ProbeSeat::Unbuildable(redact_foreign_error(&e)),
        }
    }
}

/// Acepta un digest solo si es EXACTAMENTE 64 caracteres hexadecimales en minúscula
/// (REQ-A16b). Cualquier otra cosa se descarta — la ventana medida sobrevive igual.
///
/// No byte-indexa nada: `str::bytes()` es un iterador total sobre los bytes de una cadena
/// UTF-8 válida, nunca panica en un borde de carácter (a diferencia de `&s[a..b]`).
fn validate_digest(raw: Option<String>) -> Option<String> {
    raw.filter(|d| {
        d.len() == DIGEST_HEX_LEN
            && d.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// Mide los modelos indicados, por composición sobre [`ProviderProbe`] (REQ-A24).
///
/// **Falla abierto, sin excepción**: error, timeout, esquema inesperado o endpoint que nunca
/// contesta degradan a [`Measurement::NotMeasuredThisTime`]. El techo es **por sonda**
/// (SC-A24k): con un plazo compartido, una sonda lenta consumiría el presupuesto de las
/// demás y las dejaría sin medir sin que ninguna haya fallado.
///
/// **Dedup por `(endpoint, modelo)`.** En el caso por defecto el trío hereda el endpoint de
/// raíz y puede compartir modelo con el principal, así que sondear cuatro veces lo mismo
/// cuadruplica el costo de arranque por el mismo número cuatro veces. La clave es el modelo
/// bajo el `base_url` recibido — el llamador ya fija un solo endpoint por llamada, así que
/// deduplicar por modelo dentro de esa llamada es deduplicar por el par completo.
///
/// El mapa devuelto tiene una entrada por modelo **pedido**, no por sonda emitida: los
/// duplicados de `models` comparten el resultado de la única sonda que se lanzó.
///
/// # Complejidad
///
/// Dos pasadas O(n) sobre `models` (deduplicar, y volver a expandir al final), sin ninguna
/// anidada: cada modelo único dispara como máximo dos peticiones HTTP (`window`, `digest`),
/// todas concurrentes vía [`futures::future::join_all`].
pub async fn probe_models(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    models: &[&str],
    factory: &dyn ProbeFactory,
) -> BTreeMap<String, Measurement> {
    if !kind.is_probeable() {
        return models
            .iter()
            .map(|m| ((*m).to_string(), Measurement::NotMeasurable))
            .collect();
    }

    let unique: BTreeSet<&str> = models.iter().copied().collect();
    let deadline = Duration::from_secs(PROBE_TIMEOUT_SECS);

    let futures = unique.into_iter().map(|model| async move {
        let probe = match factory.probe_for(kind, base_url, model) {
            ProbeSeat::Ready(p) => p,
            // Capacidad que el endpoint no ofrece: definitivo, y NO es un fallo (SC-A24b).
            ProbeSeat::NotProbeable => return (model.to_string(), Measurement::NotMeasurable),
            // Nuestra config, no su capacidad: arreglable, así que *no medido ESTA VEZ*.
            ProbeSeat::Unbuildable(_) => {
                return (model.to_string(), Measurement::NotMeasuredThisTime)
            }
        };

        // DOS techos independientes, no uno compartido entre `window` y `digest`: son dos
        // peticiones HTTP distintas que NO valen lo mismo. La ventana alimenta
        // `input_warn_tokens`; el digest solo decora un notice. Con un `timeout` envolviendo
        // a las dos, un digest lento tiraría a la basura una ventana perfectamente buena —
        // el mismo error de "plazo compartido" que SC-A24k prohíbe entre sondas, un nivel
        // más abajo.
        let window = tokio::time::timeout(deadline, probe.window())
            .await
            .ok()
            .and_then(|r| r.ok().flatten());

        let value = match window {
            Some(w) if (PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&w) => {
                // Si la ventana no salió, ni se pide el digest: ahorra una petición en el
                // caso que más importa, que es el endpoint lento o caído.
                let digest = tokio::time::timeout(deadline, probe.digest())
                    .await
                    .ok()
                    .and_then(|r| r.ok().flatten());
                Measurement::Measured {
                    window: w,
                    digest: validate_digest(digest),
                }
            }
            // Fuera de rango degrada a NO MEDIDO, nunca al extremo: un valor recortado se
            // usaría como si fuera real, y *no medido* tiene un camino previsto y auditable.
            _ => Measurement::NotMeasuredThisTime,
        };
        (model.to_string(), value)
    });

    let measured: BTreeMap<String, Measurement> = futures::future::join_all(futures)
        .await
        .into_iter()
        .collect();

    // Re-expandir: cada modelo PEDIDO recibe el resultado de su sonda, compartido si hubo
    // duplicados. El llamador ve una entrada por modelo pedido, no por sonda emitida — y el
    // `BTreeMap` de salida dedup-ea por construcción, así que pedir `[a, a, b]` da `{a, b}`.
    models
        .iter()
        .map(|m| {
            (
                (*m).to_string(),
                measured
                    .get(*m)
                    .cloned()
                    .unwrap_or(Measurement::NotMeasuredThisTime),
            )
        })
        .collect()
}

/// Deriva `input_warn_tokens` del **mínimo** de las ventanas medidas de los mages (REQ-A24b).
///
/// **De los MAGES, no del principal**: `input_warn_tokens` gobierna el input que reciben los
/// tres mages, y el modelo principal no recibe ese payload. Con un principal de ventana
/// grande y mages de ventana menor, derivarlo del principal daría un umbral que nunca
/// dispara — es responsabilidad del LLAMADOR pasar acá solo la tabla del trío, nunca la que
/// incluye al principal.
///
/// Un mage no medible se **omite** del mínimo en vez de bajarlo. Si ninguno es medible,
/// devuelve `None` y el llamador cae al nivel siguiente (clave declarada, después default).
#[must_use]
pub fn derive_warn_tokens(mages: &BTreeMap<String, Measurement>) -> Option<usize> {
    let min = mages
        .values()
        .filter_map(|m| match m {
            Measurement::Measured { window, .. } => Some(*window),
            Measurement::NotMeasurable | Measurement::NotMeasuredThisTime => None,
        })
        .min()?;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    Some((min as f64 * WARN_WINDOW_FRACTION) as usize)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use async_trait::async_trait;
    use magi_core::error::{ExternalErrorKind, ProviderError};
    use zeroize::Zeroizing;

    use super::*;
    use crate::magi::endpoint::{EndpointTemplate, Scope};
    use crate::vault::{SecretEntry, SecretStore, VaultError};

    /// Endpoint sintáctico compartido por los tests que no dependen de un valor concreto:
    /// `StubProbes` nunca hace I/O, así que nunca se contacta de verdad.
    const SYNTHETIC_BASE_URL: &str = "http://localhost:11434/v1";

    /// Vault que nunca debería consultarse: las URLs de estos tests no traen placeholders,
    /// así que `EndpointTemplate::resolve` no llega a pedir ninguna entrada (caso "Absent"
    /// de `locate_userinfo` — ver `src/magi/endpoint.rs`).
    ///
    /// # Exclusión de cobertura documentada (REQ-A00, revisión tarea 5.1 / F2)
    ///
    /// Los cinco métodos de este `impl` aparecen como funciones NO CUBIERTAS en
    /// `cargo llvm-cov`, y es estructural, no un hueco: `EndpointTemplate::resolve` solo
    /// puede llamar a `SecretStore::get`, y únicamente en la rama
    /// `UserinfoLocation::Found` — cuando la URL trae un placeholder `[user]`/`[password]`
    /// sin sustituir. Ninguna fixture de este módulo usa una URL con placeholders (todas
    /// resuelven por la rama `Absent`, que retorna antes de tocar el vault), así que ni
    /// siquiera `get` se ejecuta en la práctica. `set`, `remove`, `list` y `contains` no
    /// los invoca ningún camino de `resolve`, con o sin placeholder: existen solo porque
    /// `SecretStore` es el trait completo y un doble tiene que implementarlo entero.
    struct NoSecrets;

    impl SecretStore for NoSecrets {
        fn set(&mut self, _name: &str, _value: &str) -> Result<(), VaultError> {
            unreachable!("estos tests no escriben al vault")
        }
        fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
            Err(VaultError::SecretNotFound(name.to_string()))
        }
        fn remove(&mut self, _name: &str) -> Result<(), VaultError> {
            unreachable!("estos tests no borran del vault")
        }
        fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
            Ok(Vec::new())
        }
        fn contains(&mut self, _name: &str) -> Result<bool, VaultError> {
            Ok(false)
        }
    }

    /// Parsea y resuelve una `base_url` de fixture, sin placeholders. Falla el test (no
    /// degrada) si la fixture está mal formada: un helper de test que degradara en silencio
    /// escondería un fixture roto detrás de un resultado que parece válido.
    fn resolved(raw: &str) -> ResolvedEndpoint {
        EndpointTemplate::parse(raw)
            .expect("fixture de test bien formada")
            .resolve(&mut NoSecrets, Scope::Root)
            .expect("fixture de test sin placeholders")
    }

    /// El endpoint compartido de los tests que no ejercitan I/O real.
    fn test_endpoint() -> ResolvedEndpoint {
        resolved(SYNTHETIC_BASE_URL)
    }

    /// Doble de [`ProviderProbe`] configurable — nunca construido directamente por un test,
    /// solo a través de [`StubProbes`].
    struct StubProbe {
        /// El nombre por el que este `StubProbe` fue pedido, para registrar su timing.
        model: String,
        /// Lo que devuelve `window()`, ya "medido" por el doble.
        window: Option<usize>,
        /// Lo que devuelve `digest()`, ya "medido" por el doble.
        digest: Option<String>,
        /// Demora artificial antes de que `window()` resuelva.
        delay: Duration,
        /// Si `true`, `window()` devuelve un `ProviderError::External` REAL en vez de
        /// `Ok(self.window)` — distinto de un timeout: acá la sonda SÍ contestó, y lo que
        /// contestó fue un fallo tipado (F1, revisión tarea 5.1).
        window_fails: bool,
        /// Ídem para `digest()`, independiente de `window_fails`: un digest que falla no
        /// debe tirar una ventana ya medida con éxito.
        digest_fails: bool,
        /// Dónde registrar cuánto tardó `window()` en resolver — incluida la cancelación.
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    /// Registra en `Drop` cuánto vivió la llamada a `window()`, tanto si terminó normal como
    /// si `tokio::time::timeout` la canceló al expirar el techo — es la única forma honesta
    /// de medir "la sonda lenta agotó SU techo completo" (SC-A24k), porque una cancelación
    /// nunca llega al final del cuerpo de la función que canceló.
    struct TimingGuard {
        /// El modelo bajo el cual registrar el tiempo transcurrido.
        model: String,
        /// Cuándo arrancó, en el reloj de tokio — así respeta el reloj pausado de los tests.
        start: tokio::time::Instant,
        /// El mismo mapa compartido que expone [`StubProbes::elapsed_of`].
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    impl TimingGuard {
        /// Arranca el cronómetro para `model`.
        fn new(model: String, timings: Arc<Mutex<BTreeMap<String, Duration>>>) -> Self {
            Self {
                model,
                start: tokio::time::Instant::now(),
                timings,
            }
        }
    }

    impl Drop for TimingGuard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            if let Ok(mut map) = self.timings.lock() {
                map.insert(self.model.clone(), elapsed);
            }
        }
    }

    /// Una razón de `Unbuildable` ya redactada, para `StubProbes::always_unbuildable`. El
    /// único constructor de `SafeErrorText` es `redact_foreign_error`, así que un doble que
    /// necesite producir una no puede simplemente envolver un `String`.
    fn unbuildable_reason() -> SafeErrorText {
        redact_foreign_error(&std::io::Error::other("stub: construcción rechazada"))
    }

    #[async_trait]
    impl ProviderProbe for StubProbe {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            let _guard = TimingGuard::new(self.model.clone(), Arc::clone(&self.timings));
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.window_fails {
                return Err(ProviderError::external(
                    "stub: synthetic window failure",
                    ExternalErrorKind::Network,
                ));
            }
            Ok(self.window)
        }

        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            if self.digest_fails {
                return Err(ProviderError::external(
                    "stub: synthetic digest failure",
                    ExternalErrorKind::Network,
                ));
            }
            Ok(self.digest.clone())
        }
    }

    /// Fábrica de [`StubProbe`]s — la costura de inyección de R-A04: sin ella, todo test de
    /// `probe_models` necesitaría un servidor HTTP real.
    struct StubProbes {
        /// Ventana que emite toda sonda que no sea la "lenta" ni derive la suya (modo
        /// `derive_per_model`).
        default_window: Option<usize>,
        /// Digest que emite toda sonda en el mismo caso que `default_window`.
        default_digest: Option<String>,
        /// El modelo con demora artificial, y cuánta — si hay uno.
        slow: Option<(String, Duration)>,
        /// En vez de un valor fijo, deriva una ventana distinta por modelo (test de dedup).
        derive_per_model: bool,
        /// Si está presente, `probe_for` devuelve ESTE asiento en vez de construir uno
        /// `Ready` — para ejercitar los brazos `NotProbeable`/`Unbuildable` de `probe_models`
        /// tal como los vería con la fábrica real, sin depender de un servidor.
        seat_override: Option<SeatOverride>,
        /// Si `true`, toda sonda `Ready` que esta fábrica construye falla `window()` con un
        /// `ProviderError` real (F1, revisión tarea 5.1) — distinto de `seat_override`, que
        /// nunca llega a construir un `StubProbe`.
        window_error: bool,
        /// Ídem para `digest()`.
        digest_error: bool,
        /// Cuántas veces se llamó a `probe_for` — una por modelo ÚNICO pedido, nunca por
        /// duplicado (SC del dedup).
        built: Arc<AtomicUsize>,
        /// Cuánto tardó `window()` de cada modelo, incluida cancelación por timeout.
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    /// Qué asiento no-`Ready` debe devolver `probe_for`, cuando `StubProbes` está configurada
    /// para eso. Existe para ejercitar los brazos de `probe_models` que la fábrica real
    /// produce (`ProbeSeat::NotProbeable`, `ProbeSeat::Unbuildable`) sin depender de un
    /// servidor: `StubProbes::measuring`/`without_window`/`one_slow`/`counting` siempre
    /// devuelven `Ready`, así que ninguno de esos otros dobles ejercita esta rama.
    #[derive(Clone, Copy)]
    enum SeatOverride {
        /// Fuerza `ProbeSeat::NotProbeable` — el caso de una fábrica cuya idea de "medible"
        /// difiere de la del `kind` que `probe_models` ya verificó.
        NotProbeable,
        /// Fuerza `ProbeSeat::Unbuildable` — el caso de una URL medible que no se pudo
        /// convertir en cliente HTTP.
        Unbuildable,
    }

    impl StubProbes {
        /// Toda sonda mide la misma ventana y el mismo digest.
        fn measuring(window: usize, digest: String) -> Self {
            Self {
                default_window: Some(window),
                default_digest: Some(digest),
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Toda sonda responde `window: None` — el caso de un `/api/show` sin
        /// `*.context_length`.
        fn without_window() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `slow_model` demora `delay` antes de resolver; el resto mide una ventana fija de
        /// inmediato.
        fn one_slow(slow_model: &str, delay: Duration) -> Self {
            Self {
                default_window: Some(128_000),
                default_digest: None,
                slow: Some((slow_model.to_string(), delay)),
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Cada modelo mide una ventana DISTINTA, derivada de su propio nombre — para
        /// distinguir sondas de distintos modelos sin depender de un contador externo.
        fn counting() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: true,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `probe_for` devuelve `ProbeSeat::NotProbeable` para todo modelo pedido.
        fn always_not_probeable() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: Some(SeatOverride::NotProbeable),
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `probe_for` devuelve `ProbeSeat::Unbuildable` para todo modelo pedido.
        fn always_unbuildable() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: Some(SeatOverride::Unbuildable),
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Toda sonda arma un `Ready`, pero `window()` devuelve un `ProviderError::External`
        /// REAL —no un timeout—. Hasta la revisión de la tarea 5.1 ningún doble distinguía
        /// "no hubo tiempo" de "el provider respondió que no puede": los dos colapsan al
        /// mismo [`Measurement::NotMeasuredThisTime`], pero por caminos de código
        /// distintos dentro de `probe_models` (`tokio::time::timeout` expirando vs.
        /// `Result::ok()` descartando un `Err` interno), y solo el primero tenía cobertura.
        fn erroring_window() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: true,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `window()` mide `window` con normalidad; `digest()` devuelve un
        /// `ProviderError::External` real. Distinto de
        /// [`Self::measuring`] con un digest de formato inválido (ya cubierto por
        /// `a_malformed_body_degrades_without_panicking`): acá el provider FALLA la
        /// petición, no responde un valor que no pasa `validate_digest`.
        fn erroring_digest(window: usize) -> Self {
            Self {
                default_window: Some(window),
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: true,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Cuántas sondas distintas se construyeron — una por modelo único pedido.
        fn probes_built(&self) -> usize {
            self.built.load(Ordering::SeqCst)
        }

        /// Cuánto tardó en resolver (o ser cancelada) la sonda de `model`.
        ///
        /// # Panics
        ///
        /// Si `model` nunca llegó a pedirse — un test que llama esto sobre un modelo que no
        /// midió tiene un error en el propio test, no algo que degradar.
        ///
        /// # Exclusión de cobertura documentada (REQ-A00, revisión tarea 5.1 / F2)
        ///
        /// La closure de pánico de abajo aparece como función NO CUBIERTA en
        /// `cargo llvm-cov`, y es deliberado: ningún test de este módulo llama a
        /// `elapsed_of` sobre un modelo que no se haya medido, así que el camino de pánico
        /// nunca se toma. Un test que sí lo tomara estaría demostrando un bug del propio
        /// test, no del código de producción — no hay nada que ejercitar acá.
        fn elapsed_of(&self, model: &str) -> Duration {
            *self
                .timings
                .lock()
                .expect("el lock del stub no se envenena en estos tests")
                .get(model)
                .unwrap_or_else(|| panic!("no se registró timing para {model}"))
        }
    }

    impl ProbeFactory for StubProbes {
        fn probe_for(
            &self,
            _kind: ProviderKind,
            _base_url: &ResolvedEndpoint,
            model: &str,
        ) -> ProbeSeat {
            self.built.fetch_add(1, Ordering::SeqCst);
            match self.seat_override {
                Some(SeatOverride::NotProbeable) => return ProbeSeat::NotProbeable,
                Some(SeatOverride::Unbuildable) => {
                    return ProbeSeat::Unbuildable(unbuildable_reason());
                }
                None => {}
            }
            let delay = self
                .slow
                .as_ref()
                .filter(|(slow_model, _)| slow_model == model)
                .map_or(Duration::ZERO, |(_, d)| *d);
            let (window, digest) = if self.derive_per_model {
                // Determinista y distinto entre modelos de nombre distinto: alcanza para
                // `assert_ne!` sin necesitar un generador aleatorio en un test.
                let derived = PROBE_WINDOW_MIN + model.bytes().map(usize::from).sum::<usize>();
                (Some(derived), None)
            } else {
                (self.default_window, self.default_digest.clone())
            };
            ProbeSeat::Ready(Arc::new(StubProbe {
                model: model.to_string(),
                window,
                digest,
                delay,
                window_fails: self.window_error,
                digest_fails: self.digest_error,
                timings: Arc::clone(&self.timings),
            }))
        }
    }

    /// SC-A16b (borde, revisión tarea 5.1 / F1): 63 caracteres es UN MENOS que el mínimo
    /// válido — el borde real de la validación, distinto de la cadena de 3 caracteres que
    /// `a_malformed_body_degrades_without_panicking` ya cubre (esa solo prueba "muy corto").
    #[test]
    fn validate_digest_rejects_sixty_three_hex_chars() {
        let short = "a".repeat(DIGEST_HEX_LEN - 1);
        assert_eq!(validate_digest(Some(short)), None);
    }

    /// SC-A16b (borde): 65 caracteres es UNO MÁS que el máximo válido.
    #[test]
    fn validate_digest_rejects_sixty_five_hex_chars() {
        let long = "a".repeat(DIGEST_HEX_LEN + 1);
        assert_eq!(validate_digest(Some(long)), None);
    }

    /// SC-A16b: el contrato exige minúscula explícitamente (REQ-A16b) — hexadecimal en
    /// mayúscula, aunque representa el mismo valor, se rechaza igual que cualquier otra
    /// cosa que no matchee byte a byte.
    #[test]
    fn validate_digest_rejects_uppercase_hex() {
        let upper = "A".repeat(DIGEST_HEX_LEN);
        assert_eq!(validate_digest(Some(upper)), None);
    }

    /// SC-A16b: la longitud exacta NO alcanza sola — un carácter fuera de `[0-9a-f]` en la
    /// posición 64 también se rechaza. `'g'` es el primer carácter ASCII después de `'f'`,
    /// así que es el vecino más cercano al rango válido.
    #[test]
    fn validate_digest_rejects_a_non_hex_character_at_the_exact_length() {
        let mut bad = "a".repeat(DIGEST_HEX_LEN - 1);
        bad.push('g');
        assert_eq!(
            bad.len(),
            DIGEST_HEX_LEN,
            "el test debe seguir probando el borde exacto"
        );
        assert_eq!(validate_digest(Some(bad)), None);
    }

    /// SC-A16b (feliz, borde exacto): exactamente 64 hex en minúscula pasa tal cual, sin
    /// modificar el valor — el caso de éxito que los cuatro rechazos de arriba delimitan.
    #[test]
    fn validate_digest_accepts_exactly_sixty_four_lowercase_hex() {
        let valid = "f".repeat(DIGEST_HEX_LEN);
        assert_eq!(validate_digest(Some(valid.clone())), Some(valid));
    }

    /// SC-A24 / SC-A24b: se mide lo medible; no medible NO es un fallo.
    #[tokio::test]
    async fn ollama_is_measured_and_the_others_are_not_a_failure() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::measuring(128_000, "a".repeat(64)),
        )
        .await;
        assert!(matches!(m["m"], Measurement::Measured { .. }));

        // Sin red: el kind no medible se resuelve en `probe_models` mismo, antes de tocar
        // la fábrica, así que ni siquiera se construye un socket.
        let m = probe_models(
            ProviderKind::Anthropic,
            &test_endpoint(),
            &["m"],
            &OllamaProbeFactory,
        )
        .await;
        assert!(
            matches!(m["m"], Measurement::NotMeasurable),
            "no es un fallo: es la capacidad que ese endpoint no ofrece"
        );
    }

    /// SC-A16b: fuera de rango degrada a NO MEDIDO, nunca al extremo del rango.
    ///
    /// `PROBE_WINDOW_MIN - 1` es el borde REAL de la validación — distinto de `1`, que solo
    /// prueba "muy chico" sin tocar el límite exacto que `(PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX)`
    /// evalúa.
    #[tokio::test]
    async fn an_out_of_range_window_degrades_instead_of_being_clamped() {
        for absurd in [
            1_usize,
            PROBE_WINDOW_MIN - 1,
            PROBE_WINDOW_MAX + 1,
            999_999_999_999,
        ] {
            let m = probe_models(
                ProviderKind::Ollama,
                &test_endpoint(),
                &["m"],
                &StubProbes::measuring(absurd, "a".repeat(64)),
            )
            .await;
            assert!(
                matches!(m["m"], Measurement::NotMeasuredThisTime),
                "recortar al extremo convertiría un dato basura en uno plausible (ventana {absurd})"
            );
        }
    }

    /// SC-A16b (bordes inclusivos): `PROBE_WINDOW_MIN` y `PROBE_WINDOW_MAX` en persona se
    /// ACEPTAN tal cual — el rango es `[MIN, MAX]`, no `(MIN, MAX)`, y el test anterior solo
    /// prueba el lado del rechazo. Sin este, un `<` que debiera ser `<=` (o viceversa) en
    /// `(PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&w)` pasaría la suite igual.
    #[tokio::test]
    async fn the_window_range_boundaries_are_accepted_inclusive() {
        for edge in [PROBE_WINDOW_MIN, PROBE_WINDOW_MAX] {
            let m = probe_models(
                ProviderKind::Ollama,
                &test_endpoint(),
                &["m"],
                &StubProbes::measuring(edge, "a".repeat(64)),
            )
            .await;
            assert!(
                matches!(m["m"], Measurement::Measured { window, .. } if window == edge),
                "el rango es cerrado: {edge} debe aceptarse sin degradar (borde de PROBE_WINDOW_MIN/MAX)"
            );
        }
    }

    /// SC-A24d: respuesta malformada degrada, no rompe.
    #[tokio::test]
    async fn a_malformed_body_degrades_without_panicking() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::without_window(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasuredThisTime));

        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::measuring(128_000, "abc".to_string()),
        )
        .await;
        match &m["m"] {
            Measurement::Measured { digest, .. } => {
                assert!(
                    digest.is_none(),
                    "un digest que no es 64 hex se descarta, la ventana sobrevive"
                );
            }
            other => panic!("esperaba Measured, salió {other:?}"),
        }
    }

    /// SC-A24d (extensión, revisión tarea 5.1 / F1): un `ProviderError` REAL en `window()`
    /// degrada exactamente igual que un timeout. Hasta ahora `StubProbe` solo podía
    /// resolver con éxito o demorarse hasta expirar el `tokio::time::timeout` de
    /// `probe_models`; el brazo donde la sonda SÍ contesta y lo que contesta es un `Err`
    /// tipado —`.and_then(|r| r.ok().flatten())` descartando ese `Err`, no un `Elapsed`—
    /// nunca se había ejercitado.
    #[tokio::test]
    async fn a_genuine_provider_error_on_window_degrades_like_a_timeout() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::erroring_window(),
        )
        .await;
        assert!(
            matches!(m["m"], Measurement::NotMeasuredThisTime),
            "un ProviderError real en window() debe fallar abierto, igual que un timeout"
        );
    }

    /// SC-A24d (extensión, revisión tarea 5.1 / F1): un `ProviderError` REAL en `digest()`
    /// no tira la ventana ya medida con éxito — mismo principio de "un campo fuera de
    /// rango/roto no contamina el otro" que `a_malformed_body_degrades_without_panicking`
    /// ya prueba para un digest de FORMATO inválido, acá aplicado a un digest que falla
    /// EXPLÍCITAMENTE con un error tipado.
    #[tokio::test]
    async fn a_genuine_provider_error_on_digest_leaves_the_window_intact() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::erroring_digest(128_000),
        )
        .await;
        match &m["m"] {
            Measurement::Measured { window, digest } => {
                assert_eq!(
                    *window, 128_000,
                    "la ventana medida con éxito debe sobrevivir"
                );
                assert!(
                    digest.is_none(),
                    "un digest que falla degrada solo, sin arrastrar la ventana"
                );
            }
            other => panic!("esperaba Measured, salió {other:?}"),
        }
    }

    /// SC-A24c / SC-A24k: techo POR SONDA — una lenta no arrastra a las otras.
    ///
    /// Corre con el reloj de tokio PAUSADO: el techo real es de varios segundos, y un test
    /// que durmiera de verdad esa duración sería exactamente el defecto que este proyecto ya
    /// diagnosticó dos veces bajo carga (`nextest` en paralelo). `probe_models` no hace
    /// `tokio::spawn`, así que las cuatro sondas viven en la misma tarea y el auto-avance del
    /// reloj pausado las destraba a todas sin bloquear un solo hilo real.
    #[tokio::test(start_paused = true)]
    async fn a_slow_probe_does_not_starve_the_others() {
        let started = Instant::now();
        let stub = StubProbes::one_slow("a", Duration::from_secs(PROBE_TIMEOUT_SECS + 5));
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["a", "b", "c", "d"],
            &stub,
        )
        .await;

        assert!(
            started.elapsed() < Duration::from_secs(PROBE_TIMEOUT_SECS + 1),
            "corren en paralelo: el peor caso es UN techo, no cuatro"
        );
        assert!(matches!(m["a"], Measurement::NotMeasuredThisTime));
        assert_eq!(
            m.values()
                .filter(|v| matches!(v, Measurement::Measured { .. }))
                .count(),
            3,
            "con plazo compartido, la lenta habría dejado a las otras sin presupuesto"
        );
        // El techo es POR SONDA: la lenta consumió el suyo entero sin recortar el de nadie.
        assert!(
            stub.elapsed_of("a") >= Duration::from_secs(PROBE_TIMEOUT_SECS),
            "la sonda lenta debe agotar SU techo completo, no una fracción compartida"
        );
        for fast in ["b", "c", "d"] {
            assert!(
                stub.elapsed_of(fast) < Duration::from_secs(PROBE_TIMEOUT_SECS),
                "{fast} no debe haber esperado a la lenta"
            );
        }
    }

    /// Regresión de estructura: `probe_models` maneja `ProbeSeat::NotProbeable` devuelto POR
    /// LA FÁBRICA, no solo el atajo de `kind.is_probeable() == false` de la línea de arriba.
    /// Con `StubProbes::measuring`/`without_window`/`one_slow`/`counting` la fábrica SIEMPRE
    /// arma `Ready`, así que ninguno de esos dobles ejercita este brazo — solo lo hace una
    /// fábrica cuya noción de "medible" difiere de la del `kind`, que es exactamente lo que
    /// haría la real si algún día `is_probeable()` y `probe_for` se desincronizan.
    #[tokio::test]
    async fn a_seat_reported_not_probeable_mid_stream_is_not_a_failure() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::always_not_probeable(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasurable));
    }

    /// Ídem para `ProbeSeat::Unbuildable`: es el camino que la fábrica REAL toma cuando el
    /// `kind` es medible pero la URL no arma un cliente — arreglable, así que degrada a *no
    /// medido esta vez*, no a *no medible*.
    #[tokio::test]
    async fn an_unbuildable_seat_degrades_to_not_measured_this_time() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::always_unbuildable(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasuredThisTime));
    }

    /// SC-A24j: el umbral sale del MÍNIMO de los mages, no del principal.
    #[test]
    fn the_warn_threshold_comes_from_the_minimum_mage_window() {
        let mages = BTreeMap::from([
            (
                "melchior".to_string(),
                Measurement::Measured {
                    window: 1_000_000,
                    digest: None,
                },
            ),
            (
                "balthasar".to_string(),
                Measurement::Measured {
                    window: 128_000,
                    digest: None,
                },
            ),
            ("caspar".to_string(), Measurement::NotMeasuredThisTime),
        ]);
        let derived = derive_warn_tokens(&mages).expect("hay al menos un mage medido");
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let expected = (128_000.0 * WARN_WINDOW_FRACTION) as usize;
        assert_eq!(
            derived, expected,
            "manda el primero que deja de aceptar el payload"
        );
        assert!(derived < 128_000, "una fracción, nunca la ventana entera");
    }

    /// SC-A24j (regresión): un mage no medible se OMITE del mínimo, no lo baja.
    #[test]
    fn an_unmeasurable_mage_is_omitted_from_the_minimum() {
        let none = BTreeMap::from([("m".to_string(), Measurement::NotMeasuredThisTime)]);
        assert_eq!(
            derive_warn_tokens(&none),
            None,
            "sin ninguno medible se cae al nivel siguiente"
        );
    }

    /// Dedup: cuatro modelos pedidos, dos distintos ⇒ DOS sondas construidas, DOS entradas
    /// en el mapa devuelto (el mapa dedup-ea por clave; pedir `[a, a, b, a]` da `{a, b}` — no
    /// hay forma de que produzca cuatro entradas para tres nombres distintos).
    #[tokio::test]
    async fn identical_endpoint_and_model_are_probed_once_and_shared() {
        let counting = StubProbes::counting();
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["a", "a", "b", "a"],
            &counting,
        )
        .await;

        assert_eq!(counting.probes_built(), 2, "solo dos modelos distintos");
        assert_eq!(m.len(), 2, "el mapa dedup-ea por clave");
        // Las tres entradas de "a" comparten el resultado de la ÚNICA sonda de "a", y eso se
        // ve contra el conteo de arriba — comparar `m["a"] == m["a"]` sería tautológico.
        assert_ne!(
            m["a"], m["b"],
            "sondas de modelos distintos dan resultados distintos"
        );
        assert!(matches!(m["a"], Measurement::Measured { .. }));
    }

    /// Regresión de D-A07/R-A02: la fábrica REAL sondea la RAÍZ del daemon (`/api/show`),
    /// nunca bajo el prefijo `/v1` de las completions.
    ///
    /// `StubProbes` cubre el comportamiento de `probe_models` y NO la construcción real de
    /// la sonda: si `OllamaProbeFactory` empezara a pegarle a `/v1/api/show`, ningún otro
    /// test de este módulo lo vería. `ProviderProbe` no expone la URL que usa internamente
    /// (por diseño — ver `magi_core::providers::provider_url`), así que la única forma
    /// honesta de fijar esta propiedad es ejercitarla contra un servidor real: si el mock
    /// registrado en `/api/show` nunca es golpeado, `mock.assert_async()` hace fallar el
    /// test en vez de dejar pasar una URL equivocada en silencio.
    #[tokio::test]
    async fn the_real_factory_probes_the_daemon_root_not_the_v1_prefix() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/show")
            .with_status(200)
            .with_body(r#"{"model_info":{"x.context_length":128000}}"#)
            .create_async()
            .await;

        let base = resolved(&format!("{}/v1", server.url()));
        let seat = OllamaProbeFactory.probe_for(ProviderKind::Ollama, &base, "m");
        let ProbeSeat::Ready(probe) = seat else {
            panic!("ollama es medible, tenía que producir una sonda lista");
        };

        let window = probe.window().await.expect("el mock responde 200");
        mock.assert_async().await;
        assert_eq!(
            window,
            Some(128_000),
            "si hubiera pegado en /v1/api/show, el mock de /api/show nunca se habría \
             golpeado y `mock.assert_async()` ya habría hecho fallar este test antes de \
             llegar acá"
        );

        assert!(
            matches!(
                OllamaProbeFactory.probe_for(ProviderKind::Anthropic, &base, "m"),
                ProbeSeat::NotProbeable
            ),
            "un kind no medible no produce sonda"
        );
    }

    /// B11: cuando `OllamaProvider::new` rechaza la URL (acá, por esquema — `ftp` no es
    /// `http`/`https`), la fábrica real reporta `Unbuildable` con una razón ya pasada por
    /// `redact_foreign_error`, nunca `e.to_string()` crudo. No hay una credencial que filtrar
    /// en ESTE caso puntual (el rechazo de esquema de magi-core solo cita el nombre del
    /// esquema, nunca la URL completa — verificado contra `providers/provider_url.rs`), pero
    /// la propiedad que importa acá es de PLOMERÍA: que el camino de error pasa por el
    /// redactor y no por un `.to_string()` directo, así que un error futuro de magi-core que
    /// SÍ interpole una URL con credenciales queda cubierto por construcción, no por
    /// vigilancia.
    #[test]
    fn the_real_factory_redacts_the_reason_when_construction_fails() {
        let bad = resolved("ftp://host/v1");
        match OllamaProbeFactory.probe_for(ProviderKind::Ollama, &bad, "m") {
            ProbeSeat::Unbuildable(reason) => {
                assert!(
                    !reason.as_str().is_empty(),
                    "una razón vacía no es diagnosticable"
                );
            }
            ProbeSeat::Ready(_) => panic!("un esquema ftp no debería construir un cliente"),
            ProbeSeat::NotProbeable => {
                panic!("ollama es medible, esto es un fallo de construcción, no de capacidad")
            }
        }
    }
}
