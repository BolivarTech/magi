// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Dobles compartidos de los tests de integración de MS2 (Task 0.7).
//!
//! # Por qué es un módulo con dueño y no un detalle de cada test
//!
//! El plan cita cerca de veinte dobles en doce tareas y **no los presupuestaba en ninguna**.
//! Dejarlos implícitos hace dos daños concretos, los dos ya vistos en este proyecto:
//!
//! 1. **Rompe la fase Red.** «Rojo por la razón correcta» exige que el único símbolo faltante
//!    sea el que la tarea implementa. Un doble sin escribir hace que el test **no compile**,
//!    que es un rojo distinto y una violación de §3 de `CLAUDE.local.md`.
//! 2. **Esconde el costo donde no se ve.** El type-check y los imports de un módulo compartido
//!    no aparecen en el `use` del archivo que lo consume, así que es justo el costo que se
//!    subestima al presupuestar una tarea.
//!
//! # Se construye incremental, con la fase que lo estrena
//!
//! No se escriben los veinte acá. Cada fase agrega los suyos en su propio Step 1; lo que esta
//! tarea fija es el **dueño y el lugar**, para que no vuelvan a aparecer como nombres sin
//! archivo. Hoy viven los de Fase 0.
//!
//! **`MockEndpoint` NO está acá todavía, y es a propósito:** necesita `wiremock`, que entra
//! como dev-dependency recién en Task 0.5. Escribirlo antes dejaría este módulo sin compilar,
//! que es exactamente el fallo que viene a prevenir. Lo agrega esa tarea, junto con su
//! dependencia.
//!
//! # Imports lazy
//!
//! Lo consumen tareas de las siete fases, así que va a traer tipos que nacen en Fase 5. Cada
//! doble importa **lo suyo dentro de su propio bloque**, nunca en la cabecera: de otro modo
//! una tarea posterior rompe la colección **entera** de tests y el fallo no señala a la tarea
//! que lo causó.

// Un módulo bajo `tests/` se compila UNA VEZ POR BINARIO de test, y cada binario usa un
// subconjunto distinto. El `dead_code` que eso produce es estructural del layout, no un
// símbolo olvidado: es el patrón que documenta el propio Rust Book para `tests/common`. Sin
// esto, agregar un doble para un binario haría fallar `-D warnings` en todos los demás.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use magi_core::error::{ExternalErrorKind, ProviderError};
// `CompletionConfig` vive en `provider`, NO en `orchestrator`: ahí solo está importado y es
// privado. El plan pegaba la ruta de `orchestrator` en los tres dobles.
use magi_core::provider::{CompletionConfig, LlmProvider};
use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

/// Nombre que reportan los dobles por `LlmProvider::name`.
///
/// `LlmProvider` exige **tres** métodos, no solo `complete`: `name` y `model` no tienen impl
/// por defecto. Es telemetría —magi-core los usa para nombrar al provider en un reporte— así
/// que un doble puede devolver un valor fijo, pero no puede omitirlos.
const DOUBLE_PROVIDER_NAME: &str = "test-double";

/// Modelo que reportan los dobles por `LlmProvider::model`. Ver [`DOUBLE_PROVIDER_NAME`].
const DOUBLE_MODEL_NAME: &str = "test-double-model";

/// Un veredicto que satisface el schema de magi-core.
///
/// Va **entre los marcadores** en todos los dobles que lo emiten: magi-core 3.0.0 borró su
/// parser de búsqueda, así que un JSON pelado ya no se parsea por más válido que sea.
#[must_use]
pub fn valid_verdict_json() -> String {
    r#"{"agent":"melchior","verdict":"approve","confidence":0.9,
        "summary":"ok","reasoning":"ok","findings":[],"recommendation":"ok"}"#
        .to_string()
}

/// Envuelve un veredicto en los marcadores que magi-core exige para extraerlo.
#[must_use]
pub fn marked_verdict() -> String {
    format!("{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}", valid_verdict_json())
}

/// Los tres asientos, en minúscula, tal como magi-core los espera en el campo `agent`.
const SEAT_NAMES: [&str; 3] = ["melchior", "balthasar", "caspar"];

/// Un veredicto **con hallazgos**, emitido a nombre de `agent`.
///
/// Dos cosas que este helper existe para resolver, ambas descubiertas ejecutando:
///
/// 1. **Los hallazgos no pueden ir vacíos.** El de [`valid_verdict_json`] los trae así, y con
///    la lista vacía el reporte puede no emitir la sección de hallazgos en absoluto — o sea
///    que un spike sobre él no podría decidir si esa sección es localizable, que es justo lo
///    que Task 0.6 tiene que decidir.
/// 2. **El campo `agent` DEBE coincidir con el asiento que preguntó.** magi-core valida esa
///    correspondencia y descarta el veredicto que no la cumple: un doble que responde
///    `"melchior"` a los tres deja `succeeded: 1` y el orquestador aborta con
///    `InsufficientAgents`. El guardián de reintentos no lo notaba porque solo cuenta
///    llamadas, así que la restricción quedó invisible hasta que un test miró el reporte.
#[must_use]
pub fn verdict_json_with_findings(agent: &str) -> String {
    format!(
        r#"{{"agent":"{agent}","verdict":"conditional","confidence":0.85,
        "summary":"resumen de una linea","reasoning":"razonamiento del mage",
        "findings":[
          {{"severity":"critical","title":"Primer hallazgo","detail":"detalle del primero",
           "file":"src/x.rs","line":42,"category":"logic-error"}},
          {{"severity":"warning","title":"Segundo hallazgo","detail":"detalle del segundo",
           "file":"src/y.rs","line":7,"category":"performance"}}
        ],
        "recommendation":"lo que recomienda"}}"#
    )
}

/// Deduce el asiento a partir de la **primera línea** de un system prompt.
///
/// **Solo el encabezado, nunca el prompt entero, y esto costó un ciclo descubrirlo:** los
/// prompts se mencionan entre sí —el de Caspar dice *"Leave happy-path correctness analysis to
/// Melchior"*— así que buscar el nombre en todo el texto le asigna a Caspar el veredicto de
/// Melchior, magi-core lo rechaza por `agent identity mismatch` y el asiento se pierde. La
/// primera línea es `# <Nombre> — <Rol>`, que sí discrimina.
///
/// Cae en `melchior` si no reconoce ninguno: un doble no debe fallar en silencio, pero tampoco
/// panicar dentro de una tarea del orquestador — el `InsufficientAgents` resultante lo delata
/// igual, y con mejor mensaje que un panic en un `spawn`.
///
/// **Función libre y no un método de un solo doble (B3, MAGI S3 re-gate fix):** originalmente
/// vivía solo en `AdheringTrioProvider`, pero `OverlapCountingProvider` tiene la MISMA
/// necesidad — un doble que responde el mismo `"agent":"melchior"` a los tres asientos hace que
/// magi-core rechace y **reintente** los dos que no coinciden, multiplicando las llamadas a
/// `complete` más allá del número de asientos y rompiendo cualquier guardián que cuente
/// llamadas exactas (como el rendezvous de `OverlapCountingProvider`). Compartir la función
/// evita que un tercer doble la reimplemente mal.
fn seat_from_prompt(system_prompt: &str) -> &'static str {
    let header = system_prompt
        .lines()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    SEAT_NAMES
        .into_iter()
        .find(|seat| header.contains(seat))
        .unwrap_or("melchior")
}

/// Responde un veredicto válido **a nombre del asiento que preguntó**, con hallazgos.
///
/// Los tres asientos comparten la instancia y el doble los discrimina por su system prompt,
/// que es donde aparece el nombre del mage (REQ-A02). Así el reporte sale con los tres
/// adhiriendo, que es la condición para que el render exponga todas sus secciones.
pub struct AdheringTrioProvider;

#[async_trait]
impl LlmProvider for AdheringTrioProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let seat = seat_from_prompt(system_prompt);
        Ok(format!(
            "{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}",
            verdict_json_with_findings(seat)
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Devuelve schema inválido en el primer intento de cada asiento y válido en el segundo.
///
/// Sostiene el guardián de SC-A04b: un fallo de validación consume **dos** ventanas de
/// `timeout`, que es de donde sale el factor 2 de la fórmula del `--timeout` (REQ-A04).
pub struct SchemaFailsOnceProvider {
    /// Llamadas **por asiento**, discriminadas por su system prompt.
    ///
    /// `Mutex<BTreeMap>` y no `AtomicUsize`: un contador global no distingue «un asiento
    /// reintentó» de «tres asientos llamaron una vez», y con tres mages un `total >= 2` pasa
    /// **aunque magi-core no reintente nunca**. Un guardián que no puede fallar es peor que
    /// ninguno, porque además certifica.
    ///
    /// El system prompt sirve de clave porque cada mage recibe el suyo (REQ-A02), así que el
    /// doble discrimina asientos sin tener que saber de `AgentName`.
    pub calls_by_seat: Mutex<BTreeMap<String, usize>>,
    /// Latencia simulada de cada llamada.
    pub per_call: Duration,
}

impl SchemaFailsOnceProvider {
    /// Construye el doble con su mapa de conteo vacío.
    #[must_use]
    pub fn new(per_call: Duration) -> Self {
        Self {
            calls_by_seat: Mutex::new(BTreeMap::new()),
            per_call,
        }
    }

    /// Copia del conteo por asiento, para afirmar sobre él sin sostener el lock.
    ///
    /// # Panics
    ///
    /// Si el `Mutex` quedó envenenado por un panic de otro test.
    #[must_use]
    pub fn calls_by_seat(&self) -> BTreeMap<String, usize> {
        self.calls_by_seat.lock().expect("sin envenenar").clone()
    }
}

#[async_trait]
impl LlmProvider for SchemaFailsOnceProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        tokio::time::sleep(self.per_call).await;
        let previous = {
            let mut map = self.calls_by_seat.lock().expect("sin envenenar");
            let counter = map.entry(system_prompt.to_string()).or_insert(0);
            *counter += 1;
            *counter - 1
        };
        if previous == 0 {
            Ok("no soy un veredicto".to_string())
        } else {
            Ok(marked_verdict())
        }
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Nunca responde. Sostiene la otra mitad de SC-A04b: un cuelgue consume **una** ventana,
/// porque un timeout del provider **no** dispara el reintento correctivo de schema.
pub struct HangingProvider;

#[async_trait]
impl LlmProvider for HangingProvider {
    async fn complete(
        &self,
        _system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        std::future::pending::<()>().await;
        Err(ProviderError::external(
            "inalcanzable",
            ExternalErrorKind::Network,
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Registra el máximo de ejecuciones simultáneas que vio.
///
/// Sostiene SC-A04e: los tres mages **se solapan**. Si magi-core pasara a despachar en serie,
/// el peor caso saltaría de 2× a 6× el techo y el `--timeout` derivado empezaría a cortar
/// consults sanos — sin que una sola línea de magi-rs cambiara.
///
/// **Rendezvous, no `sleep` fijo (MAGI S3 re-gate, Caspar).** La versión anterior hacía que
/// cada llamada durmiera un `dwell` fijo (500 ms) antes de salir, confiando en que las tres
/// llegaran DENTRO de esa ventana para que el pico llegara a 3 — exactamente el patrón de
/// flakiness que este repo ya diagnosticó dos veces (`.config/nextest.toml`): bajo la carga
/// Argon2 del resto de la suite (este test corre en el grupo `default`, no en `heavy`), un
/// asiento retrasado en despacharse podía salir DESPUÉS de que otro ya hubiera dormido su
/// ventana entera y salido, bajando el pico observado a 2 sin que hubiera ningún defecto real.
///
/// Con un [`tokio::sync::Barrier`] de tamaño `expected`, ninguna llamada puede "salir"
/// (decrementar `live`) hasta que las `expected` hayan "llegado" — la contención de CPU solo
/// hace que el rendezvous tarde más, nunca que el pico observado sea menor. Es la misma
/// disciplina que el resto del proyecto exige para tests con reloj: esperar sobre una
/// CONDICIÓN, no sobre una duración.
///
/// **El veredicto responde a nombre del asiento que preguntó, vía [`seat_from_prompt`] — y
/// esto NO es cosmético para un rendezvous de tamaño fijo.** La primera versión de este ajuste
/// reusaba `marked_verdict()` (siempre `"agent":"melchior"`), y el rendezvous colgó: magi-core
/// rechaza el veredicto de Balthasar/Caspar por `agent identity mismatch` y **reintenta** esos
/// dos asientos, así que `complete` termina llamándose más de `expected` veces para una sola
/// `analyze`. Con un `Barrier` de tamaño 3 eso deja arribos sueltos que nunca completan un
/// grupo de tres — exactamente el hallazgo que el comentario de [`AdheringTrioProvider`]
/// documenta, aplicado acá donde además rompe la sincronización, no solo el conteo.
pub struct OverlapCountingProvider {
    /// Ejecuciones en vuelo ahora.
    pub live: Arc<AtomicUsize>,
    /// Máximo de ejecuciones simultáneas observado.
    pub peak: Arc<AtomicUsize>,
    /// Punto de encuentro: se libera recién cuando `expected` llamadas llegaron a la vez.
    barrier: Arc<tokio::sync::Barrier>,
}

impl OverlapCountingProvider {
    /// Construye el doble junto con los contadores que el test va a leer.
    ///
    /// `expected` es cuántas llamadas simultáneas debe esperar el rendezvous antes de liberar
    /// a todas — típicamente [`EXPECTED_SEATS`] en `magi_core_contract.rs`, pero el doble no
    /// hardcodea ese número: es el llamador quien conoce cuántos asientos va a despachar.
    #[must_use]
    pub fn new(expected: usize) -> (Arc<Self>, Arc<AtomicUsize>) {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(Self {
            live,
            peak: Arc::clone(&peak),
            barrier: Arc::new(tokio::sync::Barrier::new(expected)),
        });
        (provider, peak)
    }
}

#[async_trait]
impl LlmProvider for OverlapCountingProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        // Blocks until every expected call has arrived — see the struct doc for why this
        // replaces a fixed `sleep`. The caller wraps the whole `analyze` in a generous timeout
        // so a genuine regression to serial dispatch (this never resolving) fails clearly
        // instead of hanging the suite.
        self.barrier.wait().await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        // NOT `marked_verdict()` — see the struct doc for why a shared, seat-blind verdict
        // (always `"agent":"melchior"`) breaks a fixed-size rendezvous: magi-core retries the
        // two mismatched seats, and their retry calls arrive after the barrier has already
        // moved past this generation.
        let seat = seat_from_prompt(system_prompt);
        Ok(format!(
            "{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}",
            verdict_json_with_findings(seat)
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}
