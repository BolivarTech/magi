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
pub struct OverlapCountingProvider {
    /// Ejecuciones en vuelo ahora.
    pub live: Arc<AtomicUsize>,
    /// Máximo de ejecuciones simultáneas observado.
    pub peak: Arc<AtomicUsize>,
    /// Cuánto permanece «adentro» cada llamada, para que el solapamiento sea observable.
    pub dwell: Duration,
}

impl OverlapCountingProvider {
    /// Construye el doble junto con los contadores que el test va a leer.
    #[must_use]
    pub fn new(dwell: Duration) -> (Arc<Self>, Arc<AtomicUsize>) {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(Self {
            live,
            peak: Arc::clone(&peak),
            dwell,
        });
        (provider, peak)
    }
}

#[async_trait]
impl LlmProvider for OverlapCountingProvider {
    async fn complete(
        &self,
        _system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.dwell).await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        Ok(marked_verdict())
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}
