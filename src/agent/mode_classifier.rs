// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-03

//! Clasificador de modo sobre el provider PRINCIPAL (REQ-A07c).
//!
//! Vive en el **bin**, no en `src/magi/mode.rs` (lib): [`ProviderClassifier`]
//! necesita `Arc<dyn Provider>`, y `agent::provider::Provider` es un tipo del
//! binario que el lib no puede ver (ver la tabla de crate split en `CLAUDE.md`).
//! El trait puro [`magi_rs::magi::mode::ModeClassifier`] que esto implementa,
//! en cambio, sí vive en el lib — es sin I/O, así que no tiene la misma
//! restricción.

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

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use magi_core::schema::Mode;
use magi_rs::magi::mode::{normalize_label, ModeClassifier};
use magi_rs::magi::CLASSIFY_TIMEOUT_SECS;

use crate::agent::messages::Message;
use crate::agent::provider::Provider;

/// Delimitadores del contenido no confiable en el prompt de clasificación.
///
/// El contenido va **delimitado** y el prompt declara que lo de adentro es dato
/// a clasificar, nunca instrucciones. Es higiene: la contención real es
/// [`normalize_label`], porque no depende de que el modelo se porte bien.
const CONTENT_OPEN: &str = "<<<CONTENIDO_A_CLASIFICAR>>>";
/// Ver [`CONTENT_OPEN`].
const CONTENT_CLOSE: &str = "<<<FIN_CONTENIDO>>>";

/// Clave del aviso de COSTO: sin `--mode` ni `default_mode`, se agrega una
/// llamada al modelo. Ver [`NoticeSink::once`].
const NOTICE_CLASSIFY_COST: &str = "classify.cost";
/// Clave del aviso de VENCIMIENTO: la clasificación expiró o falló. Distinto
/// del anterior — este avisa que algo FALLÓ, no que algo VA A OCURRIR.
const NOTICE_CLASSIFY_TIMEOUT: &str = "classify.timeout";

/// Emisor de avisos de-una-sola-vez, **inyectable**.
///
/// **Resuelve una tensión real entre dos reglas, y por eso no es ni un campo ni
/// un `static`.** La spec exige *"una vez por proceso"*: un `AtomicBool` como
/// campo del clasificador cumple eso solo si existe exactamente un
/// clasificador — cierto hoy, **no un contrato**. Pero un `static` cumple la
/// semántica y **rompe B13** («tests aislados, sin estado compartido»): el
/// orden de los tests pasaría a decidir cuál ve el aviso.
///
/// El sink satisface las dos: **una instancia compartida a nivel proceso** en
/// producción (para `magi consult` headless, un proceso ES una corrida, así
/// que construir un [`ProcessNoticeSink`] por invocación ya cumple "una vez por
/// proceso"), **una fresca por test** en la suite. La semántica de "una vez"
/// vive en el sink, no en quien lo usa.
pub trait NoticeSink: Send + Sync {
    /// Emite `msg` la primera vez que se llama con `key`; las siguientes son
    /// no-op para esa `key`.
    fn once(&self, key: &'static str, msg: &str);
}

/// Sink de producción: escribe a stderr, deduplicando por clave.
#[derive(Default)]
pub struct ProcessNoticeSink {
    /// Claves ya emitidas, para el dedup.
    seen: Mutex<BTreeSet<&'static str>>,
}

impl NoticeSink for ProcessNoticeSink {
    fn once(&self, key: &'static str, msg: &str) {
        let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
        if seen.insert(key) {
            eprintln!("{msg}");
        }
    }
}

/// Clasificador sobre el provider PRINCIPAL (REQ-A07c).
///
/// Usa el principal y no el trío: es una clasificación de una etiqueta, no una
/// deliberación; pagarla al precio de tres mages sería absurdo.
pub struct ProviderClassifier {
    /// El provider principal ya resuelto (mismo que atiende el tool loop).
    provider: Arc<dyn Provider>,
    /// Emisor de los dos avisos de-una-sola-vez de este módulo.
    notices: Arc<dyn NoticeSink>,
}

impl ProviderClassifier {
    /// Crea un clasificador sobre `provider`, emitiendo sus avisos por `notices`.
    #[must_use]
    pub fn new(provider: Arc<dyn Provider>, notices: Arc<dyn NoticeSink>) -> Self {
        Self { provider, notices }
    }
}

#[async_trait]
impl ModeClassifier for ProviderClassifier {
    async fn classify(&self, content: &str) -> Option<Mode> {
        self.notices.once(
            NOTICE_CLASSIFY_COST,
            "notice: without `--mode` or `[magi].default_mode`, magi-rs adds a call to the \
             model to infer the lens. Declaring the mode avoids it.",
        );

        let prompt = format!(
            "Classify the delimited content into exactly one of these labels: \
             code-review, design, analysis. Respond with ONLY the label.\n\
             {CONTENT_OPEN}\n{content}\n{CONTENT_CLOSE}"
        );
        let msgs = [Message::user(&prompt)];
        let deadline = Duration::from_secs(CLASSIFY_TIMEOUT_SECS);

        match tokio::time::timeout(deadline, self.provider.send_messages(&msgs, &[], None)).await {
            Ok(Ok(reply)) => normalize_label(&reply.concat_text()),
            Ok(Err(_)) | Err(_) => {
                self.notices.once(
                    NOTICE_CLASSIFY_TIMEOUT,
                    &format!(
                        "notice: mode inference expired ({CLASSIFY_TIMEOUT_SECS}s) or failed; \
                         using `analysis`. On slow providers, declare \
                         `[magi].default_mode`."
                    ),
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use std::time::Instant;

    use anyhow::Result;
    use futures::stream::{self, BoxStream, StreamExt};

    use crate::agent::provider::ResponseChunk;
    use crate::tools::Tool;

    use super::*;

    /// Doble de [`Provider`] que espera `delay` y entonces responde con `text`.
    /// El `sleep` vive DENTRO de `stream_messages`, así que el
    /// `tokio::time::timeout` de [`ProviderClassifier::classify`] lo atrapa
    /// igual que atraparía una latencia real de red.
    struct DelayedProvider {
        /// Cuánto espera antes de responder.
        delay: Duration,
        /// Lo que responde, una vez pasado el delay.
        text: String,
    }

    #[async_trait]
    impl Provider for DelayedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            tokio::time::sleep(self.delay).await;
            let msg = Message::assistant(&self.text);
            Ok(stream::iter(vec![Ok(ResponseChunk::MessageDone(msg))]).boxed())
        }
    }

    /// Un provider que nunca responde a tiempo para [`CLASSIFY_TIMEOUT_SECS`].
    fn slow_provider(delay: Duration) -> Arc<dyn Provider> {
        Arc::new(DelayedProvider {
            delay,
            text: "design".to_string(),
        })
    }

    /// Un provider que responde `label` de inmediato.
    fn provider_returning(label: &str) -> Arc<dyn Provider> {
        Arc::new(DelayedProvider {
            delay: Duration::ZERO,
            text: label.to_string(),
        })
    }

    /// Sink de test: acumula en memoria, sin tocar stderr ni estado global —
    /// lo que mantiene B13 («tests aislados, sin estado compartido») mientras
    /// la semántica de "una vez por proceso" sigue viva en producción.
    #[derive(Default)]
    struct RecordingNoticeSink {
        /// Claves ya vistas, para el dedup (igual que `ProcessNoticeSink`).
        seen: Mutex<BTreeSet<&'static str>>,
        /// Los mensajes que sí se emitieron, en orden.
        messages: Mutex<Vec<String>>,
    }

    impl NoticeSink for RecordingNoticeSink {
        fn once(&self, key: &'static str, msg: &str) {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            if seen.insert(key) {
                self.messages
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(msg.to_string());
            }
        }
    }

    impl RecordingNoticeSink {
        /// Cuenta cuántos avisos emitidos contienen `needle`.
        fn count_matching(&self, needle: &str) -> usize {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .filter(|m| m.contains(needle))
                .count()
        }
    }

    /// SC-A07m: en un provider lento, la clasificación degrada a `None` sin
    /// exceder su propio techo — nunca hereda el delay del provider.
    #[tokio::test]
    async fn a_slow_provider_degrades_every_inference_to_default() {
        let classifier = ProviderClassifier::new(
            slow_provider(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 2)),
            Arc::new(ProcessNoticeSink::default()),
        );

        let started = Instant::now();
        let inferred = classifier.classify("x").await;
        assert_eq!(
            inferred, None,
            "un provider que no responde a tiempo falla abierto"
        );
        assert!(
            started.elapsed() < Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1),
            "el techo propio de la clasificación no se respetó"
        );
    }

    /// SC-A07n / SC-A07o: los dos avisos, cada uno UNA vez — con sink INYECTADO.
    ///
    /// Reloj PAUSADO (m2, revisión Task 2.3): el provider de 30s nunca corre en
    /// tiempo real — cada `classify` se lanza en su propia tarea y
    /// `tokio::time::advance` salta directo más allá del techo de
    /// [`CLASSIFY_TIMEOUT_SECS`], que es lo único que este test necesita
    /// observar. Con reloj real esto costaba ~18s (3 llamadas × 6s) en cada
    /// corrida de la suite.
    #[tokio::test(start_paused = true)]
    async fn the_two_notices_fire_once_each() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = Arc::new(ProviderClassifier::new(
            slow_provider(Duration::from_secs(30)),
            sink.clone(),
        ));

        for _ in 0..3 {
            let classifier = Arc::clone(&classifier);
            let handle = tokio::spawn(async move { classifier.classify("x").await });
            tokio::time::advance(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1)).await;
            let _ = handle.await.expect("the classify task must not panic");
        }

        assert_eq!(
            sink.count_matching("adds a call to the model"),
            1,
            "el aviso de COSTO llega antes de pagarlo, una vez"
        );
        assert_eq!(
            sink.count_matching("expired"),
            1,
            "el aviso de VENCIMIENTO es distinto: avisa que falló, no que va a ocurrir"
        );
    }

    /// SC-A07o: el aviso de COSTO sale aunque la clasificación FUNCIONE.
    ///
    /// El test anterior lo ejercita con una clasificación que falla, así que no
    /// distingue «avisó del costo» de «avisó del fallo». Este confirma que el
    /// aviso previo es independiente del resultado: la llamada se paga igual,
    /// y de eso avisa.
    #[tokio::test]
    async fn the_cost_notice_fires_even_when_classification_succeeds() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = ProviderClassifier::new(provider_returning("code-review"), sink.clone());

        assert_eq!(
            classifier.classify("x").await,
            Some(Mode::CodeReview),
            "clasificó bien"
        );
        assert_eq!(sink.count_matching("adds a call to the model"), 1);
        assert_eq!(
            sink.count_matching("expired"),
            0,
            "no hubo vencimiento que reportar"
        );
    }

    /// Caso borde: el provider responde a tiempo pero con algo que no es una
    /// etiqueta (prosa, JSON, etiqueta inventada). Cae a `None` igual que un
    /// fallo real, pero SIN el aviso de vencimiento — no expiró ni falló la
    /// llamada, solo no nombró un modo válido (REQ-A07j).
    #[tokio::test]
    async fn an_unrecognized_reply_yields_none_without_a_timeout_notice() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = ProviderClassifier::new(
            provider_returning("el modo apropiado seria code-review"),
            sink.clone(),
        );

        assert_eq!(
            classifier.classify("x").await,
            None,
            "prosa no es una etiqueta"
        );
        assert_eq!(
            sink.count_matching("expired"),
            0,
            "una respuesta no reconocida no es un vencimiento"
        );
    }

    /// El aislamiento es real: dos tests no se contaminan aunque corran en
    /// cualquier orden (B13).
    ///
    /// Reloj PAUSADO (m2): mismo motivo que `the_two_notices_fire_once_each` —
    /// dos clasificaciones contra un provider de 30s no necesitan tiempo real
    /// para probar el aislamiento del sink.
    #[tokio::test(start_paused = true)]
    async fn two_independent_sinks_do_not_share_state() {
        let a = Arc::new(RecordingNoticeSink::default());
        let b = Arc::new(RecordingNoticeSink::default());

        let handle_a = tokio::spawn(async move {
            ProviderClassifier::new(slow_provider(Duration::from_secs(30)), a.clone())
                .classify("x")
                .await;
            a
        });
        let handle_b = tokio::spawn(async move {
            ProviderClassifier::new(slow_provider(Duration::from_secs(30)), b.clone())
                .classify("x")
                .await;
            b
        });
        tokio::time::advance(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1)).await;
        let a = handle_a.await.expect("classifier `a` task must not panic");
        let b = handle_b.await.expect("classifier `b` task must not panic");

        assert_eq!(a.count_matching("expired"), 1);
        assert_eq!(
            b.count_matching("expired"),
            1,
            "un `static` haría que este quedara en 0"
        );
    }
}
