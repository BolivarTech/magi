// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-03

//! Clasificador de modo sobre el provider PRINCIPAL (REQ-A07c).
//!
//! Vive en el **bin**, no en `src/magi/mode.rs` (lib): `ProviderClassifier`
//! necesita `Arc<dyn Provider>`, y `agent::provider::Provider` es un tipo del
//! binario que el lib no puede ver (ver la tabla de crate split en `CLAUDE.md`).
//!
//! RED phase (Task 2.3, TDD Step 1): solo los tests y sus dobles. La
//! implementación (`ProviderClassifier`, `NoticeSink`, `ProcessNoticeSink`)
//! llega en el commit `feat:` siguiente.

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

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use magi_core::schema::Mode;
use magi_rs::magi::CLASSIFY_TIMEOUT_SECS;

use crate::agent::messages::Message;
use crate::agent::provider::Provider;

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
    /// `tokio::time::timeout` de `ProviderClassifier::classify` lo atrapa
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
            let mut seen = self.seen.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if seen.insert(key) {
                self.messages
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(msg.to_string());
            }
        }
    }

    impl RecordingNoticeSink {
        /// Cuenta cuántos avisos emitidos contienen `needle`.
        fn count_matching(&self, needle: &str) -> usize {
            self.messages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        assert_eq!(inferred, None, "un provider que no responde a tiempo falla abierto");
        assert!(
            started.elapsed() < Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1),
            "el techo propio de la clasificación no se respetó"
        );
    }

    /// SC-A07n / SC-A07o: los dos avisos, cada uno UNA vez — con sink INYECTADO.
    #[tokio::test]
    async fn the_two_notices_fire_once_each() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = ProviderClassifier::new(slow_provider(Duration::from_secs(30)), sink.clone());

        for _ in 0..3 {
            let _ = classifier.classify("x").await;
        }

        assert_eq!(
            sink.count_matching("agrega una llamada al modelo"),
            1,
            "el aviso de COSTO llega antes de pagarlo, una vez"
        );
        assert_eq!(
            sink.count_matching("expiró"),
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
        assert_eq!(sink.count_matching("agrega una llamada al modelo"), 1);
        assert_eq!(sink.count_matching("expiró"), 0, "no hubo vencimiento que reportar");
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

        assert_eq!(classifier.classify("x").await, None, "prosa no es una etiqueta");
        assert_eq!(
            sink.count_matching("expiró"),
            0,
            "una respuesta no reconocida no es un vencimiento"
        );
    }

    /// El aislamiento es real: dos tests no se contaminan aunque corran en
    /// cualquier orden (B13).
    #[tokio::test]
    async fn two_independent_sinks_do_not_share_state() {
        let a = Arc::new(RecordingNoticeSink::default());
        let b = Arc::new(RecordingNoticeSink::default());
        let _ = ProviderClassifier::new(slow_provider(Duration::from_secs(30)), a.clone())
            .classify("x")
            .await;
        let _ = ProviderClassifier::new(slow_provider(Duration::from_secs(30)), b.clone())
            .classify("x")
            .await;
        assert_eq!(a.count_matching("expiró"), 1);
        assert_eq!(
            b.count_matching("expiró"),
            1,
            "un `static` haría que este quedara en 0"
        );
    }
}
