// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Anclas del reporte de magi-core — **dueño único**, sin copia del lado del test.
//!
//! # Procedencia: una OBSERVACIÓN, no un contrato publicado
//!
//! El formato de `MagiReport::report` **no es API pública** de magi-core: es markdown que la
//! crate genera para consumo humano. Estas anclas salen del spike de Task 0.6, ejecutado
//! contra magi-core 3.1.0 el 2026-08-02, y verificadas después contra `src/reporting.rs` del
//! propio crate para saber **cuáles son incondicionales** y cuáles dependen del contenido.
//!
//! Por eso viven en un módulo propio y con su procedencia escrita: cuando magi-core cambie el
//! render se toca **un** archivo, y el guardián
//! `report_shape_matches_what_the_truncation_design_assumes` avisa antes que un usuario.
//!
//! # Qué se verificó, y dónde
//!
//! `ReportFormatter` compone el reporte en este orden (`reporting.rs:795-817`):
//!
//! | Sección | ¿Siempre presente? |
//! |---|---|
//! | Caja de veredicto (`MAGI SYSTEM -- VERDICT`) | **sí** |
//! | Notas de estimación, fallos de extracción, tamaño de entrada | condicionales |
//! | `## Key Findings` | solo si hay hallazgos |
//! | `## Dissenting Opinion` | solo si hay disenso |
//! | `## Conditions for Approval` | solo si hay condiciones |
//! | `## Recommended Actions` | **sí**, y es la última |
//!
//! De ahí sale la elección de `findings_end`: **no** `## Conditions for Approval`, que es
//! opcional, sino `## Recommended Actions`, que siempre está y va después de todo lo que el
//! recorte quiere conservar.

/// Anclas de SECCIÓN, **con nombre**. No es un `&[&str]` indexado por posición.
///
/// Una lista dejaría el contrato en los índices —`.first()`, `.get(1)`, `.get(2)`— y entonces
/// un spike que encuentra dos anclas en vez de tres **compila igual** y baja el techo de
/// recorte **en silencio**, con `report_truncated` todavía diciendo `structural`. Ese es el
/// peor modo de fallo disponible: el consumidor cree tener una garantía que ya no rige.
///
/// Con campos nombrados, un ancla que falta es un `Option` que hay que manejar y una que sobra
/// no tiene dónde ir: el desacuerdo entre lo medido y lo asumido pasa a ser un error de
/// compilación.
pub struct SectionAnchors {
    /// Dónde empieza el bloque de veredicto. Sin esto no hay nivel `Structural`.
    ///
    /// Es el texto interior de la caja ASCII y no la línea de `+===+`, que se repite cuatro
    /// veces y no distingue el principio del final.
    pub verdict_start: &'static str,
    /// Dónde empieza la sección de hallazgos — el corte NO va acá, va después.
    ///
    /// **Puede faltar**: magi-core omite la sección entera cuando no hay hallazgos. Un
    /// consumidor que asuma su presencia trata «no hubo hallazgos» como «no pude localizar»,
    /// que son dos cosas distintas y solo una es una degradación.
    pub findings_start: &'static str,
    /// Dónde termina la región que se conserva.
    ///
    /// `## Recommended Actions` y no `## Conditions for Approval`: aquella es incondicional y
    /// va última, así que todo lo conservable —veredicto, hallazgos, disenso, condiciones—
    /// queda de este lado. Anclar en una sección opcional dejaría el fin sin definir
    /// exactamente en los reportes que no la traen.
    pub findings_end: &'static str,
}

/// Lo que midió el spike de Task 0.6. `None` ⇒ `Structural` no es alcanzable.
///
/// Es `Some`: el reporte expone encabezados markdown estables, así que los tres niveles de
/// recorte de REQ-A11b son implementables.
pub const SECTION_ANCHORS: Option<SectionAnchors> = Some(SectionAnchors {
    verdict_start: "MAGI SYSTEM -- VERDICT",
    findings_start: "## Key Findings",
    findings_end: "## Recommended Actions",
});

/// Anclas CONTRACTUALES: el subconjunto que magi-core emite **siempre**.
///
/// No-vacío ⇒ al menos el nivel `Anchored` es alcanzable, y ese es su papel: son el fallback
/// para un reporte donde `findings_start` no existe porque no hubo hallazgos. Con estas dos
/// se puede seguir delimitando el veredicto sin caer al conteo de bytes.
pub const CONTRACTUAL_ANCHORS: &[&str] = &["MAGI SYSTEM -- VERDICT", "## Recommended Actions"];
