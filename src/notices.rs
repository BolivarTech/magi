// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-03

//! Orden y tope de los notices de arranque (Task 1.5).
//!
//! # Por qué vive en el LIB y no bajo `system/`/`tui/`
//!
//! Es puro: sin I/O, sin red, sin estado. `main.rs` lo consume para ensamblar la lista
//! de notices que la TUI muestra al arrancar.
//!
//! # A qué SUPERFICIE aplica esto, y a cuál NO
//!
//! El tiering de este módulo es para notices **renderizados a un humano** en una lista
//! de arranque — hoy, únicamente la TUI. La ruta headless (`magi query`/`consult`) tiene
//! su propio contrato de salida (el envelope JSON y el run log, REQ-H23) y no consume
//! [`Notice`]: no hay ahí una lista de arranque que un humano lea, así que cargarle un
//! tier sería una representación que nada en esa ruta necesita. Es el límite correcto
//! del módulo, no un recorte de alcance — si headless alguna vez gana una lista de
//! arranque legible por humanos, ESE es el momento de decidir si consume este tipo.

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

use std::collections::HashSet;

/// Prioridad de un notice de arranque.
///
/// **El orden de declaración del enum ES el orden de impresión**: el `derive(Ord)` no
/// es decorativo — [`render_notices`] ordena con `sort_by_key(|n| n.tier)` y depende de
/// que `Blocking < Resolution < Info` en ese sentido exacto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NoticeTier {
    /// Algo que el usuario pidió NO está disponible. Acción requerida.
    Blocking,
    /// La config se resolvió distinto de lo que el archivo parece decir — o un
    /// diagnóstico de bajo nivel que no llega a bloquear pero tampoco es ruido:
    /// hardening/vault (mlock, dump suppression), un fallo al abrir o derivar la
    /// clave del vault, o la pérdida de persistencia. Ninguno de estos casos exige
    /// una acción inmediata como `Blocking`, pero todos sorprenden lo suficiente
    /// como para sobrevivir siempre al tope de [`NOTICE_MAX_INFO`].
    Resolution,
    /// Diagnóstico. Útil, nunca urgente.
    Info,
}

/// Un notice de arranque, con la prioridad que decide su lugar en la lista final.
///
/// **Toda fuente empuja `Notice`, no `String`** — antes de esta tarea, varias fuentes de
/// `main.rs` empujaban `String` planos a una lista compartida mientras el diseño de
/// tiers vivía solo en la spec, así que el orden no podía aplicarse a nada real.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    /// Prioridad — gobierna el orden de impresión y si el tope de [`render_notices`]
    /// puede alcanzarlo.
    pub tier: NoticeTier,
    /// Texto a mostrar, ya formateado por quien lo construyó.
    pub text: String,
}

impl Notice {
    /// Construye un notice `Blocking`: algo que el usuario pidió no está disponible.
    pub fn blocking(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Blocking,
            text: text.into(),
        }
    }

    /// Construye un notice `Resolution`: la config se resolvió distinto de lo escrito.
    pub fn resolution(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Resolution,
            text: text.into(),
        }
    }

    /// Construye un notice `Info`: diagnóstico, nunca urgente.
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            tier: NoticeTier::Info,
            text: text.into(),
        }
    }
}

/// Cuántos `Info` sobreviven al tope de [`render_notices`].
///
/// 5: con diez fuentes posibles, media pantalla es lo que alguien lee de verdad al
/// arrancar. No es una medición — es el mismo tipo de número elegido a mano que los
/// umbrales del gate de complejidad (REQ-A20), y se dice para no fingir lo contrario.
pub const NOTICE_MAX_INFO: usize = 5;

/// Ordena por tier (`Blocking` primero), deduplica por texto exacto, y recorta solo los
/// `Info` que excedan [`NOTICE_MAX_INFO`].
///
/// # Contrato
/// - **Orden**: `Blocking` → `Resolution` → `Info`. El `sort_by_key` es estable, así que
///   dos notices del mismo tier conservan el orden en que se pasaron.
/// - **Dedup**: dos notices con el mismo `text` colapsan en uno solo — el trío puede
///   emitir el mismo aviso de normalización de `base_url` tres veces (una por asiento),
///   y el usuario no necesita leerlo tres veces. Se aplica DESPUÉS de ordenar, así que
///   sobrevive la primera aparición en orden de tier.
/// - **Tope**: `Blocking` y `Resolution` NUNCA se recortan — el tope existe para el
///   ruido de diagnóstico, no para lo accionable ni lo sorprendente. Cuando recorta, la
///   última línea del resultado dice cuántos `Info` se omitieron.
///
/// Complejidad: `O(n log n)` por el sort más `O(n)` por el dedup (un `HashSet` de
/// textos ya vistos) — aceptable porque `n` es la cantidad de notices de UN arranque
/// (un puñado de fuentes, nunca miles).
pub fn render_notices(notices: Vec<Notice>) -> Vec<String> {
    let mut sorted = notices;
    sorted.sort_by_key(|n| n.tier);

    let mut seen_text = HashSet::with_capacity(sorted.len());
    let deduped = sorted
        .into_iter()
        .filter(|n| seen_text.insert(n.text.clone()));

    let mut info_seen = 0usize;
    let mut dropped = 0usize;
    let mut out = Vec::new();
    for n in deduped {
        if n.tier == NoticeTier::Info {
            info_seen += 1;
            if info_seen > NOTICE_MAX_INFO {
                dropped += 1;
                continue;
            }
        }
        out.push(n.text);
    }
    if dropped > 0 {
        out.push(format!("… {dropped} more diagnostic notice(s) omitted"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lo accionable primero, sin importar el orden en que se descubrió.
    #[test]
    fn notices_are_ordered_by_tier_not_by_discovery() {
        let out = render_notices(vec![
            Notice::info("ventana medida: 128k"),
            Notice::blocking("el trío no es construible: falta OPENAI_API_KEY"),
            Notice::resolution("`[embedding].base_url` heredó la raíz"),
        ]);
        assert!(
            out[0].contains("no es construible"),
            "primero lo que exige acción"
        );
        assert!(out[1].contains("heredó"));
        assert!(out[2].contains("ventana medida"));
    }

    /// El tope recorta RUIDO, nunca señales.
    #[test]
    fn the_cap_truncates_info_only_and_says_how_many_it_dropped() {
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO + 3)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        v.push(Notice::blocking("b1"));
        v.push(Notice::resolution("r1"));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains("b1")),
            "Blocking NUNCA se recorta"
        );
        assert!(out.iter().any(|n| n.contains("r1")), "Resolution tampoco");
        assert_eq!(
            out.iter().filter(|n| n.starts_with('d')).count(),
            NOTICE_MAX_INFO
        );
        assert!(out.last().unwrap().contains('3'), "dice cuántos omitió");
    }

    /// Dos fuentes pueden producir el MISMO aviso: la normalización de `/v1` la emiten los tres
    /// asientos con la misma `base_url`.
    #[test]
    fn identical_notices_are_emitted_once() {
        let n = "notice: `base_url` de Ollama sin sufijo `/v1`";
        let out = render_notices(vec![
            Notice::resolution(n),
            Notice::resolution(n),
            Notice::resolution(n),
        ]);
        assert_eq!(out.len(), 1, "tres asientos, un aviso");
    }

    /// Caso borde vacío (B13): nada que ordenar, deduplicar ni recortar — nunca panica,
    /// y sin `Info` que recortar no hay línea de "omitidos".
    #[test]
    fn empty_input_renders_to_an_empty_list() {
        let out = render_notices(vec![]);
        assert!(out.is_empty());
    }

    /// Frontera exacta del tope: `info_seen > NOTICE_MAX_INFO` es estricto, así que
    /// exactamente `NOTICE_MAX_INFO` notices `Info` no disparan NINGÚN recorte. Solo el
    /// caso por-encima-del-tope estaba cubierto antes de este test; el off-by-one en la
    /// frontera es el defecto clásico de este tipo de guard.
    #[test]
    fn exactly_the_cap_worth_of_info_drops_nothing() {
        let v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("d{i}")))
            .collect();
        let out = render_notices(v);
        assert_eq!(
            out.len(),
            NOTICE_MAX_INFO,
            "ninguno se recorta en la frontera exacta"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "sin recorte no hay línea de omitidos: {out:?}"
        );
    }

    /// La propiedad señal-vs-ruido que el módulo existe para garantizar: mismo texto,
    /// tiers distintos — sobrevive el más severo (`Blocking`), no el `Info`.
    ///
    /// Con texto IDÉNTICO, cuál sobrevivió no se puede leer directo del `String` de
    /// salida (son el mismo string). Se prueba por su EFECTO en el tope: se agregan
    /// exactamente `NOTICE_MAX_INFO` rellenos `Info` distintos, que por sí solos no
    /// disparan ningún recorte (ver el test de frontera exacta arriba). Si el
    /// duplicado sobreviviera como `Info` en vez de `Blocking`, sumaría un `Info`
    /// más y SÍ dispararía el recorte. Que no lo dispare, y que el texto duplicado
    /// siga presente, es la prueba de que sobrevivió el `Blocking` — que nunca
    /// cuenta contra el tope.
    #[test]
    fn cross_tier_duplicate_text_keeps_the_more_severe_tier() {
        let dup_text = "el trío no es construible: falta OPENAI_API_KEY";
        let mut v: Vec<Notice> = (0..NOTICE_MAX_INFO)
            .map(|i| Notice::info(format!("filler{i}")))
            .collect();
        v.push(Notice::info(dup_text));
        v.push(Notice::blocking(dup_text));

        let out = render_notices(v);
        assert!(
            out.iter().any(|n| n.contains(dup_text)),
            "el duplicado debe sobrevivir (bajo el tier Blocking): {out:?}"
        );
        assert!(
            !out.iter().any(|n| n.contains("omitted")),
            "si sobreviviera el Info, se pasaría del tope y algo se recortaría: {out:?}"
        );
    }
}
