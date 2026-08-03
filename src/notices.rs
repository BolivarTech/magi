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
        assert!(out[0].contains("no es construible"), "primero lo que exige acción");
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
        assert!(out.iter().any(|n| n.contains("b1")), "Blocking NUNCA se recorta");
        assert!(out.iter().any(|n| n.contains("r1")), "Resolution tampoco");
        assert_eq!(out.iter().filter(|n| n.starts_with('d')).count(), NOTICE_MAX_INFO);
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
}
