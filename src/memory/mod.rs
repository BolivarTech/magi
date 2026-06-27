// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Memory subsystem — tiered, consultable memory with forgetting.
//!
//! Organised as a set of cooperating sub-modules. This first task adds only the
//! configuration surface (`config`); subsequent tasks will add storage, retrieval,
//! decay/eviction, context assembly, and benchmarking.

pub mod config;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_kind_serializes_to_stable_strings() {
        assert_eq!(MemoryKind::Episodic.as_str(), "episodic");
        assert_eq!(MemoryKind::Preference.as_str(), "preference");
    }

    #[test]
    fn test_embedding_error_converts_into_memory_error_and_formats() {
        // #[from] lets `?` lift an EmbeddingError into a MemoryError.
        fn lift() -> Result<(), MemoryError> {
            Err(EmbeddingError::Auth)?;
            Ok(())
        }
        let e = lift().unwrap_err();
        assert!(matches!(e, MemoryError::Embedding(EmbeddingError::Auth)));
        assert_eq!(e.to_string(), "embedding error: embedding auth failed (401/403)");
    }

    #[test]
    fn test_rate_limited_variant_formats() {
        assert_eq!(EmbeddingError::RateLimited.to_string(), "embedding rate-limited (429)");
    }
}
