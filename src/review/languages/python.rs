// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Python-specific semantic extractor using indentation-based block detection.
//!
//! Recognizes `def`, `async def`, `class`, `import`/`from`, and module-level
//! assignments. Block boundaries are determined by indentation level changes.

use std::collections::HashSet;

#[cfg(test)]
use super::UnitKind;
use super::{LanguageExtractor, SemanticUnit};

/// Indentation-based Python code extractor.
///
/// Detects:
/// - `def` / `async def` → `UnitKind::Function`
/// - `class` → `UnitKind::Class`
/// - `import` / `from ... import` → `UnitKind::DeclarationGroup`
/// - Module-level assignments (indent 0) → `UnitKind::DeclarationGroup`
#[derive(Debug, Clone, Copy)]
pub struct PythonExtractor;

impl LanguageExtractor for PythonExtractor {
    fn language_id(&self) -> &str {
        "python"
    }

    fn extensions(&self) -> &[&str] {
        &["py", "pyi"]
    }

    fn extract_units(
        &self,
        _content: &str,
        _file_path: &str,
        _changed_lines: &HashSet<u32>,
    ) -> Vec<SemanticUnit> {
        Vec::new() // Red stub
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(code: &str, changed: &[u32]) -> Vec<SemanticUnit> {
        let ext = PythonExtractor;
        let changed_set: HashSet<u32> = changed.iter().copied().collect();
        ext.extract_units(code, "test.py", &changed_set)
    }

    // --- Test 29: python_extracts_function ---
    #[test]
    fn python_extracts_function() {
        let code = "def foo():\n    x = 1\n    return x\n";
        let units = extract(code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one function");
        assert_eq!(units[0].kind, UnitKind::Function);
        assert_eq!(units[0].name, "foo");
    }

    // --- Test 30: python_extracts_class ---
    #[test]
    fn python_extracts_class() {
        let code = "class Foo:\n    def __init__(self):\n        self.x = 1\n";
        let units = extract(code, &[3]);
        assert_eq!(units.len(), 1, "Should extract one class");
        assert_eq!(units[0].kind, UnitKind::Class);
        assert_eq!(units[0].name, "Foo");
    }

    // --- Test 31: python_extracts_module_level_assignment ---
    #[test]
    fn python_extracts_module_level_assignment() {
        let code = "MAX_SIZE = 100\nMIN_SIZE = 10\n\ndef foo():\n    pass\n";
        let units = extract(code, &[1]);
        assert_eq!(units.len(), 1, "Should extract one declaration group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
    }

    // --- Test 32: python_extracts_import_group ---
    #[test]
    fn python_extracts_import_group() {
        let code = "import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    pass\n";
        let units = extract(code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one import group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
        assert_eq!(units[0].name, "imports");
    }
}
