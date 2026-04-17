# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for parse_agent_output.py — Claude CLI JSON extraction."""

import json
import os
import tempfile

import pytest

from parse_agent_output import _strip_code_fences, _extract_text, parse_agent_output


class TestStripCodeFences:
    """Verify markdown code fence removal."""

    def test_no_fences_unchanged(self):
        assert _strip_code_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences_stripped(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_bare_fences_stripped(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_uppercase_json_fences_stripped(self):
        text = '```JSON\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_fences_with_surrounding_whitespace(self):
        text = '  ```json\n{"key": "value"}\n```  '
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_nested_backticks_in_content_preserved(self):
        text = '```json\n{"code": "use `var`"}\n```'
        result = _strip_code_fences(text)
        assert "`var`" in result


class TestExtractText:
    """Verify text extraction from various Claude CLI output formats."""

    def test_result_format(self):
        data = {"result": '{"agent": "melchior", "verdict": "approve"}'}
        assert _extract_text(data) == '{"agent": "melchior", "verdict": "approve"}'

    def test_content_block_format(self):
        data = {
            "content": [{"type": "text", "text": '{"agent": "balthasar", "verdict": "reject"}'}]
        }
        assert _extract_text(data) == '{"agent": "balthasar", "verdict": "reject"}'

    def test_content_block_skips_non_text(self):
        data = {
            "content": [
                {"type": "image", "url": "http://example.com"},
                {"type": "text", "text": "extracted"},
            ]
        }
        assert _extract_text(data) == "extracted"

    def test_content_block_no_text_raises(self):
        data = {"content": [{"type": "image", "url": "http://example.com"}]}
        with pytest.raises(ValueError, match="No text block"):
            _extract_text(data)

    def test_content_must_be_a_list(self):
        """A non-list ``content`` value must be rejected, not silently
        iterated character-by-character."""
        data = {"content": "not-a-list"}
        with pytest.raises(ValueError, match="'content' must be a list"):
            _extract_text(data)

    def test_content_dict_not_accepted(self):
        """A dict under ``content`` would silently iterate its keys; reject it."""
        data = {"content": {"type": "text", "text": "inline"}}
        with pytest.raises(ValueError, match="'content' must be a list"):
            _extract_text(data)

    def test_plain_string(self):
        assert _extract_text("hello world") == "hello world"

    def test_fallback_dict_raises_value_error(self):
        data = {"unknown_key": "some_value"}
        with pytest.raises(ValueError, match="Unexpected Claude CLI output type"):
            _extract_text(data)

    def test_result_key_takes_precedence_over_content(self):
        data = {
            "result": "from_result",
            "content": [{"type": "text", "text": "from_content"}],
        }
        assert _extract_text(data) == "from_result"


def _write_temp(content: str, *, suffix: str = ".json") -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestParseAgentOutput:
    """Integration tests for the full parse pipeline."""

    def test_result_format_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }
        )
        raw = json.dumps({"result": agent_json})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "melchior"
            assert result["verdict"] == "approve"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_content_block_format_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "caspar",
                "verdict": "reject",
                "confidence": 0.7,
                "summary": "Bad",
                "reasoning": "Risky",
                "findings": [],
                "recommendation": "Rework",
            }
        )
        raw = json.dumps({"content": [{"type": "text", "text": agent_json}]})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "caspar"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_code_fenced_result_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "balthasar",
                "verdict": "conditional",
                "confidence": 0.8,
                "summary": "Maybe",
                "reasoning": "Depends",
                "findings": [],
                "recommendation": "Add tests",
            }
        )
        fenced = f"```json\n{agent_json}\n```"
        raw = json.dumps({"result": fenced})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "balthasar"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_invalid_json_raises(self):
        raw = json.dumps({"result": "not valid json at all"})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(input_path, output_path)
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_missing_input_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_agent_output("/nonexistent/input.json", "/tmp/out.json")

    def test_output_has_trailing_newline(self):
        agent_json = json.dumps(
            {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "Good",
                "reasoning": "Clean",
                "findings": [],
                "recommendation": "Ship",
            }
        )
        raw = json.dumps({"result": agent_json})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                content = f.read()
            assert content.endswith("\n")
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
