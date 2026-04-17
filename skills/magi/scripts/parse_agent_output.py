#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Parse and validate agent JSON output from Claude CLI.

Extracts structured JSON from various Claude CLI output formats,
strips markdown code fences, validates the result, and writes
clean JSON to the specified output file.

Usage:
    python3 parse_agent_output.py <input_file> <output_file>

Exit codes:
    0 - Success: valid JSON extracted and written to output file.
    1 - Failure: input could not be parsed or did not contain valid JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys

from validate import MAX_INPUT_FILE_SIZE


# Regex to strip leading ```json (case-insensitive, optional whitespace) or bare ```
_FENCE_START = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_END = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping the text.

    Handles variants such as ```json, ```JSON, ``` json, and bare ```.

    Args:
        text: Raw text potentially wrapped in code fences.

    Returns:
        Text with leading/trailing fences removed and whitespace trimmed.
    """
    text = text.strip()
    text = _FENCE_START.sub("", text)
    text = _FENCE_END.sub("", text)
    return text.strip()


def _extract_text(data: object) -> str:
    """Extract the meaningful text payload from Claude CLI JSON output.

    Supports multiple output shapes:
        - ``{"result": "..."}``
        - ``{"content": [{"type": "text", "text": "..."}]}``
        - Plain string

    Args:
        data: Deserialised JSON value from Claude CLI output.

    Returns:
        The extracted text content as a string.

    Raises:
        ValueError: If the data format is not recognised (no ``result``
            or ``content`` key in a dict, or unexpected type).
    """
    if isinstance(data, dict) and "result" in data:
        return str(data["result"])

    if isinstance(data, dict) and "content" in data:
        content = data["content"]
        if not isinstance(content, list):
            # A malformed ``content`` (e.g. a bare string or a dict) would
            # otherwise iterate character-by-character or by dict key and
            # quietly miss every text block. Reject the shape up front so
            # the caller gets a clear signal instead of a silent "No text
            # block found".
            raise ValueError(f"'content' must be a list, got {type(content).__name__}.")
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block["text"])
        raise ValueError("No text block found in 'content' array")

    if isinstance(data, str):
        return data

    raise ValueError(
        f"Unexpected Claude CLI output type: {type(data).__name__}. "
        f"Expected dict with 'result' or 'content' key, or plain string."
    )


def parse_agent_output(input_path: str, output_path: str) -> None:
    """Read raw Claude CLI output, extract and validate JSON, write result.

    Args:
        input_path:  Path to the raw Claude CLI JSON output file.
        output_path: Destination path for the cleaned JSON.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        json.JSONDecodeError: If the extracted text is not valid JSON.
        ValueError: If content extraction fails or file exceeds size limit.
    """
    file_size = os.path.getsize(input_path)
    if file_size > MAX_INPUT_FILE_SIZE:
        raise ValueError(
            f"Input file {input_path} is {file_size} bytes, "
            f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes."
        )

    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    text = _extract_text(data)
    text = _strip_code_fences(text)

    # Validate that the cleaned text is valid JSON.
    parsed = json.loads(text)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(parsed, fh, indent=2)
        fh.write("\n")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        print(
            "Usage: parse_agent_output.py <input_file> <output_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        parse_agent_output(input_path, output_path)
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
