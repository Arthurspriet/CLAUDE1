"""Best-effort JSON repair for malformed tool call arguments from small models.

Handles common issues:
- Trailing commas
- Missing closing braces/brackets
- Single-quoted strings
- Unquoted keys
- Missing quotes around string values
- Truncated JSON
"""

import json
import re


def repair_json(text: str) -> dict | list | str:
    """Try to parse JSON, repairing common issues if it fails.

    Returns the parsed object on success, or a dict with {"raw": text} on failure.
    """
    text = text.strip()
    if not text:
        return {}

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Apply repairs
    repaired = text

    # Fix single-quoted strings -> double-quoted
    repaired = _fix_single_quotes(repaired)

    # Fix trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Fix unquoted keys: {key: "value"} -> {"key": "value"}
    repaired = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r' "\1":',
        repaired,
    )

    # Balance braces/brackets
    repaired = _balance_brackets(repaired)

    # Try parsing again
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: try wrapping in braces
    if not repaired.startswith("{") and ":" in repaired:
        try:
            return json.loads("{" + repaired + "}")
        except json.JSONDecodeError:
            pass

    return {"raw": text}


def _fix_single_quotes(text: str) -> str:
    """Replace single-quoted strings with double-quoted strings.

    Handles escaped quotes and avoids replacing apostrophes within words.
    """
    result = []
    i = 0
    in_double_quote = False

    while i < len(text):
        ch = text[i]

        if ch == '"' and (i == 0 or text[i - 1] != '\\'):
            in_double_quote = not in_double_quote
            result.append(ch)
        elif ch == "'" and not in_double_quote:
            # Check if this looks like a JSON string boundary
            # (preceded by :, [, {, , or start; or followed by :, ], }, , or end)
            before = text[i - 1] if i > 0 else ''
            if before in (':', '[', '{', ',', ' ', '\n', '\t', ''):
                # Find the matching close quote
                j = text.find("'", i + 1)
                if j >= 0:
                    # Replace both quotes
                    inner = text[i + 1:j].replace('"', '\\"')
                    result.append('"')
                    result.append(inner)
                    result.append('"')
                    i = j + 1
                    continue
            result.append(ch)
        else:
            result.append(ch)
        i += 1

    return ''.join(result)


def _balance_brackets(text: str) -> str:
    """Add missing closing braces/brackets."""
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()

    # Append missing closers
    while stack:
        text += stack.pop()

    return text
