"""Line-based status protocol between the dispatcher and worker subprocesses.

Workers emit exactly one protocol line per document on **stdout**; all other
output (MinerU/vLLM logs, tracebacks) goes to **stderr** and is teed to a log
file. Each protocol line is a sentinel prefix followed by a JSON object:

    @@OMNIDOC@@ {"event": "done", "doc": "a.pdf", "pages": 10, "seconds": 12.5}

Driving progress off this unambiguous prefix — rather than regex-scanning
human-readable output — means a doc is counted exactly once, never double
counted by a summary line that happens to contain the same glyph.
"""

from __future__ import annotations

import json

PREFIX = "@@OMNIDOC@@"


def encode_status(event: str, **fields: object) -> str:
    """Encode a status event as a single protocol line (no trailing newline).

    Args:
        event: Event name, e.g. ``"done"`` or ``"failed"``.
        **fields: JSON-serialisable payload (``doc``, ``pages``, ``error``, …).

    Returns:
        A ``"@@OMNIDOC@@ {json}"`` line.
    """
    payload = {"event": event, **fields}
    return f"{PREFIX} {json.dumps(payload, ensure_ascii=False)}"


def decode_status(line: str) -> dict | None:
    """Decode a protocol line, or return ``None`` for non-protocol/junk lines.

    Args:
        line: A single line of worker output.

    Returns:
        The decoded payload dict, or ``None`` if the line is not a valid
        protocol line (so callers can safely feed it every stdout line).
    """
    line = line.strip()
    if not line.startswith(PREFIX):
        return None
    body = line[len(PREFIX):].strip()
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None
