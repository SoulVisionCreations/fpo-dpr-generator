"""Logging setup for the service.

Without this, `logging.getLogger("dpr.serve")` has no handler under uvicorn and
every structured access line is silently discarded — the failure mode this module
exists to prevent.

Everything goes to stdout as one line per event, which is what a container log
driver expects.
"""
from __future__ import annotations

import json
import logging
import os
import sys

LEVEL = os.environ.get("DPR_LOG_LEVEL", "INFO").upper()
JSON_LOGS = os.environ.get("DPR_LOG_JSON", "1") != "0"


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record.

    The access-log middleware already formats its payload as JSON, so pass that
    through as `event` rather than nesting an escaped string inside a string.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
        }
        if msg.startswith("{") and msg.endswith("}"):
            try:
                entry["event"] = json.loads(msg)
            except ValueError:
                entry["message"] = msg
        else:
            entry["message"] = msg
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure() -> None:
    """Idempotent: safe to call from both lifespan and a test fixture."""
    root = logging.getLogger("dpr")
    if getattr(root, "_dpr_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if JSON_LOGS else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]
    root.setLevel(LEVEL)
    # Our handler is the only one that should print these; without this the root
    # logger would emit a second, unformatted copy of every line.
    root.propagate = False
    root._dpr_configured = True
