"""Replay recorded Ollama exchanges.

The transport loads one version directory. A fixture whose behaviour is
``no_reply`` fails the call. Until the Publisher commits a version directory,
tests replay the catalog's own documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from interlock.adapters.ollama_cases import RecordedNoReply, request_path, require_recorded_behaviour


def replay(document: dict) -> dict:
    """Return the recorded response, or fail when the recording did not answer."""

    if document.get("behaviour") == "no_reply":
        raise RecordedNoReply(f"recorded behaviour is no reply ({document.get('case')})")
    response = document.get("response")
    if not isinstance(response, dict):
        raise AssertionError(f"{document.get('case')} has no response object")
    return response


class ReplayTransport:
    """A transport that answers from a recorded version directory."""

    def __init__(self, documents: list[dict]):
        self.documents = list(documents)

    @classmethod
    def load(cls, folder: Path) -> ReplayTransport:
        documents = []
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                documents.append(json.loads(path.read_text(encoding="utf-8")))
        return cls(documents)

    def __call__(self, method: str, url: str, body):
        require_recorded_behaviour(method, url, body)
        path = request_path(url)
        matches = [
            document
            for document in self.documents
            if document.get("request", {}).get("method") == method
            and document.get("request", {}).get("path") == path
            and document.get("request", {}).get("body") == body
        ]
        if not matches:
            raise AssertionError(f"no fixture for {method} {path}")
        for document in matches:
            if document.get("behaviour") == "no_reply":
                raise RecordedNoReply(f"recorded behaviour is no reply ({document.get('case')})")
        response = matches[0].get("response")
        if not isinstance(response, dict):
            raise AssertionError(f"no response object for {method} {path}")
        return response
