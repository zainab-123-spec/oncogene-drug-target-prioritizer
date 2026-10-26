"""Cache, retry, and rate-limit behavior of the HTTP client."""

from __future__ import annotations

import json

import pytest
import requests

from odtp.http import HttpClient, RequestError
from tests.conftest import json_response


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedSession:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def request(self, method: str, url: str, params=None, data=None, headers=None, timeout=None):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_cache_avoids_second_request(tmp_path) -> None:
    session = ScriptedSession([json_response({"ok": True})])
    client = HttpClient(tmp_path, session=session, sleep=lambda _: None, time_fn=lambda: 0.0)
    first = client.get_json("https://example.test/item", params={"q": "egfr"})
    second = client.get_json("https://example.test/item", params={"q": "egfr"})
    assert first == {"ok": True}
    assert second == {"ok": True}
    assert session.calls == 1
    assert client.cache_hits == 1


def test_retries_then_succeeds(tmp_path) -> None:
    session = ScriptedSession(
        [
            json_response({"err": True}, status=503),
            json_response({"err": True}, status=503),
            json_response({"ok": 1}),
        ]
    )
    clock = Clock()
    client = HttpClient(
        tmp_path,
        session=session,
        min_interval=0,
        sleep=clock.sleep,
        time_fn=clock.time,
        max_retries=4,
    )
    payload = client.get_json("https://example.test/retry")
    assert payload == {"ok": 1}
    assert session.calls == 3
    assert clock.sleeps


def test_rate_limiter_waits_between_calls(tmp_path) -> None:
    session = ScriptedSession([json_response({"n": 1}), json_response({"n": 2})])
    clock = Clock()
    client = HttpClient(
        tmp_path,
        session=session,
        min_interval=0.25,
        sleep=clock.sleep,
        time_fn=clock.time,
    )
    client.get_json("https://example.test/a")
    client.get_json("https://example.test/b")
    assert clock.sleeps
    assert clock.sleeps[0] == pytest.approx(0.25)


def test_does_not_cache_http_errors(tmp_path) -> None:
    session = ScriptedSession([json_response({"no": True}, status=404)])
    client = HttpClient(tmp_path, session=session, sleep=lambda _: None, time_fn=lambda: 1.0)
    with pytest.raises(RequestError):
        client.get_json("https://example.test/missing")
    assert list(tmp_path.glob("*.json")) == []


def test_graphql_error_without_data_is_not_cached(tmp_path) -> None:
    session = ScriptedSession([json_response({"errors": [{"message": "bad query"}]})])
    client = HttpClient(tmp_path, session=session, sleep=lambda _: None, time_fn=lambda: 1.0)
    with pytest.raises(RequestError):
        client.post_json("https://example.test/graphql", {"query": "{ disease }"})
    assert list(tmp_path.glob("*.json")) == []


def test_connection_error_retries(tmp_path) -> None:
    session = ScriptedSession([requests.ConnectionError("reset"), json_response({"ok": True})])
    clock = Clock()
    client = HttpClient(
        tmp_path,
        session=session,
        min_interval=0,
        sleep=clock.sleep,
        time_fn=clock.time,
        max_retries=3,
    )
    assert client.get_json("https://example.test/flake") == {"ok": True}
    assert session.calls == 2


def test_cached_payload_roundtrip_is_json(tmp_path) -> None:
    session = ScriptedSession([json_response({"values": [1, 2, 3]})])
    client = HttpClient(tmp_path, session=session, sleep=lambda _: None, time_fn=lambda: 5.0)
    client.get_json("https://example.test/list")
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored == {"values": [1, 2, 3]}
