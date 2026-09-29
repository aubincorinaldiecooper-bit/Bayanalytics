"""``bayanalytics serve --host ::`` listens on IPv4 and IPv6 through one dual-stack socket."""

from __future__ import annotations

import argparse
import socket

import pytest
import uvicorn

from bayanalytics import cli


def _ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


class _FakeServer:
    """Stands in for uvicorn.Server: records the sockets instead of serving."""

    instances: list[_FakeServer] = []

    def __init__(self, config: uvicorn.Config) -> None:
        self.config = config
        self.sockets: list[object] | None = None
        self.started = True
        _FakeServer.instances.append(self)

    def run(self, sockets: list[object] | None = None) -> None:
        self.sockets = sockets


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> type[_FakeServer]:
    _FakeServer.instances = []
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(cli, "set_settings", lambda _settings: None)
    return _FakeServer


@pytest.mark.skipif(not _ipv6_available(), reason="this host has no IPv6 stack")
def test_dual_stack_socket_accepts_ipv4_and_ipv6() -> None:
    sock = cli.dual_stack_socket(0)
    try:
        assert sock.family == socket.AF_INET6
        assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
        sock.listen()
        port = sock.getsockname()[1]
        for address in ("127.0.0.1", "::1"):
            with socket.create_connection((address, port), timeout=2):
                conn, _peer = sock.accept()
                conn.close()
    finally:
        sock.close()


def test_serve_on_ipv6_any_passes_a_dual_stack_socket(
    fake_server: type[_FakeServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    ports: list[int] = []

    def fake_socket(port: int) -> object:
        ports.append(port)
        return sentinel

    monkeypatch.setattr(cli, "dual_stack_socket", fake_socket)
    assert cli._serve(argparse.Namespace(host="::", port=8123)) == 0
    assert ports == [8123]
    assert fake_server.instances[0].sockets == [sentinel]


def test_serve_on_other_hosts_lets_uvicorn_bind(
    fake_server: type[_FakeServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_socket(port: int) -> object:
        raise AssertionError("only '::' needs the dual-stack socket")

    monkeypatch.setattr(cli, "dual_stack_socket", no_socket)
    for host in ("127.0.0.1", "0.0.0.0"):
        assert cli._serve(argparse.Namespace(host=host, port=8123)) == 0
    assert [s.sockets for s in fake_server.instances] == [None, None]
    assert [s.config.host for s in fake_server.instances] == ["127.0.0.1", "0.0.0.0"]


def test_serve_reports_a_failed_startup(
    fake_server: type[_FakeServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    class NeverStarts(_FakeServer):
        def run(self, sockets: list[object] | None = None) -> None:
            self.started = False

    monkeypatch.setattr(uvicorn, "Server", NeverStarts)
    assert cli._serve(argparse.Namespace(host="127.0.0.1", port=8123)) == 3
