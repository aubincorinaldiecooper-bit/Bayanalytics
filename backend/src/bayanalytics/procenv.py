"""Allow-listed environment for third-party child processes.

The backend spawns three kinds of children: the Laya Node worker, ``llama-server`` and the
whisper.cpp / ffmpeg command-line tools. None of them needs the backend's own configuration,
and every ``BAY_*`` value, ``DATABASE_URL`` and any API key in the parent environment would be
readable by that third-party code (and by anything it loads) if the environment were inherited
wholesale. ``child_env`` therefore builds the child's environment from a fixed allow list:

- process basics: ``PATH``, ``HOME``, ``USER``, ``LOGNAME``, ``SHELL``, temp and locale settings,
  ``TZ``;
- TLS and proxy settings (both spellings, plus Node's ``NODE_EXTRA_CA_CERTS``), so bundle
  downloads work through a corporate proxy;
- Hugging Face cache and endpoint settings, and ``HF_TOKEN`` when it is set (the Laya bundle
  download is the only child that uses it; it is never injected, only forwarded);
- Laya / Node knobs (``LAYA_CACHE``, ``LAYA_MODULE``, ``NODE_OPTIONS``, ``NODE_PATH``);
- thread and library-path knobs (``OMP_NUM_THREADS``, ``LLAMA_ARG_*``, ``LD_LIBRARY_PATH``,
  ``DYLD_LIBRARY_PATH``).

Everything else is dropped. Callers add call-specific values through ``extra`` and can forward
further parent variables by name through ``keep``.

``FORCED_ENV`` is then applied last and cannot be overridden: it switches off the usage
reporting that bundled third-party libraries would otherwise send (ONNX Runtime, loaded by the
Laya worker, reports to Microsoft from Linux unless ``ORT_DISABLE_TELEMETRY`` is set). The backend
makes no outbound connection the operator has not approved.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

SAFE_ENV_VARS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HF_ENDPOINT",
        "LAYA_CACHE",
        "LAYA_MODULE",
        "NODE_OPTIONS",
        "NODE_PATH",
        "NODE_EXTRA_CA_CERTS",
        "OMP_NUM_THREADS",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
    }
)
"""Parent variables forwarded to every child process (only when set)."""

SAFE_ENV_PREFIXES: tuple[str, ...] = ("LLAMA_ARG_",)
"""Prefixes whose variables are forwarded (llama.cpp reads its defaults from ``LLAMA_ARG_*``)."""

FORCED_ENV: dict[str, str] = {
    "ORT_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
"""Set in every child's environment after everything else: usage reporting stays off."""


def is_forwarded(name: str, keep: Iterable[str] = ()) -> bool:
    """True when a parent variable called ``name`` is forwarded by :func:`child_env`."""
    if name in SAFE_ENV_VARS or name in set(keep):
        return True
    return name.startswith(SAFE_ENV_PREFIXES)


def child_env(
    extra: Mapping[str, str] | None = None, *, keep: Iterable[str] = ()
) -> dict[str, str]:
    """Environment for a third-party child: the allow list, ``keep`` names, ``extra``, then
    ``FORCED_ENV``.

    Variables are copied from ``os.environ`` only when they exist there. ``extra`` values are
    added verbatim and override forwarded ones (a caller injecting ``LAYA_MODULE`` wins over an
    inherited one); ``FORCED_ENV`` overrides both.
    """
    wanted = set(keep)
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in SAFE_ENV_VARS or name in wanted or name.startswith(SAFE_ENV_PREFIXES):
            env[name] = value
    if extra:
        for name, value in extra.items():
            env[str(name)] = str(value)
    env.update(FORCED_ENV)
    return env


__all__ = ["FORCED_ENV", "SAFE_ENV_PREFIXES", "SAFE_ENV_VARS", "child_env", "is_forwarded"]
