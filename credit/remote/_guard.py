"""Policy, secret redaction and the audit log for ``credit.remote`` (stdlib + yaml only).

The policy lives in ``remote.yaml`` files that can only *narrow* what the
remote serves::

    roots:            # replaces the default roots (home, work, scratch)
      - /glade/work/$USER/CREDIT_runs
      - /glade/work/$USER/configs
    redact: true      # mask secrets in text sent back (default true)
    audit: true       # append one line per call to $CREDIT_HOME/audit.jsonl (default true)

Two files are read, ``~/.credit/remote.yaml`` in the account's real home
(from the password database, so an environment change can't skip it) and
``$CREDIT_HOME/remote.yaml``.  Each one that sets ``roots`` narrows the result
further; ``redact: false`` or ``audit: false`` takes effect only when no file
says otherwise.
"""

import os
import pwd
import re

import yaml

from credit import runs

MARK = "«redacted»"

_TOKEN_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
]
# Credentials in URLs (scheme://user:secret@host) and Authorization headers keep their prefix.
_URL_PASSWORD = re.compile(r"(\b[a-zA-Z][\w+.-]*://[^/\s:@]+:)([^@\s/]+)(@)")
_BEARER = re.compile(r"(?i)(\bauthorization\s*[:=]\s*['\"]?bearer\s+)([^\s'\"]+)")
# key: value / key=value where the key names a secret.  The value must contain a letter and be at
# least 6 characters, so `token: anon`, `max_tokens: 100000` and `secret: null` are left alone.
_KEY_VALUE = re.compile(
    r"(?i)(\b[\w.-]*(?:password|passwd|secret|api[_-]?key|access[_-]?key|private[_-]?key|token|credential)[\w.-]*"
    r"\s*[:=]\s*['\"]?)(?!«)(?=[^\s'\"#,}]*[A-Za-z])([^\s'\"#,}]{6,})"
)


def owner_home() -> str:
    """The account's home directory from the password database, ignoring ``$HOME``."""
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return os.path.expanduser("~")


def _policy_files() -> list:
    paths = [os.path.join(owner_home(), ".credit", "remote.yaml"), os.path.join(runs.credit_home(), "remote.yaml")]
    out = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in out and os.path.isfile(rp):
            out.append(rp)
    return out


def policy() -> dict:
    """Merge the policy files: ``roots`` narrow in turn; ``redact``/``audit`` are true unless every file says false."""
    roots = None
    flags = {"redact": [], "audit": []}
    for path in _policy_files():
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            data = {"roots": []}  # an unreadable policy fails closed
        if not isinstance(data, dict):
            data = {"roots": []}
        for key in flags:
            if key in data:
                flags[key].append(bool(data[key]))
        if "roots" in data:
            listed = [os.path.realpath(runs.expand(str(r))) for r in (data.get("roots") or [])]
            if roots is None:
                roots = listed
            else:
                roots = [r for r in listed if any(runs._under(r, base) for base in roots)] + [
                    base for base in roots if any(runs._under(base, r) for r in listed) and base not in listed
                ]
    return {
        "roots": roots,
        "redact": not flags["redact"] or any(flags["redact"]),
        "audit": not flags["audit"] or any(flags["audit"]),
        "files": _policy_files(),
    }


def redact(text):
    """Return ``(text, count)`` with secrets replaced by :data:`MARK`; non-strings pass through."""
    if not isinstance(text, str) or not text or not policy_redacts():
        return text, 0
    count = 0
    for pattern in _TOKEN_PATTERNS:
        text, n = pattern.subn(MARK, text)
        count += n
    for pattern in (_URL_PASSWORD, _BEARER):
        text, n = pattern.subn(lambda m: m.group(1) + MARK + (m.group(3) if m.lastindex >= 3 else ""), text)
        count += n
    text, n = _KEY_VALUE.subn(lambda m: m.group(1) + MARK, text)
    return text, count + n


_REDACTS = None


def policy_redacts() -> bool:
    global _REDACTS
    if _REDACTS is None:
        _REDACTS = policy()["redact"]
    return _REDACTS


def redact_event(event: dict) -> dict:
    """Mask secrets in the free-text fields of ``log`` and ``error`` events."""
    if event.get("type") in ("log", "error") and isinstance(event.get("data"), dict):
        data = dict(event["data"])
        for key in ("message", "traceback_tail"):
            data[key], _ = redact(data.get(key))
        event = {**event, "data": data}
    return event


_AUDIT_KEYS = (
    "path",
    "run_id",
    "job_key",
    "kind",
    "scope",
    "config",
    "save_loc",
    "scheduler",
    "cluster",
    "page",
    "name",
)


def audit(verb: str, req: dict, response: dict) -> None:
    """Append one line describing the call to ``$CREDIT_HOME/audit.jsonl`` (never file contents)."""
    try:
        if not policy()["audit"]:
            return
        summary = {k: str(req[k])[:300] for k in _AUDIT_KEYS if isinstance(req, dict) and req.get(k)}
        if isinstance(req, dict) and isinstance(req.get("request"), dict):  # submit
            summary.update(
                {k: str(req["request"][k])[:300] for k in ("run_id", "kind", "config") if req["request"].get(k)}
            )
        if isinstance(req, dict) and isinstance(req.get("runs"), dict):  # sync / follow
            summary["runs"] = len(req["runs"])
        line = {
            "ts": runs.now_ts(),
            "verb": verb,
            "ok": bool(response.get("ok")),
            "code": (response.get("error") or {}).get("code"),
            "client": (os.environ.get("SSH_CONNECTION") or "").split(" ")[0] or None,
            "req": summary,
        }
        runs.append_line(os.path.join(runs.credit_home(), "audit.jsonl"), line, lock=True)
    except Exception:  # noqa: BLE001 — auditing must never break a verb
        pass
