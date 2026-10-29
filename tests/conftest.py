import pytest


@pytest.fixture(autouse=True)
def _pbs_account(monkeypatch):
    """Give every test a PBS account; credit has no built-in fallback (issue #486).

    Tests that exercise the missing-account path delete it with ``monkeypatch.delenv``.
    """
    monkeypatch.setenv("PBS_ACCOUNT", "NAML0001")
