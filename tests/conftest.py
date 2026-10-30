"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_credit_home(tmp_path_factory, monkeypatch):
    """Point the run ledger (``$CREDIT_HOME``) at a temp dir so no test touches ``~/.credit``."""
    monkeypatch.setenv("CREDIT_HOME", str(tmp_path_factory.mktemp("credit_home")))
