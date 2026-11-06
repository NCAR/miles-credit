import pytest


@pytest.fixture(autouse=True)
def _pbs_account(monkeypatch):
    """Give every test a PBS account and no SLURM account; credit has no built-in fallback (issue #486).

    Tests that exercise the missing-account path delete it with ``monkeypatch.delenv``.
    """
    monkeypatch.setenv("PBS_ACCOUNT", "TEST0000")
    # A developer's exported $SLURM_ACCOUNT (e.g. on NERSC) would otherwise leak into SLURM script tests.
    monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
