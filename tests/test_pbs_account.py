"""Account resolution in the gen1 PBS launchers (credit/pbs.py); see issue #486."""

import pytest
import yaml

from credit.pbs import _pbs_account, launch_script, launch_script_mpi, launch_script_torchrun


@pytest.mark.parametrize(
    ("pbs_options", "env_account", "expected"),
    [
        ({"project": "PROJ0001"}, "ENV0001", "PROJ0001"),  # config beats $PBS_ACCOUNT
        ({"account": "ALIAS0001"}, "ENV0001", "ALIAS0001"),  # `account` is an alias for `project`
        ({"project": "PROJ0001", "account": "ALIAS0001"}, None, "PROJ0001"),  # `project` wins over the alias
        ({}, "ENV0001", "ENV0001"),
        ({"project": ""}, "ENV0001", "ENV0001"),  # an empty key counts as unset
        ({"project": "  "}, "ENV0001", "ENV0001"),  # so does a whitespace-only one
        ({"project": " PROJ0001 "}, None, "PROJ0001"),  # values are stripped
    ],
)
def test_pbs_account_resolution(monkeypatch, pbs_options, env_account, expected):
    if env_account:
        monkeypatch.setenv("PBS_ACCOUNT", env_account)
    else:
        monkeypatch.delenv("PBS_ACCOUNT", raising=False)
    assert _pbs_account(pbs_options) == expected


@pytest.mark.parametrize("env_account", [None, "", "  "])
def test_pbs_account_missing_raises(monkeypatch, env_account):
    if env_account is None:
        monkeypatch.delenv("PBS_ACCOUNT", raising=False)
    else:
        monkeypatch.setenv("PBS_ACCOUNT", env_account)
    with pytest.raises(ValueError, match="PBS_ACCOUNT"):
        _pbs_account({})


def _write_config(tmp_path, pbs):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"save_loc": str(tmp_path / "run"), "pbs": pbs}))
    return str(path)


_LAUNCHERS = [launch_script, launch_script_mpi, launch_script_torchrun]
_PBS = {
    "conda": "credit",
    "ngpus": 1,
    "ncpus": 8,
    "mem": "64GB",
    "walltime": "01:00:00",
    "queue": "casper",
    "job_name": "t",
}


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda f: f.__name__)
def test_launchers_raise_without_account(tmp_path, monkeypatch, launcher):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PBS_ACCOUNT", raising=False)
    (tmp_path / "run").mkdir()
    with pytest.raises(ValueError, match="PBS_ACCOUNT"):
        launcher(_write_config(tmp_path, _PBS), "train.py", launch=False)


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(("pbs_extra", "expected"), [({}, "ENV0001"), ({"account": "ALIAS0001"}, "ALIAS0001")])
def test_launchers_write_resolved_account(tmp_path, monkeypatch, launcher, pbs_extra, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PBS_ACCOUNT", "ENV0001")
    (tmp_path / "run").mkdir()
    launcher(_write_config(tmp_path, {**_PBS, **pbs_extra}), "train.py", launch=False)
    assert f"#PBS -A {expected}" in (tmp_path / "launch.sh").read_text()


@pytest.mark.parametrize(
    ("options", "env_account", "expected"),
    [
        ({"project": "m0000"}, "ENV_SLURM", "#SBATCH --account=m0000"),  # config beats $SLURM_ACCOUNT
        ({"account": "m0000"}, None, "#SBATCH --account=m0000"),  # alias
        ({"project": ""}, "ENV_SLURM", "#SBATCH --account=ENV_SLURM"),  # empty counts as unset
        ({"constraint": "gpu"}, "m0000", "#SBATCH --account=m0000_g"),  # Perlmutter _g suffix on env value
        ({}, None, None),  # nothing set: omit --account rather than charge a literal "default"
    ],
)
def test_gen1_slurm_account(monkeypatch, options, env_account, expected):
    from credit.slurm import _sbatch_directives

    monkeypatch.delenv("NERSC_HOST", raising=False)
    if env_account:
        monkeypatch.setenv("SLURM_ACCOUNT", env_account)
    else:
        monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
    lines = [ln for ln in _sbatch_directives(options, 1, 4).splitlines() if ln.startswith("#SBATCH --account")]
    assert lines == ([expected] if expected else [])


def test_shipped_configs_do_not_hard_code_an_account():
    """No config under config/ may set a scheduler account (issue #486).

    A hard-coded project silently outranks the user's $PBS_ACCOUNT / $SLURM_ACCOUNT and charges
    someone else's allocation. Ship it commented out, e.g. ``# project: YOUR_ACCOUNT``.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config"
    offenders = []
    for path in sorted(root.rglob("*.y*ml")):
        try:
            conf = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(conf, dict):
            continue
        for block in ("pbs", "slurm"):
            section = conf.get(block)
            if isinstance(section, dict):
                for key in ("project", "account"):
                    if section.get(key):
                        offenders.append(f"{path.relative_to(root.parent)}: {block}.{key}={section[key]!r}")
    assert not offenders, "Hard-coded scheduler accounts:\n" + "\n".join(offenders)
