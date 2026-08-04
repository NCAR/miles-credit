"""Shared conda environment resolution for generated job scripts.

The PBS (``credit/pbs.py``), Slurm (``credit/slurm.py``), and ``credit`` CLI
(``credit/cli/_common.py``) launchers all emit shell scripts that must invoke
``torchrun`` from a specific conda environment. They previously each carried
their own copy of the resolution logic, which drifted. This module holds the
single definition.

Kept deliberately dependency-free (stdlib only, no ``credit`` imports) so the
launchers can import it without pulling in the CLI package.
"""


def conda_prefix_expr(conda_env: str) -> str:
    """Return a shell expression that expands to the prefix of *conda_env*.

    A value containing a path separator is already a prefix and is returned as
    is. A bare environment *name* is resolved at run time, never at generation
    time, so it is not mistaken for a same-named directory in the current
    working directory (e.g. the repo's ``credit/`` package dir, which made a
    ``conda: credit`` config yield a bogus ``credit/bin/torchrun``).

    Resolution order in the generated script:

    1. ``$CONDA_PREFIX`` — set by the ``conda activate`` line the scripts emit
       just above this, and correct no matter where the env actually lives.
    2. ``conda info --envs`` lookup by name — covers scripts that never
       activate. This honors ``envs_dirs`` from ``~/.condarc``, unlike
       ``$(conda info --base)/envs/<name>``, which silently produced a
       nonexistent path for envs kept outside the base install (e.g.
       ``/glade/work/$USER/conda-envs``).

    Args:
        conda_env: A conda environment name (e.g. ``"credit"``) or a full
            environment prefix path (e.g. ``"/glade/work/me/envs/credit"``).

    Returns:
        A shell expression that expands to the environment prefix.
    """
    if "/" in conda_env:
        return conda_env
    lookup = f"conda info --envs | awk -v n={conda_env} '$1==n {{print $NF; exit}}'"
    return f"${{CONDA_PREFIX:-$({lookup})}}"


def torchrun_path(conda_env: str) -> str:
    """Return a shell expression for ``torchrun`` inside *conda_env*.

    Args:
        conda_env: A conda environment name or full prefix path, as accepted by
            :func:`conda_prefix_expr`.

    Returns:
        The ``<prefix>/bin/torchrun`` path, with the prefix resolved at run time
        for a bare environment name.
    """
    return f"{conda_prefix_expr(conda_env)}/bin/torchrun"
