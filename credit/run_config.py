"""Keep ``save_loc/model.yml`` in sync with the config used for a run.

``model.yml`` is read by downstream scripts (rollout, evaluation, checkpoint loading), so it must reflect the
most recent run's config. Every change is recorded in a git repository that CREDIT creates inside ``save_loc``
(``git -C <save_loc> log -p model.yml``). Git tracking is best-effort: the copy always happens, and any git
failure only logs a warning.
"""

import filecmp
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# The .gitignore ignores itself too, so it never shows up as an untracked file in `git status`
_GIT_IGNORE = "*\n!model.yml\n"
# Local git config key marking a repo as created (and therefore safe to commit to) by CREDIT
_GIT_MARKER = "credit.managed"


def _git(save_loc: str, *args: str, identity: bool = False) -> subprocess.CompletedProcess:
    """Run one ``git`` command inside ``save_loc`` and return the completed process (never raises on failure).

    Args:
        save_loc (str): Directory to run git in.
        *args (str): Arguments after ``git``.
        identity (bool): Also set a fallback commit author (used when the user has no git identity).
    """
    # GPG signing is disabled so a user's global signing setup cannot hang a batch job.
    cmd = ["git", "-c", "commit.gpgsign=false"]
    if identity:
        cmd += ["-c", "user.name=credit", "-c", "user.email=credit@localhost"]
    return subprocess.run(cmd + list(args), cwd=save_loc, capture_output=True, text=True, timeout=60)


def _git_commit_model_yml(save_loc: str, message: str) -> str | None:
    """Stage ``model.yml`` and commit it if it differs from HEAD.

    Args:
        save_loc (str): Run directory containing the git repo.
        message (str): Commit message.

    Returns:
        str | None: The short commit hash, or None if there was nothing to commit.

    Raises:
        RuntimeError: If ``git add`` or ``git commit`` fails.
    """
    # -f is required because the repo's .gitignore ignores everything except model.yml and itself
    res = _git(save_loc, "add", "-f", "model.yml")
    if res.returncode != 0:
        raise RuntimeError(f"`git add` failed: {res.stderr.strip()}")
    if _git(save_loc, "diff", "--cached", "--quiet").returncode == 0:
        return None
    # Only fall back to a placeholder author when the user has no git identity configured
    has_identity = all(_git(save_loc, "config", k).stdout.strip() for k in ("user.name", "user.email"))
    res = _git(save_loc, "commit", "-q", "-m", message, "--", "model.yml", identity=not has_identity)
    if res.returncode != 0:
        raise RuntimeError(f"`git commit` failed: {res.stderr.strip() or res.stdout.strip()}")
    return _git(save_loc, "rev-parse", "--short", "HEAD").stdout.strip()


def sync_run_config(config_path: str, save_loc: str) -> str:
    """Copy ``config_path`` to ``save_loc/model.yml`` and record the change in a git repo inside ``save_loc``.

    The saved ``model.yml`` always reflects the most recent run; previous versions are kept in git history
    (``git -C <save_loc> log -p model.yml``). Git must be installed and on PATH. Tracking is best-effort: if git
    is missing or fails, a warning is logged and ``model.yml`` is still copied.

    Call from a single rank only (concurrent calls would race on the copy and on git's index lock).

    Args:
        config_path (str): The config file used for this run.
        save_loc (str): The run directory (created if missing).

    Returns:
        str: Path to ``save_loc/model.yml``.
    """
    save_loc = os.path.expandvars(save_loc)
    os.makedirs(save_loc, exist_ok=True)
    dest = os.path.join(save_loc, "model.yml")

    exists = os.path.exists(dest)
    same_file = exists and os.path.realpath(config_path) == os.path.realpath(dest)
    replaced = exists and not same_file and not filecmp.cmp(config_path, dest, shallow=False)

    def warn_no_history(reason: str) -> None:
        logger.warning(
            "%s. git is required to track config history in %s; model.yml was still copied but its history "
            "is NOT being recorded. Install git (e.g. `module load git` on Casper/Derecho).",
            reason,
            save_loc,
        )

    use_git = False
    if shutil.which("git") is None:
        warn_no_history("git was not found on PATH")
    else:
        try:
            if not os.path.exists(os.path.join(save_loc, ".git")):
                res = _git(save_loc, "init", "-q")
                if res.returncode != 0:
                    raise RuntimeError(f"`git init` failed: {res.stderr.strip()}")
                res = _git(save_loc, "config", "--local", _GIT_MARKER, "true")
                if res.returncode != 0:
                    raise RuntimeError(f"`git config` failed: {res.stderr.strip()}")
                with open(os.path.join(save_loc, ".gitignore"), "w") as f:
                    f.write(_GIT_IGNORE)
            if _git(save_loc, "config", "--local", "--get", _GIT_MARKER).stdout.strip() != "true":
                # save_loc is itself someone's repo (e.g. the repo root): never commit onto their branch.
                logger.warning(
                    "%s is a git repository not created by CREDIT; skipping config history tracking there. "
                    "model.yml was still copied. Consider setting save_loc to a directory outside the "
                    "miles-credit repo (or any other git repo), e.g. under your scratch space.",
                    save_loc,
                )
            else:
                if replaced:
                    # Preserve whatever is on disk (e.g. a run dir created before git tracking) before overwriting.
                    _git_commit_model_yml(save_loc, "Previous model.yml (before this run)")
                use_git = True
        except Exception as e:
            warn_no_history(str(e))

    if not same_file:
        shutil.copy(config_path, dest)
    if replaced:
        logger.warning("%s already existed with different contents; replaced by %s.", dest, config_path)

    if use_git:
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            commit = _git_commit_model_yml(save_loc, f"Update model.yml ({stamp})")
            if commit:
                logger.info("Recorded %s in git (commit %s).", dest, commit)
        except Exception as e:
            warn_no_history(str(e))
    return dest
