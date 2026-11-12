# Verifying findings in CREDIT

## Setup

Use a scratch directory outside the repo (`$SCRATCH` below), e.g. the agent's scratchpad.

```bash
# PR worktree, next to the user's checkout (never switch branches in their checkout)
git fetch origin
git worktree add --detach ../miles-credit-pr<N> origin/main
(cd ../miles-credit-pr<N> && gh pr checkout <N>)

# Clean base worktree, for "does this also happen on main?" comparisons
git worktree add --detach $SCRATCH/base-wt origin/<base>

# CPU-only environment, like CI (Python >= 3.11; CI tests 3.11-3.13)
uv venv -p 3.12 $SCRATCH/credit-env
VIRTUAL_ENV=$SCRATCH/credit-env uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=$SCRATCH/credit-env uv pip install -e "../miles-credit-pr<N>[develop]"
```

Reuse an existing environment if one is already set up. Use `SKIP_REMOTE=1` unless the PR touches remote
datasets; several dataset tests stream from AWS/GCS and are slow or fail offline.

**Running against the base branch.** The environment is an editable install of the PR, so a plain run
imports the PR's code even from another directory. Point Python at the base worktree explicitly and check:

```bash
PYTHONPATH=$SCRATCH/base-wt python -c "import credit; print(credit.__file__)"   # must print base-wt
```

## Techniques

Pick the ones that fit the candidate. The first two apply to almost every PR.

1. **Mutation test each source hunk.** Copy the PR tree (`rsync -a --exclude .git`), revert one hunk (or
   replace the new logic with the old behavior), and run the PR's tests with `PYTHONPATH` pointing at the
   copy. If the tests still pass, that hunk is untested: run an experiment to show what breaks without it,
   and report it as a coverage finding with that consequence.
   - "The new tests fail on main" is weak evidence when they fail only because a new symbol does not
     exist yet (ImportError). Mutate the behavior, not the existence.
2. **Sweep the options the change touches.** If the changed path depends on a config choice
   (`scheduler_type`, `model.type`, parallel mode, `dataset_type`, AMP on/off), write a short script that
   runs the scenario for every option, on the PR and on base. Report the table. Untested options are
   where gaps hide.
3. **Use a real model, not only the toy one in the tests.** Build one from a repo config with
   `credit.models.load_model(credit.parser.credit_main_parser(conf))`. Small configs that build on CPU:
   `config/gen_1/applications/other_models/wxformer_1dg_test.yml` (CrossFormer, spectral norm on) and the
   other `*_test.yml` files there. Note that model test configs live under `config/`, not `tests/`.
4. **Round-trip training state.** For anything near checkpoints, optimizers, schedulers or preflight:
   train N steps, save, resume through `credit.trainers.utils.load_model_states_and_optimizer`, train N more,
   and compare losses with an uninterrupted run (`rel=1e-6`). Also try a checkpoint shaped like one an
   **older version** wrote (e.g. a state-dict key removed) — users resume existing runs.
5. **Trace the caller before claiming a distributed bug.** `base_trainer.py` skips some code for wrapped or
   sharded models (e.g. the GPU preflight). A path that never runs under DDP/FSDP cannot have a DDP bug.
6. **Config-level checks.** For config, registry or schema changes, run `credit check -c <config>` on every
   example config the PR touches or should touch (`config/gen_2/examples/`), and a deliberately broken one
   to see that the new rule fires.
7. **Docs build.** For docstring or `docs/` changes, build the docs if the environment allows it; RST errors
   surface on the generated `docs/source/api/` pages.

Some distributed logic **can** be checked on CPU: multi-process `gloo` tests (`tests/test_fsdp2_parallel.py`,
`tests/test_native_tp.py`, `tests/test_ring_crps.py`) show the pattern; copy it for a 2-process repro.
What CPU cannot verify: NCCL-specific hangs, multi-node behavior, real GPU memory numbers, PBS behavior.
Label those PLAUSIBLE, and suggest the matching manual test under `tests/manual/`.

Keep every script you write in `$SCRATCH`: they are the evidence, and the author can be pointed to them.

## Cleanup

```bash
git worktree remove ../miles-credit-pr<N>        # after the PR merges, if a re-review was possible
git worktree remove $SCRATCH/base-wt
```
