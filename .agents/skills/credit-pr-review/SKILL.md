---
name: credit-pr-review
description: Review a CREDIT (NCAR/miles-credit) pull request like a senior maintainer and verify every finding by running code before reporting it. Use when asked to review, check or comment on a miles-credit PR by number, branch or URL.
---

# CREDIT PR review (self-verifying)

Review the pull request given as the argument (number, branch or URL). The goal is a review that is
**deep but short**, where every finding has been **checked by running code**, not just by reading it.
Read `AGENTS.md` at the repo root first: it holds the architecture and the repo rules this review
checks against.

Work in four stages. Do not skip stage 3: a finding that has not been through it is not reported as fact.

## Stage 0 — Set up (once per PR)

Follow [references/verification.md](references/verification.md) §Setup: a worktree for the PR, a clean
worktree of the base branch, and a CPU-only Python environment. Never switch branches in the user's
own checkout.

PRs from forks run someone else's code on this machine. For an unfamiliar author, say so and ask
before running anything.

## Stage 1 — Context

1. `gh pr view <n>` (body, author, commits, `closingIssuesReferences`) and every linked issue with its
   comments. Note what the PR **claims** to do: you will check the code against those claims.
2. **Review history** — follow [references/history.md](references/history.md): earlier reviews and the commit
   each one reviewed, every inline thread and comment, commits since the last review, description edits,
   force-pushes, CI status. Build the ledger of points raised and their status, and note disagreements
   and unanswered questions. The rest of the review builds on this, so don't skip it even when the PR
   looks fresh.
3. `git diff <base>...HEAD --stat`, then the full diff. If there was an earlier review, also the diff since
   that review's commit: it is the least-reviewed code in the PR.
4. For every changed function, **trace its callers up to an entry point** (trainer `fit`, a
   `credit/applications/*` main, a CLI subcommand). Record:
   - gen1, gen2 or both (see AGENTS.md "gen1 vs gen2"); shared code such as `base_trainer.py` affects both;
   - which parallelism modes reach it (`none`/`ddp`/`fsdp`/`fsdp2`, tensor/domain > 1), and which are
     skipped by the caller;
   - which config options select the path (registry keys, `scheduler_type`, `trainer.*` flags).
   This step decides which findings are possible at all. Skipping it produces false positives.

## Stage 2 — Find candidates

Look at the diff along these dimensions. Use parallel subagents (one per dimension) when the PR is
large; for small PRs, do it in one pass.

- **Correctness and numerics**: wrong axis/dtype/device, in-place mutation of shared state, NaN/inf paths,
  off-by-one in `forecast_len` (gen2 is 1-indexed), normalization applied twice or not at all.
- **Training state**: anything touching the model, optimizer, scheduler, scaler, EMA or RNG must survive
  save → resume exactly. Checkpoints written by older versions must still load.
- **Distributed**: rank-0-only code paths, collectives that not every rank joins, DDP/FSDP wrapping.
- **CREDIT contracts** (AGENTS.md): registry entries registered, not special-cased; a new registry key,
  config key or structural rule needs a matching check in `credit check`; preblock chains end with
  `concat`, postblock chains start with `reconstruct`; `bridgescaler` fitting sets `channels_last: False`.
- **API and back-compat**: config schema changes, renamed keys, default changes, `credit convert` impact.
- **Tests**: do the new tests fail without the fix (stage 3 checks this)? Are all options of a touched
  code path covered (every scheduler type, every parallel mode the caller allows)?
- **Docs and PR hygiene**: matching `docs/source/*.md` page updated (AGENTS.md "Docs"); docstrings valid
  RST; PR description matches the code; the PR title is release-note quality (it goes into release notes).
- **Scientific-software practice**: reproducibility (seeds, determinism), units, performance regressions,
  readability, maintainability.

For each candidate, write down: the claim, `file:line`, the concrete failure scenario, and how you will
verify it. Candidates are hypotheses, not findings.

Check each candidate against the ledger from Stage 1: drop it if someone already raised it, unless you
have new evidence, in which case it becomes a reply in that thread or a line in the summary.

## Stage 3 — Verify

Apply the techniques in [references/verification.md](references/verification.md) §Techniques. Then label
each candidate:

- **CONFIRMED** — reproduced by running code (a failing test, a mutation that survives, an experiment).
- **PLAUSIBLE** — the code path is real but cannot be run here (e.g. needs multiple GPUs or HPC data).
  State why it cannot be run.
- **Dropped** — disproved. Keep a one-line note of why: dropped candidates show the review was thorough
  and stop the next reviewer from raising them again.

Always run, whatever the candidates: the PR's new tests on the PR; a **mutation test of each source hunk**
(revert it, rerun the tests; a hunk whose reversion no test notices is a coverage finding); `ruff check`
and `ruff format --check` on changed files.

## Stage 4 — Report

Write the draft to a file in the scratchpad using [references/report.md](references/report.md), then
**show it to the user and wait**. Post only after explicit approval, exactly as approved, following
report.md §Posting. Then clean up per verification.md §Cleanup (keep the PR worktree until the PR merges
if a re-review is likely).
