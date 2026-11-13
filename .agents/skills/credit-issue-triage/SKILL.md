---
name: credit-issue-triage
description: Triage a CREDIT (NCAR/miles-credit) GitHub issue like a senior maintainer - classify it, find duplicates, try to reproduce it, locate the code involved, and suggest labels and priority. Use when asked to triage, review or assess a miles-credit issue by number or URL.
---

# CREDIT issue triage

Triage the issue given as the argument. The output is a short triage note for the maintainer, plus an
optional comment for the issue thread. Read `AGENTS.md` at the repo root first for the architecture.

Nothing is posted or labelled without the user's approval.

## 1. Understand

`gh issue view <n> --comments`. Note what the reporter observed, what they expected, their environment
(gen1/gen2, config, machine, GPU count, parallel mode), and anything they already tried.

Read the discussion so far as carefully as the report: who has already diagnosed or reproduced it, what was
asked of the reporter and whether they answered, existing labels/assignees, and PRs or issues that mention
it (`gh api repos/NCAR/miles-credit/issues/<n>/timeline` → `cross-referenced` events). Build on that
rather than repeating it, and list unanswered questions or disagreements under **Needs your call**.

## 2. Classify

- **Type**: bug, feature request, question/support, docs, performance, CI/infra, HPC/environment
  (modules, conda env, MPI/NCCL or PBS on Casper/Derecho; often not a CREDIT code bug).
- **Security-sensitive** (exposed credentials, a vulnerability): stop here. Do not draft a public comment
  or reproduce it in public; tell the user, who decides how to handle it privately.
- **Area**: which subsystem (dataset, preblock, model, postblock, loss, metric, trainer, rollout, CLI,
  PBS/HPC, docs) and which generation (gen1, gen2, both).
- **Labels**: suggest only from the repo's existing labels (`gh label list`), e.g. `bug`, `gen2-hardening`,
  `scalability`, `documentation`, `testing`.

## 3. Look for duplicates and related work

Search open **and** closed issues and PRs with 2-3 keyword variants:
`gh search issues --repo NCAR/miles-credit "<keywords>"` and `gh pr list --state all --search "<keywords>"`.
Also check whether a PR already claims to fix it: on the issue, that is `closedByPullRequestsReferences`
(`gh api graphql -f query='query{repository(owner:"NCAR",name:"miles-credit"){issue(number:<n>){
closedByPullRequestsReferences(first:10,includeClosedPrs:true){nodes{number title state}}}}}'`).

## 4. Locate and reproduce

1. If the reporter shared a gen2 config, run `credit check -c <config>` on it first: many user issues are
   config errors, and the check names the problem directly.
2. Check whether it is already fixed: get the reporter's version (`pip show miles-credit`; a dev version
   such as `2026.2.1.dev126+gfb6fb1c0c` names the commit after the `g`) and run
   `git log <their-commit>..origin/main -- <files involved>`. If a later commit fixed it, say which.
3. Find the code involved and trace it to an entry point, as in the PR-review skill's Stage 1
   (`.agents/skills/credit-pr-review/SKILL.md`).
4. For a bug, try to reproduce it on current `main` in a clean worktree with the CPU environment from
   `.agents/skills/credit-pr-review/references/verification.md` §Setup. Write the smallest script that
   shows it. Result is one of:
   - **Reproduced** — include the script and its output.
   - **Not reproduced** — say what was tried; ask the reporter for the missing detail.
   - **Needs HPC/GPU** — say why it cannot run on CPU; point to the likely code path.
5. Give a root-cause hypothesis with `file:line`, labelled confirmed or suspected.

For a **feature request**, instead of reproducing: does it fit CREDIT's scope (AGENTS.md "About"), can it
be done today with a registry entry or `custom_objects` (`docs/source/Custom.md`), and does it overlap
open work? For a **question**, find the `docs/source/` page that answers it; if none does, that is a docs gap
worth noting.

## 5. Report

Show the user, in this order and briefly:

```
#<n> <title>
Type / area / gen:      ...
Duplicates / related:   #.. (why) — or none found
Reproduction:           Reproduced | Not reproduced | Needs HPC/GPU — one line, script path
Likely cause:           file:line — confirmed | suspected
Suggested labels:       ...
Priority (advisory):    P0 (data loss, wrong results, blocks training) | P1 | P2 — one-line reason
Size (advisory):        S | M | L — one-line reason
Next step:              who/what, e.g. "good first issue", "needs reporter's config", "fix with #PR"
Needs your call:        open disagreements or unanswered questions from the thread, if any
```

Priority and size are advice to the maintainer, not repo labels.

Then, if useful, a draft comment for the issue thread (under ~100 words: what was reproduced, what is
needed from the reporter, or the duplicate link). Post the comment or apply labels only after the user
approves, with `gh issue comment` / `gh issue edit --add-label`.
