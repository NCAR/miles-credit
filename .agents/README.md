# Agent skills for CREDIT

Reusable instructions for AI coding agents working on this repository. The repo-wide context every
agent should read is [`AGENTS.md`](../AGENTS.md) at the repo root; the skills here build on it.

| Skill | Use it to |
|---|---|
| [`credit-pr-review`](skills/credit-pr-review/SKILL.md) | Review a PR and verify every finding by running code before reporting it |
| [`credit-issue-triage`](skills/credit-issue-triage/SKILL.md) | Classify an issue, find duplicates, try to reproduce it and suggest labels and priority |

Each skill is a folder with a `SKILL.md` (name, description, steps) and optional `references/` files that
the steps link to.

## Using them

- **Claude Code**: `.claude/skills/` holds symlinks to these folders, so the skills are available as
  `/credit-pr-review <PR number>` and `/credit-issue-triage <issue number>`. Claude Code reads
  `AGENTS.md` itself from version 2.1.281; on older versions run `claude update`.
- **Other agents**: agents that read `.agents/skills/` pick them up directly. For any other agent,
  point it at the skill's `SKILL.md`, e.g. "Follow `.agents/skills/credit-pr-review/SKILL.md` for PR 564".

Windows: the `.claude/skills/` symlinks need `git config core.symlinks true` (and Developer Mode or admin
rights) before checkout; otherwise they appear as small text files.

## Ground rules shared by the skills

- Never post to GitHub, label, or push without the user's approval of the exact text.
- Work in git worktrees and a scratch environment; never switch branches in the user's checkout.
- Report a finding as fact only after reproducing it; otherwise say it was not reproduced and why.

## Adding a skill

Create `.agents/skills/<name>/SKILL.md` with `name` and `description` front matter, add a symlink
`ln -s ../../.agents/skills/<name> .claude/skills/<name>`, and add a row to the table above.
