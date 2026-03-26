# AI Agent (`credit agent`)

`credit agent` is an agentic AI assistant built into CREDIT.  Unlike a plain chatbot, it
has **eyes** — it reads your actual files, inspects your logs, and runs shell commands before
answering.  You describe a problem in plain English; the agent figures out what to look at and
comes back with a concrete, specific answer grounded in your real data.

---

## Quick start

**NCAR users on Casper or Derecho** — shared Anthropic credits are available now.
No personal API key or billing required:

```bash
# Add to ~/.bashrc so it persists across sessions
module use /glade/work/bdobbins/llms/modules
module load llms

pip install "miles-credit[agent]"

credit agent "why did my last training run crash?"
credit agent -c config.yml "is my learning rate too high for 0.25 degree?"
```

**Everyone else:**

```bash
pip install "miles-credit[agent]"
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com → API keys

credit agent "why did my last training run crash?"
```

> **Note:** A Claude.ai Pro subscription does *not* include API access — billing is
> separate at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing).
> A typical session costs **$0.01–0.05**. See [Cost](#cost) for details.

---

## What it does that `credit ask` can't

`credit ask` is a single-shot Q&A — you paste in a question, it answers from its training
knowledge plus whatever context you inject with `-c`.  It's fast and cheap and works with
four free/low-cost providers (Groq, Gemini, OpenAI, Anthropic).

`credit agent` runs a **multi-turn agentic loop**:

```
Your question
    ↓
Agent decides what to look at
    ↓
Reads PBS log  →  finds CUDA OOM traceback
    ↓
Reads config   →  sees train_batch_size: 8
    ↓
Reads source   →  confirms memory layout for 0.25° × 18 levels
    ↓
Answer: "Reduce batch size to 4 or enable amp: True …"
```

It keeps going — reading more files, running more commands — until it has enough information
to give you a specific, actionable answer.

| | `credit ask` | `credit agent` |
|---|---|---|
| **Providers** | Anthropic, OpenAI, Gemini, Groq | Anthropic only |
| **File access** | No | Yes — reads any file you have access to |
| **Shell access** | No | Yes — safe read-only commands |
| **Multi-turn** | No | Yes (default: 20 turns) |
| **Best for** | Quick questions, free usage | Diagnosing crashes, deep config review |
| **Model** | Claude Haiku (fast, cheap) | Claude Sonnet (more capable) |

---

## Installation and setup

### 1. Install the package

```bash
pip install "miles-credit[agent]"
```

If you already installed CREDIT without extras, this adds only `anthropic` — nothing else changes.

### 2. Get an API key

**NCAR users (Casper / Derecho):** shared Anthropic credits are available for all NCAR staff.
Add these two lines to your `~/.bashrc`:

```bash
module use /glade/work/bdobbins/llms/modules
module load llms
```

Then `source ~/.bashrc` (or log in again).  `credit agent` will pick up the key automatically.
Contact [milescore@ucar.edu](mailto:milescore@ucar.edu) if you expect heavy usage.

**Everyone else:**

1. Sign up or log in at [console.anthropic.com](https://console.anthropic.com)
2. Go to **API Keys** → **Create Key**
3. Add credits at **Settings → Billing** (pay-as-you-go, no subscription required)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Persist across sessions
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
```

---

## Usage reference

```
credit agent [-c CONFIG] [--max-turns N] QUESTION
```

| Argument | Description |
|---|---|
| `QUESTION` | Your question or task in plain English |
| `-c CONFIG` | Path to your run's YAML config — the agent gets your config, training log, and most recent PBS output as starting context |
| `--max-turns N` | Stop after N agentic turns (default: 20). Increase for very complex investigations. |

---

## Example sessions

### Diagnose a training crash

```bash
credit agent -c config/wxformer_1dg_6hr_v2.yml "why did my training run crash?"
```

The agent will:
1. Read your config to find `save_loc`
2. Glob for PBS output files (`*.o*`) and read the most recent one
3. Locate the traceback
4. If it's an OOM, read your config's batch size, model dimensions, and `amp` setting
5. Return a specific fix: e.g. "reduce `train_batch_size` from 8 to 4, or set `amp: True`"

### Check job queue and walltime

```bash
credit agent "how many of my jobs are queued on Derecho, and when does the running one expire?"
```

The agent runs `qstat -u $USER`, parses the output, and tells you exactly what's running,
how much walltime remains, and whether anything is stuck in queue.

### Config review before a long run

```bash
credit agent -c config/big_run.yml \
  "I'm about to start a 200-epoch run on 8 H100s. Review my config for anything that would waste compute or cause it to fail."
```

The agent reads your full config and cross-references it with the source code to flag issues —
wrong `num_epoch` vs `epochs` ratio, missing `save_best_weights`, `use_scheduler: False` with
a large run, etc.

### Understand source code

```bash
credit agent "walk me through how ConcatPreblock assembles the batch tensor — what goes into x and what goes into y?"
```

The agent reads `credit/preblock/concat.py` and the relevant trainer code and gives you a
plain-English explanation with line references.

### Compare two configs

```bash
credit agent "compare config/run_a.yml and config/run_b.yml and explain every difference"
```

The agent reads both files and produces a structured diff with explanations of what each
difference means for training behaviour.

### Debug a data loading hang

```bash
credit agent -c config.yml "my training job starts but then hangs and never prints a loss — what's wrong?"
```

The agent checks your `thread_workers`, `prefetch_factor`, and dataset size, estimates
DataLoader memory usage, and flags if you're likely hitting an OOM or deadlock.

---

## What the agent can access

The agent has three tools. All are **read-only** — it cannot modify, delete, or move files,
and cannot submit or cancel jobs.

### `read_file`

Reads any file you have filesystem access to.  Returns up to 400 lines from the end of the
file by default (configurable via the `tail` parameter the agent chooses internally).

Best used for: configs, PBS output logs, Python tracebacks, source files, checkpoint metadata.

### `list_files`

Glob-style file discovery.  The agent uses this to find your PBS logs, locate configs, or
discover checkpoint directories.

Examples it might run internally:
```
*.o*          → find PBS output files in current directory
logs/**/*.txt → find all log files recursively
save_loc/**   → find checkpoints for your run
```

### `bash`

Runs read-only shell commands with a 30-second timeout.  Permitted commands include:

| Command | What it's used for |
|---|---|
| `qstat` / `squeue` | Check job queue status |
| `grep` | Search files for patterns |
| `tail` / `head` | Read the end/start of large files |
| `find` | Locate files by name or modification time |
| `git log` / `git diff` | Inspect repo history |
| `ls` / `wc` | List files, count lines |
| `diff` | Compare two files |

The following are **blocked** regardless of how they're phrased:
`rm`, `mv`, `cp`, `git push`, `git reset`, `git checkout`, `qdel`, `scancel`, `kill`,
`pip install`, `conda install`, `sudo`, and any output-redirect operators (`>`, `>>`).

---

## Tips for best results

**Give it your config with `-c`.** Without it the agent has to search for context; with it
the agent starts with your full run setup and gets to the answer faster.

**Be specific about what went wrong.**  "it crashed" forces the agent to explore; "it crashed
with CUDA OOM at epoch 3" lets it skip the discovery phase and go straight to solutions.

**For source code questions, name the thing.**  "how does ConcatPreblock work?" is better
than "how does the data pipeline work?" because the agent can immediately `read_file` the
right module.

**Use `--max-turns` for very complex tasks.**  The default of 20 is enough for most
debugging sessions.  For a full config audit or multi-file code review, `--max-turns 40`
gives the agent more room.

---

## Cost

`credit agent` uses `claude-sonnet-4-6` — Anthropic's mid-tier model that balances
capability with cost.

| Session type | Typical turns | Approximate cost |
|---|---|---|
| Simple Q&A (no files needed) | 1–2 | < $0.005 |
| Diagnose a crash (read log + config) | 3–6 | $0.01–0.02 |
| Full config review | 5–10 | $0.02–0.05 |
| Multi-file code investigation | 10–20 | $0.05–0.15 |

Pricing is based on Anthropic's published input/output token rates.
A full PBS output log is typically 5,000–20,000 tokens; a config is 1,000–3,000 tokens.

---

## Troubleshooting

**`Anthropic API key has no credits`**
Add credits at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing).
API credits are separate from a Claude.ai subscription.

**`ANTHROPIC_API_KEY is not set`**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**`anthropic package required`**
```bash
pip install "miles-credit[agent]"
```

**Agent gives a generic answer and doesn't read files**
Make sure you're passing `-c config.yml` so it has a starting point.  You can also be explicit:
```bash
credit agent "read the most recent *.o* file in this directory and tell me if there are any errors"
```

**Agent hits `max_turns` without finishing**
Increase the limit:
```bash
credit agent --max-turns 40 -c config.yml "do a full audit of my training setup"
```
