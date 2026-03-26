# AI Assistant (`credit ask`)

`credit ask` is a unified AI assistant built into CREDIT.  It automatically runs in the
best mode available based on your API keys:

- **Agent mode** (when `ANTHROPIC_API_KEY` is set) — multi-turn agentic loop that reads
  your files, inspects logs, and runs shell commands before answering.
- **Simple chat** (fallback when Anthropic is unavailable) — one-shot Q&A using Groq,
  Gemini, OpenAI, or Anthropic Haiku, whichever key is set.

---

## Quick start

**NCAR users on Casper or Derecho** — shared Anthropic credits are available now.
No personal API key or billing required:

```bash
# Add to ~/.bashrc so it persists across sessions
module use /glade/work/bdobbins/llms/modules
module load llms

pip install "miles-credit[ask]"

credit ask "why did my last training run crash?"
credit ask -c config.yml "is my learning rate too high for 0.25 degree?"
```

**Everyone else:**

```bash
pip install "miles-credit[ask]"
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com → API keys

credit ask "why did my last training run crash?"
```

> **Note:** A Claude.ai Pro subscription does *not* include API access — billing is
> separate at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing).
> A typical session costs **$0.01–0.05**. See [Cost](#cost) for details.

---

## Two modes, one command

`credit ask` automatically picks the best mode for your setup:

### Agent mode (Anthropic)

When `ANTHROPIC_API_KEY` is set and the `anthropic` package is installed, `credit ask`
runs a **multi-turn agentic loop**:

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

It keeps going — reading more files, running more commands — until it has enough
information to give you a specific, actionable answer.

### Simple chat (fallback)

When Anthropic is unavailable, `credit ask` falls back to one-shot Q&A using whichever
provider key is set.  Provider priority (first found wins):

| Provider | Env var | Model | Cost |
|----------|---------|-------|------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Haiku | Pay-per-use (NCAR: shared) |
| OpenAI | `OPENAI_API_KEY` | GPT-4o | Pay-per-use |
| Google | `GOOGLE_API_KEY` | Gemini 1.5 Pro | Free for NCAR via AI Studio |
| Groq | `GROQ_API_KEY` | Llama 3 Instant | Free tier (no card needed) |

Simple chat injects your config, training log, and most recent PBS output as context
when you pass `-c`.

---

## Installation and setup

### 1. Install the package

```bash
pip install "miles-credit[ask]"
```

### 2. Get an API key

**NCAR users (Casper / Derecho):** shared Anthropic credits are available for all NCAR staff.
Add these two lines to your `~/.bashrc`:

```bash
module use /glade/work/bdobbins/llms/modules
module load llms
```

Then `source ~/.bashrc` (or log in again).  `credit ask` will pick up the key automatically.
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

Or use a free provider (no Anthropic key needed):

```bash
export GROQ_API_KEY=gsk_...    # https://console.groq.com  (free, no card)
```

---

## Usage reference

```
credit ask [-c CONFIG] [--max-turns N] [--provider PROVIDER] QUESTION
```

| Argument | Description |
|---|---|
| `QUESTION` | Your question or task in plain English |
| `-c CONFIG` | Path to your run's YAML config — agent gets your config, training log, and most recent PBS output as starting context |
| `--max-turns N` | Stop after N agentic turns (default: 20). Only applies in agent mode. |
| `--provider PROVIDER` | Force a specific provider for simple chat: `anthropic`, `openai`, `gemini`, `groq`. Anthropic agent is always tried first unless this forces a different provider. |

---

## Example sessions

### Diagnose a training crash

```bash
credit ask -c config/wxformer_1dg_6hr_v2.yml "why did my training run crash?"
```

The agent will:
1. Read your config to find `save_loc`
2. Glob for PBS output files (`*.o*`) and read the most recent one
3. Locate the traceback
4. If it's an OOM, read your config's batch size, model dimensions, and `amp` setting
5. Return a specific fix: e.g. "reduce `train_batch_size` from 8 to 4, or set `amp: True`"

### Check job queue and walltime

```bash
credit ask "how many of my jobs are queued on Derecho, and when does the running one expire?"
```

The agent runs `qstat -u $USER`, parses the output, and tells you exactly what's running,
how much walltime remains, and whether anything is stuck in queue.

### Config review before a long run

```bash
credit ask -c config/big_run.yml \
  "I'm about to start a 200-epoch run on 8 H100s. Review my config for anything that would waste compute or cause it to fail."
```

The agent reads your full config and cross-references it with the source code to flag issues —
wrong `num_epoch` vs `epochs` ratio, missing `save_best_weights`, `use_scheduler: False` with
a large run, etc.

### Understand source code

```bash
credit ask "walk me through how ConcatPreblock assembles the batch tensor — what goes into x and what goes into y?"
```

The agent reads `credit/preblock/concat.py` and the relevant trainer code and gives you a
plain-English explanation with line references.

### Compare two configs

```bash
credit ask "compare config/run_a.yml and config/run_b.yml and explain every difference"
```

The agent reads both files and produces a structured diff with explanations of what each
difference means for training behaviour.

### Debug a data loading hang

```bash
credit ask -c config.yml "my training job starts but then hangs and never prints a loss — what's wrong?"
```

The agent checks your `thread_workers`, `prefetch_factor`, and dataset size, estimates
DataLoader memory usage, and flags if you're likely hitting an OOM or deadlock.

---

## What the agent can access

In agent mode the assistant has three tools. All are **read-only** — it cannot modify,
delete, or move files, and cannot submit or cancel jobs.

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
than "how does the data pipeline work?" because the agent can immediately read the right module.

**Use `--max-turns` for very complex tasks.**  The default of 20 is enough for most
debugging sessions.  For a full config audit or multi-file code review, `--max-turns 40`
gives the agent more room.

---

## Cost

Agent mode uses `claude-sonnet-4-6` — Anthropic's mid-tier model that balances
capability with cost.

| Session type | Typical turns | Approximate cost |
|---|---|---|
| Simple Q&A (no files needed) | 1–2 | < $0.005 |
| Diagnose a crash (read log + config) | 3–6 | $0.01–0.02 |
| Full config review | 5–10 | $0.02–0.05 |
| Multi-file code investigation | 10–20 | $0.05–0.15 |

Simple chat (Haiku/Groq/Gemini) costs significantly less or nothing.

Pricing is based on Anthropic's published input/output token rates.
A full PBS output log is typically 5,000–20,000 tokens; a config is 1,000–3,000 tokens.

---

## Troubleshooting

**`Anthropic API key has no credits`**
Add credits at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing).
API credits are separate from a Claude.ai subscription.
`credit ask` will automatically fall back to simple chat providers if Anthropic credits run out.

**`No API key found`**
```bash
# NCAR users:
module use /glade/work/bdobbins/llms/modules
module load llms

# Everyone else:
export ANTHROPIC_API_KEY=sk-ant-...
# or free alternatives:
export GROQ_API_KEY=gsk_...
```

**`anthropic package required`**
```bash
pip install "miles-credit[ask]"
```

**Agent gives a generic answer and doesn't read files**
Make sure you're passing `-c config.yml` so it has a starting point.  You can also be explicit:
```bash
credit ask "read the most recent *.o* file in this directory and tell me if there are any errors"
```

**Agent hits `max_turns` without finishing**
Increase the limit:
```bash
credit ask --max-turns 40 -c config.yml "do a full audit of my training setup"
```
