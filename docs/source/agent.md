# AI Agent (`credit agent`)

`credit agent` is an agentic AI assistant that can **read your files, inspect logs, and run
read-only shell commands** to diagnose problems and answer questions about your CREDIT runs.

Unlike [`credit ask`](quickstart.md#get-help-from-the-ai-assistant) — which is a single-shot
Q&A — `credit agent` runs a multi-turn loop.  It looks at your actual config, training logs,
and PBS output, iterates until it has enough information, then gives you a concrete answer.

---

## Requirements

`credit agent` requires an Anthropic API key **with active API credits**.

> **Note:** A Claude.ai Pro subscription does **not** include API access.
> API billing is separate at [console.anthropic.com](https://console.anthropic.com/settings/billing).

```bash
# Install the agent extra
pip install "miles-credit[agent]"

# Set your API key (add to ~/.bashrc to persist)
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

```
credit agent [-c CONFIG] [--max-turns N] QUESTION
```

| Flag | Description |
|---|---|
| `QUESTION` | Your question or task (quote it) |
| `-c CONFIG` | Path to a YAML config — injects config, training log, and most recent PBS output as context |
| `--max-turns N` | Maximum agentic turns before stopping (default: 20) |

---

## Examples

**Diagnose a crashed training run:**
```bash
credit agent -c config/wxformer_1dg_6hr_v2.yml "why did my training run crash?"
```
The agent will read your PBS output log, find the traceback, read relevant source files,
and explain what went wrong with a suggested fix.

**Check running jobs:**
```bash
credit agent "what PBS jobs are currently running and how long have they been running?"
```

**Config review:**
```bash
credit agent -c config.yml "is my learning rate schedule appropriate for 0.25 degree training on 16 H100s?"
```

**Compare configs:**
```bash
credit agent "diff config/my_run.yml against config/starter_v2.yml and explain the differences"
```

**Understand source code:**
```bash
credit agent "how does the ConcatPreblock assemble the batch tensor?"
```

---

## What the Agent Can Do

The agent has access to three tools scoped to your environment:

| Tool | Description |
|---|---|
| `read_file` | Read any file — configs, PBS logs, Python source, checkpoints metadata |
| `list_files` | Glob for files by pattern (`*.yml`, `logs/*.o*`, etc.) |
| `bash` | Run safe read-only commands: `grep`, `tail`, `find`, `qstat`, `git log`, `git diff` |

Destructive commands (`rm`, `mv`, `git push`, `qdel`, etc.) are blocked.

---

## How It Works

```
You: "why did my training run crash?"
         ↓
Agent reads PBS output log (tail 400 lines)
         ↓
Finds: CUDA OOM traceback in trainerERA5v2.py:312
         ↓
Agent reads config.yml → sees train_batch_size: 8
         ↓
Agent reads credit/trainers/trainerERA5v2.py:312 for context
         ↓
Answer: "Your batch size of 8 on a 40 GB A100 is too large for 0.25°
         with 18 levels × 4 variables.  Try train_batch_size: 4, or
         enable amp: True to reduce memory usage."
```

---

## vs. `credit ask`

| | `credit ask` | `credit agent` |
|---|---|---|
| **Providers** | Anthropic, OpenAI, Gemini, Groq | Anthropic only |
| **File access** | No (context injected once) | Yes (reads files during the loop) |
| **Multi-turn** | No | Yes (up to `--max-turns`) |
| **Best for** | Quick questions, free-tier usage | Deep diagnosis, code understanding |
| **Cost** | Low (Haiku) | Higher (Sonnet — more capable) |

Use `credit ask` for quick questions and when you don't have Anthropic credits.
Use `credit agent` when you need it to actually look at your data.

---

## Cost

`credit agent` uses `claude-sonnet-4-6`.  A typical debugging session (5–10 turns,
reading a few files) costs roughly **$0.01–0.05** depending on file sizes.
