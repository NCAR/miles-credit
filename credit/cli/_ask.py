"""credit ask and credit agent command handlers."""

import logging
import os
import sys

from ._common import (
    _dispatch_tool,
    _is_ncar_system,
)

logger = logging.getLogger(__name__)

_CREDIT_SYSTEM_PROMPT = """\
You are CREDIT-Ask, an AI assistant for the CREDIT software package (Community Research Earth Digital Intelligence Twin),
an AI-based numerical weather prediction framework developed by the NCAR MILES group.
When introducing yourself, use the name "CREDIT-Ask". Do not call yourself "CREDIT" — that is the name of the software package you support.

## What CREDIT is
CREDIT trains deep learning models (primarily WXFormer) to forecast global atmospheric state.
It runs on NCAR HPC clusters: Casper (single-node, A100/H100 GPUs) and Derecho (multi-node, A100 GPUs).
The main entry point is the `credit` CLI.

## Key CLI commands
- `credit train -c config.yml`                    — start/resume training
- `credit submit --cluster casper|derecho -c config.yml [--mode train|rollout|realtime] --gpus N [--nodes N] [--chain N] [--reload] [--jobs N] [--init-time YYYY-MM-DDTHH] [--steps N]`
- `credit plot -c config.yml --field VAR_2T --denorm`   — quick visualisation from checkpoint
- `credit rollout -c config.yml`                  — batch forecast to NetCDF
- `credit realtime -c config.yml --init-time YYYY-MM-DDTHH --steps N`
- `credit init --grid 1deg|0.25deg -o my_config.yml`    — generate starter config
- `credit ask "..."`                              — this command

## v2 data schema (YAML)
```yaml
data:
  source:
    ERA5:
      levels: [...]          # pressure/model levels
      variables:
        prognostic:
          vars_3D: [U, V, T, Q, Z]    # each × n_levels channels
          vars_2D: [SP, VAR_2T, ...]
        diagnostic:
          vars_2D: [precip, evap, ...]
  mean_path: /path/to/mean.nc
  std_path:  /path/to/std.nc
```

## Trainer config
```yaml
trainer:
  type: era5-gen2
  mode: ddp           # none | ddp | fsdp
  train_batch_size: 8
  num_epoch: 5        # epochs per PBS job
  epochs: 70          # total training target
  thread_workers: 4   # DataLoader workers per GPU
  prefetch_factor: 4
  use_tensorboard: True
  use_ema: True
  ema_decay: 0.9999
  use_scheduler: True
  scheduler:
    scheduler_type: linear-warmup-cosine
    warmup_steps: 1000
    total_steps: 500000
    min_lr: 1.0e-5
  dataloader_timeout_s: 300   # preflight hang detection
```

## Cluster specifics
- **Casper**: 1 node, torchrun --standalone, GPUs: V100/A100/H100, queue: casper
  - Pre-built env: `/glade/u/home/schreck/.conda/envs/credit-casper`
- **Derecho single-node**: torchrun --standalone (NOT mpiexec)
- **Derecho multi-node**: mpiexec + torchrun --rdzv-backend=c10d
  - Pre-built env: `/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221`
- Data root: `/glade/campaign/cisl/aiml/ksha/CREDIT_data/`
- Default save_loc: `/glade/derecho/scratch/$USER/CREDIT_runs/`

## Common problems and fixes
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Training loop hangs on startup | DataLoader OOM (too many workers × prefetch × batch × channels) | Reduce `thread_workers` to 1 or 0, or `prefetch_factor` to 1 |
| `RendezvousConnectionError` on Derecho | Single-node job using c10d rendezvous | Use `--nodes 1` so `credit submit` generates `--standalone` |
| Loss > 100 or growing | Bad normalization or wrong data paths | Check `mean_path`/`std_path`; run `credit plot --denorm` |
| Loss stuck (not decreasing) | LR too low/high, wrong scheduler, EMA misconfigured | Check scheduler config; try reducing LR 10×; check warmup_steps |
| `KeyError: 'linear-warmup-cosine'` | Old CREDIT version | `pip install -e . --no-deps` to reinstall |
| Checkpoint not found | Wrong `save_loc` or first epoch | Set `load_weights: False` for first run |
| PBS job cancelled after failure | Normal: `afterok` chain auto-cancels remaining jobs | Use `credit submit --reload --chain N` to restart |
| FSDP + EMA slow | EMA does extra full-param sync on FSDP | Use `use_ema: False` with FSDP or accept overhead |

## How --chain works (train mode)
`--chain N` submits N PBS jobs with afterok dependencies. Job 1 runs fresh (or --reload).
Jobs 2..N auto-generate `config_reload.yml` and resume from checkpoint.
Rule of thumb: chain = ceil(total_epochs / num_epoch). E.g., 70 epochs / 5 per job = 14.

## submit --mode options
- `--mode train` (default): training job, supports --chain and --reload
- `--mode rollout`: N parallel jobs covering all init times, use --jobs N; reads predict: section
- `--mode realtime`: single forecast job, requires --init-time YYYY-MM-DDTHH and --steps N

## What healthy training looks like
- After epoch 1: train_loss ≈ 1–3 (order 1)
- Loss should decrease steadily each epoch
- Validation loss should track training loss (not diverge)
- `credit plot -c config.yml --field VAR_2T --denorm` should show recognisable weather patterns after ~10 epochs

Be concise, specific, and actionable. When referencing config keys use inline code. If you see a training log or config in the context, use it to give run-specific advice.
"""

_AGENT_SYSTEM_PROMPT = """\
You are CREDIT-Agent, an agentic AI assistant for the CREDIT software package (Community Research Earth Digital Intelligence Twin),
an AI-based numerical weather prediction framework developed by the NCAR MILES group.
When introducing yourself, use the name "CREDIT-Agent". Do not call yourself "CREDIT" — that is the name of the software package you support.

You have access to tools that let you read files, list files, and run safe read-only shell commands.
Use them to investigate the user's question thoroughly before answering.

Typical tasks:
- Diagnose why a training run crashed (read PBS logs, config, Python tracebacks)
- Explain what a config option does (read the relevant source file)
- Suggest config changes based on the user's hardware and dataset
- Check whether a job is still running (qstat) and interpret its output
- Diff configs between two experiments

Guidelines:
- Always read relevant files before speculating — the answer is usually in the logs or config.
- When reading PBS output files (*.o*), focus on the last 100 lines first.
- Suggest concrete, actionable fixes — not generic advice.
- Keep responses concise; use markdown headers and code blocks.
"""

_AGENT_TOOL_DEFS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Useful for configs (*.yml), PBS output logs (*.o*), "
            "Python tracebacks, and source code. Returns at most 400 lines from the end of the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (absolute or relative to cwd)"},
                "tail": {
                    "type": "integer",
                    "description": "Read only the last N lines (default 400). Pass 0 for the full file.",
                    "default": 400,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Useful for finding configs, logs, checkpoints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to cwd, e.g. '*.yml', 'logs/*.txt', '**/*.py'",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a read-only shell command and return stdout+stderr. "
            "Allowed commands: grep, find, tail, head, cat, ls, wc, diff, git log, git diff, git status, "
            "qstat, squeue, python -c (imports only). Destructive commands are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
]


def _collect_run_context(args) -> str:
    """Gather config, training log, and recent PBS output for context injection."""
    import glob as _glob
    import yaml

    parts = []

    if getattr(args, "config", None):
        try:
            with open(args.config) as f:
                raw = f.read()
            parts.append(f"## Active config ({args.config})\n```yaml\n{raw}\n```")
        except OSError:
            pass

        try:
            import pandas as pd

            with open(args.config) as f:
                conf = yaml.safe_load(f)
            save_loc = conf.get("save_loc", "")
            log_path = os.path.join(save_loc, "training_log.csv")
            if os.path.exists(log_path):
                df = pd.read_csv(log_path)
                tail = df.tail(20).to_string(index=False)
                parts.append(f"## training_log.csv (last {min(20, len(df))} rows)\n```\n{tail}\n```")
        except Exception:
            pass

        try:
            pbs_files = _glob.glob("credit_gen2.o*") + _glob.glob("credit.o*")
            if pbs_files:
                newest = max(pbs_files, key=os.path.getmtime)
                with open(newest) as f:
                    lines = f.readlines()
                tail_lines = "".join(lines[-60:])
                parts.append(f"## Most recent PBS output ({newest}, last 60 lines)\n```\n{tail_lines}\n```")
        except Exception:
            pass

    return "\n\n".join(parts)


class _ProviderError(Exception):
    """Raised when a provider call fails in a way that should trigger fallback."""


def _ask_anthropic(user_msg: str) -> None:
    import anthropic

    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_CREDIT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
    except anthropic.BadRequestError as e:
        if "credit balance is too low" in str(e):
            raise _ProviderError(
                "Anthropic API key has no credits.\n"
                "  → Add credits: https://console.anthropic.com/settings/billing\n"
                "  → Note: Claude.ai Pro does NOT include API access."
            ) from e
        raise


def _ask_groq(user_msg: str) -> None:
    import groq

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _CREDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def _ask_openai(user_msg: str) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stream = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _CREDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def _ask_gemini(user_msg: str) -> None:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=_CREDIT_SYSTEM_PROMPT,
    )
    for chunk in model.generate_content(user_msg, stream=True):
        print(chunk.text, end="", flush=True)


_OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _ask_openrouter(user_msg: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    stream = client.chat.completions.create(
        model=_OPENROUTER_MODEL,
        max_tokens=8192,
        extra_body={"enable_thinking": True},
        messages=[
            {"role": "system", "content": _CREDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    in_thinking = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_thinking:
                print(f"{_DIM}<thinking>", flush=True)
                in_thinking = True
            print(f"{reasoning}", end="", flush=True)
        if delta.content:
            if in_thinking:
                print(f"</thinking>{_RESET}\n", flush=True)
                in_thinking = False
            print(delta.content, end="", flush=True)
    if in_thinking:
        print(f"</thinking>{_RESET}", flush=True)


_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic", "Claude Haiku"),
    "openai": ("OPENAI_API_KEY", "openai", "GPT-4o"),
    "gemini": ("GOOGLE_API_KEY", "google.generativeai", "Gemini 1.5 Pro"),
    "groq": ("GROQ_API_KEY", "groq", "Llama 3 Instant (free)"),
    "openrouter": ("OPENROUTER_API_KEY", "openai", "Qwen3-Next-80B thinking (OpenRouter free)"),
}
_PROVIDER_INSTALL = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google-generativeai",
    "groq": "groq",
    "openrouter": "openai",
}
_PROVIDER_RUNNERS = {
    "anthropic": _ask_anthropic,
    "openai": _ask_openai,
    "gemini": _ask_gemini,
    "groq": _ask_groq,
    "openrouter": _ask_openrouter,
}


def _ask(args) -> None:
    """Unified AI assistant: tries agentic mode first, falls back to simple chat."""
    question = " ".join(args.question)
    context = _collect_run_context(args)
    explicit = getattr(args, "provider", None)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    _try_agent = (explicit is None or explicit == "anthropic") and bool(api_key)
    _skip_anthropic = False

    if _try_agent:
        _ant = None
        try:
            import anthropic as _ant_mod

            _ant = _ant_mod
        except ImportError:
            pass

        if _ant is not None:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            user_msg = f"{context}\n\n## Task\n{question}" if context else question
            client = _ant.Anthropic(api_key=api_key)
            messages = [{"role": "user", "content": user_msg}]
            max_turns = getattr(args, "max_turns", 20)
            print()

            for _turn in range(max_turns):
                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=4096,
                        system=_AGENT_SYSTEM_PROMPT,
                        tools=_AGENT_TOOL_DEFS,
                        messages=messages,
                    )
                except _ant.BadRequestError as exc:
                    if "credit balance is too low" in str(exc):
                        print("Anthropic API key has no credits — falling back to simple chat.\n", file=sys.stderr)
                        _skip_anthropic = True
                        break
                    print(f"API error: {exc}", file=sys.stderr)
                    sys.exit(1)

                for block in response.content:
                    if block.type == "text" and block.text:
                        print(block.text, end="", flush=True)

                tool_calls = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason == "end_turn" or not tool_calls:
                    print("\n")
                    return

                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tc in tool_calls:
                    if not any(b.type == "text" for b in response.content):
                        print(f"\n[{tc.name}: {tc.input}]", flush=True)
                    result = _dispatch_tool(tc.name, tc.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
            else:
                print(f"\n[reached max_turns={max_turns}]", file=sys.stderr)
                print("\n")
                return

    # Simple chat fallback
    user_msg = f"{context}\n\n## Question\n{question}" if context else question

    if explicit:
        if explicit not in _PROVIDERS:
            print(f"Unknown provider {explicit!r}. Choose: {', '.join(_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)
        env_key, _pkg, _label = _PROVIDERS[explicit]
        if not os.environ.get(env_key):
            print(f"{env_key} is not set.", file=sys.stderr)
            sys.exit(1)
        ordered = [explicit]
    else:
        ordered = [p for p in _PROVIDERS if os.environ.get(_PROVIDERS[p][0])]
        if _skip_anthropic and "anthropic" in ordered:
            ordered.remove("anthropic")

    if not ordered:
        if _skip_anthropic:
            print(
                "\nNo working provider found.\n"
                "Anthropic credits are exhausted. Set a fallback key:\n"
                "  export GROQ_API_KEY=gsk_...             # https://console.groq.com  (free)\n"
                "  export OPENROUTER_API_KEY=sk-or-...    # https://openrouter.ai     (free tier)\n"
                "  export GOOGLE_API_KEY=AIza...          # https://aistudio.google.com\n"
                "  export OPENAI_API_KEY=sk-...           # https://platform.openai.com",
                file=sys.stderr,
            )
        else:
            print(
                "No API key found.\n\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com\n"
                "  export OPENAI_API_KEY=sk-...           # https://platform.openai.com\n"
                "  export GOOGLE_API_KEY=AIza...          # https://aistudio.google.com  (free for NCAR)\n"
                "  export GROQ_API_KEY=gsk_...            # https://console.groq.com     (free tier)\n"
                "  export OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai         (free tier)\n\n"
                "See: https://miles-credit.readthedocs.io/en/latest/quickstart.html"
                "#get-help-from-the-ai-assistant",
                file=sys.stderr,
            )
        sys.exit(1)

    print()
    for attempt, p in enumerate(ordered):
        _, pkg, label = _PROVIDERS[p]
        try:
            __import__(pkg)
        except ImportError:
            print(
                f"Skipping {label}: package not installed.  Fix: pip install {_PROVIDER_INSTALL[p]}",
                file=sys.stderr,
            )
            continue
        if p != "anthropic" or attempt > 0:
            print(f"(using {label})\n")
        try:
            _PROVIDER_RUNNERS[p](user_msg)
            print("\n")
            return
        except _ProviderError as exc:
            print(f"\n{exc}", file=sys.stderr)
            remaining = [x for x in ordered[attempt + 1 :] if os.environ.get(_PROVIDERS[x][0])]
            if remaining:
                print("Trying next provider…\n", file=sys.stderr)

    print(
        "\nNo working provider found. Set one of:\n"
        "  export GROQ_API_KEY=gsk_...            # https://console.groq.com  (free)\n"
        "  export OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai     (free tier)\n"
        "  export GOOGLE_API_KEY=AIza...          # https://aistudio.google.com  (free for NCAR)\n"
        "  export OPENAI_API_KEY=sk-...           # https://platform.openai.com\n"
        "  export ANTHROPIC_API_KEY=sk-ant-       # https://console.anthropic.com  (requires credits)",
        file=sys.stderr,
    )
    sys.exit(1)


def _agent(args) -> None:
    """Run an agentic session: Claude reads files and runs commands to answer your question."""
    try:
        import anthropic
    except ImportError:
        print("anthropic package required: pip install 'miles-credit[agent]'", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        msg = "ANTHROPIC_API_KEY is not set.\n"
        if not _is_ncar_system():
            msg += (
                "\ncredit agent requires an Anthropic API key with active credits.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  → https://console.anthropic.com"
            )
        print(msg, file=sys.stderr)
        sys.exit(1)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    question = " ".join(args.question)
    context = _collect_run_context(args)
    user_msg = f"{context}\n\n## Task\n{question}" if context else question

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_msg}]
    max_turns = getattr(args, "max_turns", 20)

    print()
    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_AGENT_SYSTEM_PROMPT,
                tools=_AGENT_TOOL_DEFS,
                messages=messages,
            )
        except anthropic.BadRequestError as exc:
            if "credit balance is too low" in str(exc):
                print(
                    "Anthropic API key has no credits.\n"
                    "  → Add credits: https://console.anthropic.com/settings/billing\n"
                    "  → Note: Claude.ai Pro does NOT include API access.",
                    file=sys.stderr,
                )
            else:
                print(f"API error: {exc}", file=sys.stderr)
            sys.exit(1)

        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text, end="", flush=True)

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_calls:
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            if not any(b.type == "text" for b in response.content):
                print(f"\n[{tc.name}: {tc.input}]", flush=True)
            result = _dispatch_tool(tc.name, tc.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"\n[reached max_turns={max_turns}]", file=sys.stderr)

    print("\n")
