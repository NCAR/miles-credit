"""CREDIT unified command-line interface.

Single entrypoint for training, rollout, job submission, and config generation.

Examples
--------
  credit init     --grid 0.25deg -o my_config.yml
  credit train    -c config.yml
  credit realtime -c config.yml --init-time 2024-01-15T00 --steps 40
  credit rollout  -c config.yml
  credit submit   --cluster derecho -c config.yml --gpus 4 --nodes 2
  credit submit   --cluster casper  -c config.yml --mode rollout --jobs 10
  credit submit   --cluster casper  -c config.yml --mode realtime --init-time 2024-01-15T00
"""

from ._ask import (
    _AGENT_SYSTEM_PROMPT,
    _AGENT_TOOL_DEFS,
    _CREDIT_SYSTEM_PROMPT,
    _ProviderError,
    _agent,
    _ask,
    _collect_run_context,
)
from ._common import (
    _AGENT_BASH_BLOCKLIST,
    _PBS_DEFAULTS,
    _agent_bash,
    _agent_list_files,
    _agent_read_file,
    _dispatch_tool,
    _find_torchrun,
    _is_ncar_system,
    _prompt,
    _prompt_bool,
    _repo_root,
    _setup_logging,
)
from ._convert import _convert, _init, _write_reload_config
from ._parser import _build_parser, main
from ._plot import _build_channel_map, _build_denorm_stats, _metrics, _plot
from ._submit import (
    _build_pbs_script,
    _build_realtime_pbs_script,
    _build_rollout_pbs_script,
    _compute_chain,
    _do_submit_realtime,
    _do_submit_rollout,
    _load_pbs_config,
    _print_ensemble_rollout_plan,
    _print_job_plan,
    _qsub,
    _realtime,
    _resolve_pbs_opts,
    _rollout,
    _rollout_ensemble,
    _submit,
    _train,
)

__all__ = [
    "_PBS_DEFAULTS",
    "_AGENT_BASH_BLOCKLIST",
    "_AGENT_SYSTEM_PROMPT",
    "_AGENT_TOOL_DEFS",
    "_CREDIT_SYSTEM_PROMPT",
    "_ProviderError",
    "_agent",
    "_agent_bash",
    "_agent_list_files",
    "_agent_read_file",
    "_ask",
    "_build_channel_map",
    "_build_denorm_stats",
    "_build_parser",
    "_build_pbs_script",
    "_build_realtime_pbs_script",
    "_build_rollout_pbs_script",
    "_collect_run_context",
    "_compute_chain",
    "_convert",
    "_dispatch_tool",
    "_do_submit_realtime",
    "_do_submit_rollout",
    "_find_torchrun",
    "_init",
    "_is_ncar_system",
    "_load_pbs_config",
    "_metrics",
    "_plot",
    "_print_ensemble_rollout_plan",
    "_print_job_plan",
    "_prompt",
    "_prompt_bool",
    "_qsub",
    "_realtime",
    "_repo_root",
    "_resolve_pbs_opts",
    "_rollout",
    "_rollout_ensemble",
    "_setup_logging",
    "_submit",
    "_train",
    "_write_reload_config",
    "main",
]
