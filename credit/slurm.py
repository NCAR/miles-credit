"""SLURM analog of :mod:`credit.pbs`.

Generates and (optionally) submits SLURM batch scripts for CREDIT training and
rollout jobs.  The public API mirrors :mod:`credit.pbs` so callers can swap
``from credit.pbs import ...`` for ``from credit.slurm import ...`` with no other
changes.

All launchers use ``torchrun`` rather than ``mpirun``/``mpiexec``: on SLURM the
``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` variables that
:func:`credit.distributed.get_rank_info` reads are set by torchrun, whereas
``srun`` alone only exports ``SLURM_PROCID`` (which is not recognized).  Single
node jobs run torchrun directly on the batch node; multi-node jobs use ``srun``
to launch one torchrun per node with a c10d rendezvous on the head node.

Config is read from a ``slurm:`` section if present, otherwise the ``pbs:``
section is reused (SLURM keys map as ``project``/``account`` -> --account,
``queue``/``partition`` -> --partition, ``ngpus`` -> GPUs per node, ``ncpus`` ->
--cpus-per-task, ``mem`` -> --mem, ``walltime`` -> --time).  Optional keys:
``gpu_type`` (adds ``--gres=gpu:<type>:<n>``), ``constraint`` (``--constraint``),
``qos`` (``--qos``), ``modules`` (str or list, passed to ``module load``), and
``env_setup`` (str or list of extra shell lines).

GPU request style differs by site: generic SLURM clusters request GPUs with
``--gres=gpu:N``, but Perlmutter (NERSC) rejects that ("Job request does not
match any supported policy") and instead selects GPU nodes via
``--constraint=gpu`` + ``--qos`` + ``--gpus-per-node``, needs no ``--partition``
or ``--mem`` line, and requires a ``_g`` account suffix.  Perlmutter is detected
from ``NERSC_HOST``, a ``constraint`` config key, or ``cluster: perlmutter``, and
the correct directives are emitted automatically.
"""

import re
import os
import yaml
import shutil
import logging
import subprocess

from credit.cli._common import _PERLMUTTER_ENV_SETUP

logger = logging.getLogger(__name__)

# ``module load`` argument for Perlmutter's NCCL build (paired with the explicit
# env vars in ``_PERLMUTTER_ENV_SETUP``).
_PERLMUTTER_MODULES = "nccl/2.24.3"


def _slurm_options(config):
    """Return the SLURM options dict, falling back to the PBS section."""
    return config.get("slurm") or config.get("pbs") or {}


def _gres(num_gpus, gpu_type=None):
    """Return a ``--gres=gpu:...`` value, optionally pinned to a GPU type."""
    if gpu_type:
        return f"gpu:{gpu_type}:{num_gpus}"
    return f"gpu:{num_gpus}"


def _is_perlmutter(options):
    """Return True if the job targets Perlmutter (NERSC) GPU nodes."""
    return (
        os.environ.get("NERSC_HOST") == "perlmutter"
        or bool(options.get("constraint"))
        or options.get("cluster") == "perlmutter"
    )


def _sbatch_directives(options, num_nodes, num_gpus, default_cpus=8, default_mem="128GB", default_job="credit"):
    """Return the ``#SBATCH`` resource directive block (no shebang).

    Emits Perlmutter-style GPU directives (``--constraint``/``--qos``/
    ``--gpus-per-node``, no partition/mem, ``_g`` account) when the job targets
    Perlmutter, and the portable ``--gres=gpu:N`` form otherwise.
    """
    perlmutter = _is_perlmutter(options)

    account = options.get("project", options.get("account", "default"))
    if perlmutter and account and not str(account).endswith("_g"):
        account = f"{account}_g"

    lines = [
        f"#SBATCH --job-name={options.get('job_name', default_job)}",
        f"#SBATCH --account={account}",
    ]

    # Constraint / QOS (Perlmutter defaults: -C gpu, -q regular).
    constraint = options.get("constraint") or ("gpu" if perlmutter else None)
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    qos = options.get("qos") or ("regular" if perlmutter else None)
    if qos:
        lines.append(f"#SBATCH --qos={qos}")

    # Partition: required on generic sites, omitted on Perlmutter unless set.
    partition = options.get("partition", options.get("queue")) or (None if perlmutter else "gpu")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")

    lines += [
        f"#SBATCH --nodes={num_nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]

    # GPU request: --gpus-per-node when a constraint is used, else --gres.
    if constraint:
        lines.append(f"#SBATCH --gpus-per-node={num_gpus}")
    else:
        lines.append(f"#SBATCH --gres={_gres(num_gpus, options.get('gpu_type'))}")

    lines.append(f"#SBATCH --cpus-per-task={options.get('ncpus', default_cpus)}")

    mem = options.get("mem", None if perlmutter else default_mem)
    if mem:
        lines.append(f"#SBATCH --mem={mem}")

    lines.append(f"#SBATCH --time={options.get('walltime', '12:00:00')}")
    return "\n".join(lines)


def _module_lines(options):
    """Return shell lines for optional ``module load`` / extra env setup.

    On Perlmutter, falls back to the NCCL/libfabric module and env vars that
    route torch's bundled NCCL over the Slingshot fabric when the config does
    not override ``modules`` / ``env_setup``.
    """
    perlmutter = _is_perlmutter(options)

    lines = []
    modules = options.get("modules")
    if not modules and perlmutter:
        modules = _PERLMUTTER_MODULES
    if modules:
        if isinstance(modules, (list, tuple)):
            modules = " ".join(str(m) for m in modules)
        lines.append(f"module load {modules}")
    # On Perlmutter, prepend the NCCL/libfabric exports, then any config lines
    # (so a config can add NCCL_DEBUG etc. without dropping the defaults).
    env_setup = list(_PERLMUTTER_ENV_SETUP) if perlmutter else []
    cfg_env = options.get("env_setup")
    if cfg_env:
        env_setup += [cfg_env] if isinstance(cfg_env, str) else list(cfg_env)
    lines.extend(str(line) for line in env_setup)
    return "\n".join(lines)


def _resolve_torchrun(conda):
    """Return the torchrun path for a conda env name or full path."""
    if "/" in conda:
        return f"{conda}/bin/torchrun"
    return f"$(conda info --base)/envs/{conda}/bin/torchrun"


def _submit_script(script, save_loc, launch):
    """Write ``launch.sh``, optionally ``sbatch`` it, copy to *save_loc*, return job id."""
    script = re.sub(r"^\s+", "", script, flags=re.MULTILINE)

    with open("launch.sh", "w") as script_file:
        script_file.write(script)

    jobid = None
    if launch:
        out = subprocess.Popen(
            "sbatch launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        out = out.decode("utf-8").strip()
        # sbatch prints "Submitted batch job 123456"
        match = re.search(r"(\d+)", out)
        jobid = match.group(1) if match else out
        logger.info(jobid)

    if save_loc:
        dst = os.path.join(save_loc, "launch.sh")
        if os.path.realpath("launch.sh") != os.path.realpath(dst):
            try:
                shutil.copy("launch.sh", dst)
            except shutil.SameFileError:
                pass
    if launch:
        try:
            os.remove("launch.sh")
        except FileNotFoundError:
            pass
    return jobid


def launch_script(config_file, script_path, launch=True, backend="nccl"):
    """Generate and optionally submit a single-node SLURM script using torchrun.

    Args:
        config_file (str): Path to the YAML configuration file.
        script_path (str): Path to the script that will be executed by the SLURM job.
        launch (bool, optional): If True, submit the job with ``sbatch``. Defaults to True.
        backend (str, optional): Backend for distributed training. Defaults to 'nccl'.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    options = _slurm_options(config)

    num_gpus = options.get("ngpus", 1)
    save_loc = os.path.expandvars(config["save_loc"])
    config_save_path = os.path.join(save_loc, "model.yml")

    conda = options.get("conda", "credit")
    torchrun = _resolve_torchrun(conda)
    modules = _module_lines(options)

    directives = _sbatch_directives(options, 1, num_gpus, default_cpus=8, default_mem="128GB", default_job="credit")

    script = f"""#!/bin/bash -l
    {directives}

    source ~/.bashrc
    {modules}
    conda activate {conda}

    {torchrun} --standalone --nnodes=1 --nproc-per-node={num_gpus} \\
        {script_path} -c {config_save_path} --backend {backend}
    """

    return _submit_script(script, save_loc, launch)


def launch_script_mpi(config_file, script_path, launch=True, backend="nccl"):
    """Generate and optionally submit a multi-node SLURM script using torchrun.

    Uses ``srun`` to launch one torchrun per node with a c10d rendezvous on the
    head node -- the SLURM equivalent of the mpiexec launcher in :mod:`credit.pbs`.

    Args:
        config_file (str): Path to the YAML configuration file.
        script_path (str): Path to the script that will be executed.
        launch (bool, optional): If True, submit the job with ``sbatch``. Defaults to True.
        backend (str, optional): Backend for distributed training. Defaults to 'nccl'.
    """
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    options = _slurm_options(config)

    num_nodes = options.get("nodes", 1)
    num_gpus = options.get("ngpus", 1)

    save_loc = os.path.expandvars(config["save_loc"])
    config_save_path = os.path.join(save_loc, "model.yml")

    # Copy the config to model.yml in save_loc
    source_path = config_file
    destination_path = config_save_path
    if os.path.exists(destination_path) and os.path.realpath(source_path) != os.path.realpath(destination_path):
        os.remove(destination_path)
        logger.info(f"Removed the old model.yml at {destination_path}")
    try:
        os.makedirs(save_loc, exist_ok=True)
        shutil.copy(source_path, destination_path)
        logger.info(f"Copied the new {source_path} to {destination_path}")
    except shutil.SameFileError:
        pass

    conda = options.get("conda", "credit")
    torchrun = _resolve_torchrun(conda)
    modules = _module_lines(options)

    directives = _sbatch_directives(
        options, num_nodes, num_gpus, default_cpus=8, default_mem="128GB", default_job="credit"
    )

    script = f"""#!/bin/bash -l
    {directives}

    source ~/.bashrc
    {modules}
    conda activate {conda}

    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO

    echo "Number of nodes: {num_nodes}"
    echo "Number of GPUs per node: {num_gpus}"
    echo "Total number of GPUs: {num_nodes * num_gpus}"

    # Head node for the torchrun rendezvous
    nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
    head_node=${{nodes[0]}}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{{print $1}}')
    echo "Head node IP: $head_node_ip"

    srun {torchrun} \\
        --nnodes={num_nodes} \\
        --nproc-per-node={num_gpus} \\
        --rdzv-id="$SLURM_JOB_ID" \\
        --rdzv-backend=c10d \\
        --rdzv-endpoint="$head_node_ip:29500" \\
        {script_path} -c {config_save_path} --backend {backend}
    """

    return _submit_script(script, save_loc, launch)


def launch_script_torchrun(config_file, script_path, launch=True, backend="nccl"):
    """Generate and optionally submit a SLURM script using torchrun.

    Preferred for FSDP2 / v2-parallelism jobs.  Single-node jobs use c10d +
    localhost; multi-node jobs broadcast the head node IP for the rendezvous
    endpoint via ``srun``.

    Args:
        config_file (str): Path to the YAML config file.
        script_path (str): Path to the training script (e.g., applications/train_gen2.py).
        launch (bool): If True, submit with ``sbatch``. Defaults to True.
        backend (str): torch.distributed backend. Defaults to 'nccl'.
    """
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    options = _slurm_options(config)
    num_nodes = options.get("nodes", 1)
    num_gpus = options.get("ngpus", 4)

    save_loc = os.path.expandvars(config["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    config_save_path = os.path.join(save_loc, "model.yml")

    try:
        shutil.copy(config_file, config_save_path)
    except shutil.SameFileError:
        pass

    conda = options.get("conda", "credit")
    torchrun = _resolve_torchrun(conda)
    modules = _module_lines(options)

    if num_nodes == 1:
        launch_cmd = (
            f"{torchrun} \\\n"
            f"    --standalone \\\n"
            f"    --nnodes=1 \\\n"
            f"    --nproc-per-node={num_gpus} \\\n"
            f"    {script_path} \\\n"
            f"        -c {config_save_path}"
        )
    else:
        launch_cmd = (
            f'nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )\n'
            f"head_node=${{nodes[0]}}\n"
            f"head_node_ip=$(srun --nodes=1 --ntasks=1 -w \"$head_node\" hostname --ip-address | awk '{{print $1}}')\n\n"
            f"srun {torchrun} \\\n"
            f"    --nnodes={num_nodes} \\\n"
            f"    --nproc-per-node={num_gpus} \\\n"
            f'    --rdzv-id="$SLURM_JOB_ID" \\\n'
            f"    --rdzv-backend=c10d \\\n"
            f'    --rdzv-endpoint="$head_node_ip:29500" \\\n'
            f"    {script_path} \\\n"
            f"        -c {config_save_path}"
        )

    directives = _sbatch_directives(
        options, num_nodes, num_gpus, default_cpus=64, default_mem="480GB", default_job="credit_v2"
    )

    script = f"""#!/bin/bash -l
{directives}

source ~/.bashrc
{modules}
conda activate {conda}

export PYTHONPATH="{os.path.dirname(str(script_path))}:${{PYTHONPATH:-}}"
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN

echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "Nodes  : {num_nodes}  GPUs/node: {num_gpus}"

{launch_cmd}

echo "Done at $(date)"
"""
    # This script is not left-stripped: the launch_cmd relies on indentation-free
    # heredoc-style body already, and SBATCH directives must start at column 0.
    with open("launch.sh", "w") as f:
        f.write(script)

    jobid = None
    if launch:
        out = subprocess.Popen(
            "sbatch launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        out = out.decode("utf-8").strip()
        match = re.search(r"(\d+)", out)
        jobid = match.group(1) if match else out
        logger.info(jobid)

    dst = os.path.join(save_loc, "launch.sh")
    if os.path.realpath("launch.sh") != os.path.realpath(dst):
        try:
            shutil.copy("launch.sh", dst)
        except shutil.SameFileError:
            pass

    return jobid


def get_num_cpus():
    """Return the number of CPUs available to the current job.

    Inside a SLURM allocation this reads ``SLURM_CPUS_ON_NODE``; otherwise it
    falls back to :func:`os.cpu_count`.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        return int(slurm_cpus)
    return int(os.cpu_count())


if __name__ == "__main__":
    config_file = "../config/vit2d.yml"
    script_path = "../applications/trainer_vit2d.py"
    launch_script(config_file, script_path, launch=False)
