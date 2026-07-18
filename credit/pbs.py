import re
import os
import yaml
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)


def launch_script(config_file, script_path, launch=True, backend="nccl"):
    """Generates and optionally launches a PBS script for a single-node MPI job on Casper.

    Args:
        config_file (str): Path to the YAML configuration file.
        script_path (str): Path to the script that will be executed by the PBS job.
        launch (bool, optional): If True, the PBS job will be submitted to the queue. Defaults to True.
        backend (str, optional): Backend for distributed training. Defaults to 'nccl'.
    """

    # Load the configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Extract PBS options from the config
    pbs_options = config["pbs"]

    num_gpus = pbs_options.get("ngpus", 1)

    save_loc = os.path.expandvars(config["save_loc"])
    config_save_path = os.path.join(save_loc, "model.yml")

    # Generate the PBS script
    script = f"""#!/bin/bash -l
    #PBS -N {pbs_options["job_name"]}
    #PBS -l select=1:ncpus={pbs_options["ncpus"]}:ngpus={num_gpus}:mem={pbs_options["mem"]}:gpu_type={pbs_options["gpu_type"]}
    #PBS -l walltime={pbs_options["walltime"]}
    #PBS -A {pbs_options["project"]}
    #PBS -q {pbs_options["queue"]}
    #PBS -j oe
    #PBS -k eod

    source ~/.bashrc

    conda activate {pbs_options["conda"]}

    mpirun -np {num_gpus} --bind-to none python {script_path} -c {config_save_path} --backend {backend}
    """

    script = re.sub(r"^\s+", "", script, flags=re.MULTILINE)

    # Save the script to a file
    with open("launch.sh", "w") as script_file:
        script_file.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        logger.info(jobid)
        save_loc = os.path.expandvars(config["save_loc"])
        if not os.path.exists(os.path.join(save_loc, "launch.sh")):
            shutil.copy("launch.sh", os.path.join(save_loc, "launch.sh"))
        os.remove("launch.sh")


def launch_script_mpi(config_file, script_path, launch=True, backend="nccl"):
    """Generates and optionally launches a PBS script for a multi-node MPI job.

    Args:
        config_file (str): Path to the YAML configuration file.
        script_path (str): Path to the script that will be executed by the MPI job.
        launch (bool, optional): If True, the PBS job will be submitted to the queue. Defaults to True.
        backend (str, optional): Backend to be used for distributed training (e.g., 'nccl'). Defaults to 'nccl'.
    """

    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    # Extract PBS options from the config
    pbs_options = config.get("pbs", {})

    user = os.environ.get("USER")
    num_nodes = pbs_options.get("nodes", 1)
    num_gpus = pbs_options.get("ngpus", 1)
    total_gpus = num_nodes * num_gpus
    total_ranks = total_gpus

    # Create the CUDA_VISIBLE_DEVICES string
    cuda_devices = ",".join(str(i) for i in range(num_gpus))
    save_loc = os.path.expandvars(config["save_loc"])

    config_save_path = os.path.join(save_loc, "model.yml")

    # Define the source and destination paths
    source_path = config_file
    destination_path = config_save_path

    # Only delete the original if the source and destination paths are different
    if os.path.exists(destination_path) and os.path.realpath(source_path) != os.path.realpath(destination_path):
        os.remove(destination_path)
        logger.info(f"Removed the old model.yml at {destination_path}")

    try:
        shutil.copy(source_path, destination_path)
        logger.info(f"Copied the new {source_path} to {destination_path}")
    except shutil.SameFileError:
        pass

    # Generate the PBS script
    script = f"""#!/bin/bash
    #PBS -A {pbs_options.get("project", "default_project")}
    #PBS -N {pbs_options.get("job_name", "default_job")}
    #PBS -l walltime={pbs_options.get("walltime", "00:10:00")}
    #PBS -l select={num_nodes}:ncpus={pbs_options.get("ncpus", 1)}:ngpus={num_gpus}:mem={pbs_options.get("mem", "4GB")}
    #PBS -q {pbs_options.get("queue", "default_queue")}
    #PBS -j oe
    #PBS -k eod

    # Load modules
    module purge
    module load ncarenv/24.12
    module reset
    module load gcc craype cray-mpich cuda cudnn/8.9.7.29-12 conda mkl
    conda activate {pbs_options.get("conda", "credit")}

    # Export environment variables
    export LSCRATCH=/glade/derecho/scratch/{user}/
    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO

    export CUDA_VISIBLE_DEVICES={cuda_devices}

    # logger.info the results
    echo "Number of nodes: {num_nodes}"
    echo "Number of GPUs per node: {num_gpus}"
    echo "Total number of GPUs: {total_gpus}"

    # Log in to WandB if needed
    # wandb login 02d2b1af00b5df901cb2bee071872de774781520

    # Launch MPIs
    nodes=( $( cat $PBS_NODEFILE ) )
    echo nodes: $nodes

    # Find headnode's IP:
    head_node=${{nodes[0]}}
    head_node_ip=$(ssh $head_node hostname -i | awk '{{print $1}}')

    MASTER_ADDR=$head_node_ip MASTER_PORT=1234 mpiexec -n {total_ranks} --ppn 4 --cpu-bind none python {script_path} -c {config_save_path} --backend {backend}
    """

    script = re.sub(r"^\s+", "", script, flags=re.MULTILINE)

    # Save the script to a file
    with open("launch.sh", "w") as script_file:
        script_file.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        logger.info(jobid)

        # Copy launch.sh to the design location
        # Define the source and destination paths
        source_path = "launch.sh"
        destination_path = os.path.join(save_loc, "launch.sh")

        # Only delete the original if the source and destination paths are different
        if os.path.exists(destination_path) and os.path.realpath(source_path) != os.path.realpath(destination_path):
            os.remove(destination_path)
            logger.info(f"Removed the old launch.sh at {destination_path}")

        try:
            shutil.copy(source_path, destination_path)
            logger.info(f"Generated the new script at {destination_path}")
        except shutil.SameFileError:
            pass


def launch_script_torchrun(config_file, script_path, launch=True, backend="nccl"):
    """Generates and optionally launches a PBS script using torchrun.

    Preferred over launch_script_mpi for FSDP2 / v2-parallelism jobs — torchrun
    manages rendezvous and sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
    Single-node jobs use c10d + localhost; multi-node jobs broadcast the head
    node IP for the rendezvous endpoint.

    Args:
        config_file (str): Path to the YAML config file.
        script_path (str): Path to the training script (e.g., applications/train_gen2.py).
        launch (bool): If True, submit with qsub. Defaults to True.
        backend (str): torch.distributed backend. Defaults to 'nccl'.
    """
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    pbs_options = config.get("pbs", {})
    user = os.environ.get("USER")
    num_nodes = pbs_options.get("nodes", 1)
    num_gpus = pbs_options.get("ngpus", 4)

    save_loc = os.path.expandvars(config["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    config_save_path = os.path.join(save_loc, "model.yml")

    try:
        shutil.copy(config_file, config_save_path)
    except shutil.SameFileError:
        pass

    conda = pbs_options.get("conda", "credit")
    # Resolve torchrun from the conda env's bin directory
    if "/" in conda:
        torchrun = f"{conda}/bin/torchrun"
    else:
        torchrun = f"$(conda info --base)/envs/{conda}/bin/torchrun"

    total_ranks = num_nodes * num_gpus
    cuda_devices = ",".join(str(i) for i in range(num_gpus))

    launch_cmd = (
        f"${{TORCHRUN}} \\\n"
        f"    --nnodes=1 \\\n"
        f"    --nproc-per-node={num_gpus} \\\n"
        f"    --rdzv-backend=c10d \\\n"
        f'    --rdzv-endpoint="localhost:29500" \\\n'
        f"    {script_path} \\\n"
        f"        -c {config_save_path}"
        if num_nodes == 1
        else (
            f"nodes=( $( cat $PBS_NODEFILE ) )\n"
            f"head_node=${{nodes[0]}}\n"
            f"head_node_ip=$(ssh $head_node hostname -i | awk '{{print $1}}')\n\n"
            f"MASTER_ADDR=$head_node_ip MASTER_PORT=29500 \\\n"
            f"mpiexec -n {total_ranks} --ppn {num_gpus} --cpu-bind none \\\n"
            f"    {torchrun.replace('/bin/torchrun', '/bin/python')} {script_path} \\\n"
            f"        -c {config_save_path}"
        )
    )

    torchrun_line = f"TORCHRUN={torchrun}\n" if num_nodes == 1 else ""

    script = f"""#!/bin/bash -l
#PBS -A {pbs_options.get("project", "NAML0001")}
#PBS -N {pbs_options.get("job_name", "credit_v2")}
#PBS -l walltime={pbs_options.get("walltime", "01:00:00")}
#PBS -l select={num_nodes}:ncpus={pbs_options.get("ncpus", 64)}:ngpus={num_gpus}:mem={pbs_options.get("mem", "480GB")}
#PBS -q {pbs_options.get("queue", "develop@desched1")}
#PBS -j oe
#PBS -k eod

module --force purge
module load ncarenv/24.12 nvhpc cuda/12.3.2 cray-mpich conda

{torchrun_line}export PYTHONPATH="{os.path.dirname(str(script_path))}:${{PYTHONPATH:-}}"
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES={cuda_devices}
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1


echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "Nodes  : {num_nodes}  GPUs/node: {num_gpus}"

{launch_cmd}

echo "Done at $(date)"
"""
    script = re.sub(r"^\s+", "", script, flags=re.MULTILINE)

    with open("launch.sh", "w") as f:
        f.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        logger.info(jobid)

        dst = os.path.join(save_loc, "launch.sh")
        try:
            shutil.copy("launch.sh", dst)
        except shutil.SameFileError:
            pass

    return


def _build_pbsdsh_script(pbs_options, script_path, config_save_path):
    """Return a multi-node PBS script that launches torchrun via pbsdsh instead of mpiexec.

    pbsdsh is PBS's own task-manager launcher (no MPI involved). This removes the last
    dependency on cray-mpich from the *launch* path: NCCL communication on Derecho already
    goes over libfabric/CXI (Slingshot-11) via the OFI plugin regardless of what spawned the
    processes, and PyTorch does not need an MPI-aware build unless torch.distributed's
    separate 'mpi' backend is used (CREDIT always uses 'nccl').

    Rank assignment is static, not the elastic c10d rendezvous: each node's torchrun is
    launched independently via its own ``pbsdsh -n <i>`` call, so ranks are handed out with
    explicit ``--node-rank`` rather than negotiated by arrival order (arrival order would
    race across independently-spawned tasks).

    PROTOTYPE — five live attempts on Derecho (2 nodes, develop@desched1):
      1. First attempt: MASTER_ADDR/node discovery resolved, then the job vanished from the
         queue with zero torchrun/Python output and no error text — consistent with an abrupt
         kill before anything downstream could flush output. Root cause not confirmed, but
         added ``pbsdsh -v`` (verbose error/exit-status reporting, per ``man pbsdsh``) and
         per-node log files written to disk independently of the job's own stdout stream, so
         a repeat leaves diagnostics behind either way.
      2. Second attempt (with the above diagnostics): every node's pbsdsh task reported
         **exit status 0 with zero output** — no torchrun banner, no Python error, nothing.
         That combination (clean exit, no output at all) is the signature of the original
         inline ``bash -l -c "<multi-line string>"`` command silently running only its first
         line: PBS's ``tm_spawn`` does not reliably preserve a single quoted argument
         containing literal newlines, so everything after ``conda activate ...;`` became
         inert, never-executed positional parameters. Fixed by writing each node's full
         command to its own plain script file (via an unquoted heredoc, so MASTER_ADDR /
         TORCHRUN / node-rank / etc. are baked in as literal values at write time) and having
         pbsdsh execute that file by path instead.
      3. Third attempt (with the script-file fix): real output at last — ``conda: command
         not found`` on both nodes, then an ``OSError: libmkl_intel_lp64.so.2`` from torchrun.
         pbsdsh spawns a fresh login shell on a (possibly different) node that does not
         inherit the outer script's already-loaded modules, so ``conda activate`` had no
         ``conda`` command to run, the env was never actually activated, and torch's MKL
         dependency (normally resolved via the activated env's lib path) couldn't be found.
         "Fixed" by repeating ``module load ncarenv/24.12 nvhpc cuda/12.3.2 conda`` inside
         each per-node script before ``conda activate`` — but see attempt 4.
      4. Fourth attempt (with the module-load fix): ``module: command not found`` AND
         ``conda: command not found`` — pbsdsh's spawned login shell has *neither* available,
         so nothing that depends on the module system can work there, no matter what the
         outer script or the per-node script itself does. Fixed by bypassing both entirely:
         when ``pbs.conda`` is an absolute path, PATH and LD_LIBRARY_PATH are pointed straight
         at that env's ``bin``/``lib`` dirs (which is what ``conda activate`` would have set
         up anyway).
      5. Fifth attempt (with the PATH/LD_LIBRARY_PATH fix) — **this is the mechanism working**:
         both nodes correctly launched torchrun, rendezvoused, and split into the expected 8
         total ranks (4 local ranks × 2 nodes) with correct rank/local-rank/host assignment.
         The run then failed with ``ImportError: cannot import name
         'distributed_model_wrapper_gen2' from 'credit.distributed'`` — but that's the
         *shared conda env's own pip-installed* ``credit`` package (stale, predates this
         branch), not a pbsdsh/rendezvous problem: nothing pointed ``PYTHONPATH`` at the dev
         checkout being tested, so Python resolved the import from site-packages instead. The
         launcher itself (node discovery, MASTER_ADDR resolution, per-node script generation,
         env setup, torchrun rendezvous across 2 nodes) is validated; getting a fully clean
         training run would need ``PYTHONPATH`` pointed at the checkout under test (not
         attempted here, since it wasn't needed to answer the actual question: does pbsdsh
         work as a launcher).

    Confirm on an actual Derecho allocation before using this in place of launch_script_torchrun:
      - A full end-to-end training run (not just rendezvous) — attempt 5 above validated the
        launch mechanism but never got past import time due to the unrelated stale-package
        issue, so actual gradient/NCCL-communication behavior during training is unverified.
      - Name-based ``pbs.conda`` values still fall back to module+activate, which attempt 4
        showed does NOT work in pbsdsh's spawned shell (neither ``module`` nor ``conda`` are
        available there) — use an absolute conda path in ``pbs.conda`` until that path is
        fixed too (e.g. by resolving the same PATH/LD_LIBRARY_PATH bypass for a conda name via
        a config-supplied base path, rather than relying on ``conda info --base``).
      - That ``pbsdsh -n <i>`` targets nodes in the same order as ``$PBS_NODEFILE`` (node 0
        here MUST be the same host as MASTER_ADDR, or rank 0's torchrun will try to bind a
        c10d store on an IP it doesn't own). Attempt 5's correct rank-to-host assignment is
        consistent with this holding on this system, but wasn't verified with node counts > 2.
      - GPU/CPU affinity: mpiexec's ``--cpu-bind none`` currently leaves binding to the
        scheduler; pbsdsh does no such binding, so CUDA_VISIBLE_DEVICES is set explicitly
        per node below, but NUMA/CPU affinity is not — watch for perf regressions.
      - Failure handling: the loop below waits on every pbsdsh task and fails the job if any
        one of them fails, but a genuinely hung (not crashed) rank won't be caught by this.

    Args:
        pbs_options (dict): The ``pbs:`` block from the config.
        script_path (str): Path to the training script (e.g., credit/applications/train_gen2.py).
        config_save_path (str): Path to the config copy the training script should read.

    Returns:
        str: The full PBS batch script.
    """
    num_nodes = pbs_options.get("nodes", 1)
    num_gpus = pbs_options.get("ngpus", 4)
    conda = pbs_options.get("conda", "credit")
    cuda_devices = ",".join(str(i) for i in range(num_gpus))

    if "/" in conda:
        torchrun = f"{conda}/bin/torchrun"
        # A live attempt found that pbsdsh's spawned login shell has neither 'module' nor
        # 'conda' available ("command not found" for both, twice, on this system), so
        # 'module load ...; conda activate' cannot work here regardless of what the outer
        # script does. Point PATH/LD_LIBRARY_PATH straight at the env's own bin/lib dirs
        # instead — this is what 'conda activate' would have set up anyway, and fixed the
        # OSError: libmkl_intel_lp64.so.2 load failure without needing 'conda'/'module' at all.
        per_node_env_setup = f'export PATH="{conda}/bin:$PATH"\nexport LD_LIBRARY_PATH="{conda}/lib:$LD_LIBRARY_PATH"'
    else:
        torchrun = f"$(conda info --base)/envs/{conda}/bin/torchrun"
        # Name-based conda envs have no known bin/lib path to export directly, so this falls
        # back to module+activate — NOT confirmed to work, since the one live test so far
        # showed 'module'/'conda' both missing in pbsdsh's spawned shell. Prefer an absolute
        # conda path in pbs.conda until this is verified.
        per_node_env_setup = (
            f"module --force purge\nmodule load ncarenv/24.12 nvhpc cuda/12.3.2 conda\nconda activate {conda}"
        )

    script = f"""#!/bin/bash -l
#PBS -A {pbs_options.get("project", "NAML0001")}
#PBS -N {pbs_options.get("job_name", "credit_v2_pbsdsh")}
#PBS -l walltime={pbs_options.get("walltime", "01:00:00")}
#PBS -l select={num_nodes}:ncpus={pbs_options.get("ncpus", 64)}:ngpus={num_gpus}:mem={pbs_options.get("mem", "480GB")}
#PBS -q {pbs_options.get("queue", "develop@desched1")}
#PBS -j oe
#PBS -k eod

module --force purge
module load ncarenv/24.12 nvhpc cuda/12.3.2 conda
conda activate {conda}

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
# NCCL talks over libfabric/CXI on Derecho's Slingshot-11 fabric regardless of launcher —
# this does not depend on cray-mpich being loaded.
export NCCL_NET=OFI

TORCHRUN={torchrun}
CONFIG={config_save_path}
NUM_NODES={num_nodes}
NUM_GPUS={num_gpus}
MASTER_PORT=29500

# Per-node logs are written to disk as each pbsdsh task runs, independently of whatever
# happens to the job's own stdout/stderr stream — kept even if the job is killed abruptly.
LOGDIR=$(dirname "${{CONFIG}}")/pbsdsh_logs
mkdir -p "${{LOGDIR}}"

# nodes[0] is used both to resolve MASTER_ADDR and as the target of `pbsdsh -n 0` below —
# this assumes pbsdsh's node indexing matches $PBS_NODEFILE order (see caveats above).
nodes=( $( cat $PBS_NODEFILE ) )
MASTER_ADDR=$(ssh "${{nodes[0]}}" hostname -i | awk '{{print $1}}')
echo "Master : ${{MASTER_ADDR}}:${{MASTER_PORT}}"
echo "Nodes  : ${{nodes[@]}}"
echo "Logs   : ${{LOGDIR}}"

pids=()
for i in "${{!nodes[@]}}"; do
    # A plain script file, not an inline multi-line -c string: pbsdsh's tm_spawn does not
    # reliably preserve a single quoted argument containing literal newlines (a first live
    # attempt with an inline `bash -l -c "<multi-line>"` string exited status 0 with zero
    # output on every node — consistent with everything after the first line being silently
    # dropped as inert positional params rather than executed).
    # Heredoc delimiter is unquoted on purpose: the outer (already-running) script expands
    # ${{TORCHRUN}}/${{MASTER_ADDR}}/etc. immediately, baking a fully-resolved, self-contained
    # script per node rather than depending on whether pbsdsh propagates shell variables
    # (as opposed to exported env vars) to the remote task at all.
    node_script="${{LOGDIR}}/node_${{i}}.sh"
    cat > "${{node_script}}" <<EOF
#!/bin/bash -l
# pbsdsh spawns this on a fresh login shell on a (possibly different) node — a live attempt
# found it has neither 'module' nor 'conda' available at all, so it cannot inherit the outer
# script's environment setup no matter how that's done there. See per_node_env_setup above.
{per_node_env_setup}
export CUDA_VISIBLE_DEVICES={cuda_devices}
export NCCL_NET=OFI
${{TORCHRUN}} \\
    --nnodes=${{NUM_NODES}} \\
    --node-rank=$i \\
    --nproc-per-node=${{NUM_GPUS}} \\
    --rdzv-backend=static \\
    --master-addr=${{MASTER_ADDR}} \\
    --master-port=${{MASTER_PORT}} \\
    {script_path} -c ${{CONFIG}}
EOF
    chmod +x "${{node_script}}"
    pbsdsh -v -n "$i" -- bash "${{node_script}}" > "${{LOGDIR}}/node_${{i}}.log" 2>&1 &
    pids+=($!)
done

# Fail the job if any node's pbsdsh task fails, instead of hanging waiting on the others.
status=0
for pid in "${{pids[@]}}"; do
    wait "$pid" || status=1
done

for i in "${{!nodes[@]}}"; do
    echo "--- node ${{i}} (${{nodes[$i]}}) log ---"
    cat "${{LOGDIR}}/node_${{i}}.log"
done

exit $status
"""
    return re.sub(r"^[ \t]+", "", script, flags=re.MULTILINE)


def launch_script_pbsdsh(config_file, script_path, launch=True, backend="nccl"):
    """Generates and optionally launches a multi-node torchrun job using pbsdsh instead of mpiexec.

    PROTOTYPE, meant to be tried side-by-side with launch_script_torchrun (mpiexec-based) on a
    real Derecho allocation, not to replace it yet — see _build_pbsdsh_script for the specific
    assumptions that still need to be confirmed on hardware.

    Args:
        config_file (str): Path to the YAML config file.
        script_path (str): Path to the training script (e.g., credit/applications/train_gen2.py).
        launch (bool): If True, submit with qsub. Defaults to True.
        backend (str): torch.distributed backend. Unused directly (NCCL is assumed) but kept
            for signature parity with the other launch_script_* functions.
    """
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    pbs_options = config.get("pbs", {})
    save_loc = os.path.expandvars(config["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    config_save_path = os.path.join(save_loc, "model.yml")

    try:
        shutil.copy(config_file, config_save_path)
    except shutil.SameFileError:
        pass

    script = _build_pbsdsh_script(pbs_options, script_path, config_save_path)

    with open("launch.sh", "w") as f:
        f.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        logger.info(jobid)

        dst = os.path.join(save_loc, "launch.sh")
        try:
            shutil.copy("launch.sh", dst)
        except shutil.SameFileError:
            pass

    return


def get_num_cpus():
    if "glade" in os.getcwd():
        num_cpus = subprocess.run(
            "qstat -f $PBS_JOBID | grep Resource_List.ncpus",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout.split()[-1]
    else:
        num_cpus = os.cpu_count()
    return int(num_cpus)


if __name__ == "__main__":
    config_file = "../config/vit2d.yml"
    # Where does this script live?
    script_path = "../applications/trainer_vit2d.py"
    launch_script(config_file, script_path, launch=False)
    # launch_script_mpi(config_file, script_path, launch = False)
