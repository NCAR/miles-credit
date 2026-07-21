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


# Module set for the pbsdsh path. Identical to `credit submit`'s derecho module line
# PLUS the libfabric module. NCCL's data plane runs over libfabric/CXI via the
# aws-ofi-nccl plugin (libnccl-net.so, bundled in the conda env); libfabric.so itself
# comes from this module (version-matched to the plugin's build). cray-mpich stays
# loaded even though we no longer launch via mpiexec, because libtorch links Cray's
# libmpi_gnu_123.so at runtime -- dropping it breaks `import torch`, not just MPI launch.
PBSDSH_DERECHO_MODULES = (
    "ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 "
    "cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1 libfabric/1.15.2.0"
)


def _pbsdsh_launch_block(torchrun, script_path, config_path, num_gpus, logdir_parent):
    r"""Return the bash block that launches *script_path* across all job nodes via pbsdsh.

    This is the launcher-specific tail of a PBS script; the caller supplies the ``#PBS``
    header, ``module load``, and ``conda activate`` lines before appending this block. It
    is shared by ``credit/pbs.py``'s legacy full-script builder and ``credit submit``'s
    integrated derecho path so the two stay in lockstep.

    Design:

    * **One pbsdsh task per node**, each running ``torchrun --nproc-per-node`` to fan out
      to that node's GPUs -- NOT torchrun *under* mpiexec, which double-spawns N^2 ranks
      and collides on GPUs (see ``tests/manual/gen2_parallelism/run_smoke_2node.pbs``).
      torchrun sets ``LOCAL_RANK``/``RANK``/``WORLD_SIZE``, which ``get_rank_info()`` reads.
    * **Static rendezvous** (``--rdzv-backend=static`` + ``--master-addr``/``--master-port``),
      not elastic c10d: each node's torchrun is spawned independently, so ranks are handed
      out explicitly via ``--node-rank`` rather than negotiated by arrival order (which
      would race across independently-spawned tasks).
    * **Baked environment.** pbsdsh's spawned per-node shell does not inherit this job's
      module/conda environment (and may lack the ``module``/``conda`` commands entirely),
      so the fully-resolved ``PATH``/``LD_LIBRARY_PATH`` from the mother-superior shell --
      where ``module load`` + ``conda activate`` have already run -- are baked verbatim
      into each per-node script. All nodes are identical hardware/software, so those paths
      are valid everywhere. They carry the conda env's ``bin``, the aws-ofi-nccl plugin
      NCCL uses to talk over libfabric, libfabric itself, and Cray's ``libmpi`` that
      libtorch links against. (Validated in a bare ``env -i`` shell: with only these two
      variables set, ``import torch``/``torchrun``/``import credit`` all resolve.)
    * **PBS Pro node indexing.** ``pbsdsh -n <i>`` targets the (i+1)-th vnode in job order;
      index 0 is the mother superior -- the host running this script -- so ``MASTER_ADDR``
      is resolved locally with ``hostname -i`` rather than via ``ssh`` to a nodefile entry.
      This guarantees rank 0's rendezvous store binds on the node rank 0 actually runs on
      (a sorted nodefile could put a different host first, breaking the bind). The node
      count comes from the deduplicated ``$PBS_NODEFILE``.

    Empirical history that produced this shape (Derecho, 2 nodes, develop queue):

    1. An inline ``bash -l -c "<multi-line string>"`` handed to pbsdsh exited status 0 with
       zero output on every node -- pbsdsh's ``tm_spawn`` does not reliably preserve a
       single quoted argument containing literal newlines, so only the first line ran.
       Fix: write each node's command to its own script file and run that by path.
    2. That surfaced ``conda: command not found``, then ``module: command not found`` --
       pbsdsh's spawned login shell has neither. Fix: stop depending on them there; bake
       the mother-superior's resolved ``PATH``/``LD_LIBRARY_PATH`` (this function).
    3. With the environment baked, both nodes launched torchrun, rendezvoused, and split
       into the expected local-ranks x nodes total ranks with correct rank/host assignment.

    Args:
        torchrun (str): torchrun invocation baked into each per-node script. A bare
            ``"torchrun"`` works because PATH is set from the resolved env; a full path
            is equally fine.
        script_path (str): Application entrypoint (e.g. ``credit/applications/train_gen2.py``).
        config_path (str): Config the entrypoint reads via ``-c``.
        num_gpus (int): GPUs -- and torchrun processes -- per node.
        logdir_parent (str): Directory under which a ``pbsdsh_logs/`` dir is created for
            per-node logs, written to disk independently of the job's own stdout so a
            job killed abruptly still leaves diagnostics behind.

    Returns:
        str: The bash launch block (no ``#PBS`` header, no ``module``/``conda`` lines).
            Intentionally NOT run through a leading-whitespace strip -- the per-node
            heredoc terminator must stay at column 0.
    """
    cuda_devices = ",".join(str(i) for i in range(int(num_gpus)))
    template = r"""
# --- pbsdsh + torchrun multi-node launch -----------------------------------
# Bake the mother-superior's fully-resolved environment into each per-node script,
# because pbsdsh's spawned shell inherits neither modules nor conda. See _pbsdsh_launch_block.
RESOLVED_PATH="$PATH"
RESOLVED_LD="${LD_LIBRARY_PATH:-}"

# One physical node per chunk on Derecho GPU nodes; pbsdsh indexes them in job order and
# index 0 is this (mother-superior) host, so resolve MASTER_ADDR here on the local host.
NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
MASTER_ADDR=$(hostname -i | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 10000 + 20000 ))

LOGDIR="@@LOGDIR@@/pbsdsh_logs"
mkdir -p "$LOGDIR"
echo "Launcher     : pbsdsh + torchrun (static rendezvous)"
echo "Nodes        : ${NUM_NODES}   GPUs/node: @@NGPUS@@"
echo "Master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "Per-node logs: ${LOGDIR}"

pids=()
for (( i=0; i<NUM_NODES; i++ )); do
    node_script="${LOGDIR}/node_${i}.sh"
    # Unquoted heredoc: $RESOLVED_PATH / $MASTER_ADDR / $NUM_NODES / $i expand NOW (this
    # shell), so each per-node script is fully self-contained with literal values -- pbsdsh
    # does not propagate shell state to the spawned task.
    cat > "${node_script}" <<EOF
#!/bin/bash
export PATH="${RESOLVED_PATH}"
export LD_LIBRARY_PATH="${RESOLVED_LD}"
export CUDA_VISIBLE_DEVICES=@@CUDA@@
export LOGLEVEL=INFO
# NCCL_DEBUG=INFO prints the selected transport (expect NET/OFI over libfabric) so a run
# can be confirmed to use libfabric rather than silently falling back to TCP sockets.
export NCCL_DEBUG=INFO
# Redirect torchrun's output to a per-node file on the (shared) LOGDIR from inside the
# per-node script, i.e. written by the remote node itself. PBS routes a pbsdsh task's
# stdout to the *job's* aggregate stream, not to the launching pbsdsh's redirected stdout,
# so capturing it here is the only way to keep clean, un-interleaved per-node logs on disk.
@@TORCHRUN@@ \
    --nnodes=${NUM_NODES} \
    --node-rank=${i} \
    --nproc-per-node=@@NGPUS@@ \
    --rdzv-backend=static \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    @@APP@@ -c @@CONFIG@@ > "${LOGDIR}/node_${i}.out" 2>&1
EOF
    chmod +x "${node_script}"
    # -v: pbsdsh reports each task's exit status to node_<i>.pbsdsh; torchrun's own output
    # goes to node_<i>.out (written by the remote node above). Backgrounded so all nodes
    # run concurrently.
    pbsdsh -v -n "$i" -- bash "${node_script}" > "${LOGDIR}/node_${i}.pbsdsh" 2>&1 &
    pids+=($!)
done

# Wait on every task; fail the job if any node's task fails. (A hung -- not crashed --
# rank is not caught by this.)
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

for (( i=0; i<NUM_NODES; i++ )); do
    echo "--- node ${i} (torchrun) ---"
    cat "${LOGDIR}/node_${i}.out" 2>/dev/null
    echo "--- node ${i} (pbsdsh status) ---"
    cat "${LOGDIR}/node_${i}.pbsdsh" 2>/dev/null
done
exit $status
"""
    return (
        template.replace("@@LOGDIR@@", str(logdir_parent))
        .replace("@@NGPUS@@", str(int(num_gpus)))
        .replace("@@CUDA@@", cuda_devices)
        .replace("@@TORCHRUN@@", str(torchrun))
        .replace("@@APP@@", str(script_path))
        .replace("@@CONFIG@@", str(config_path))
    )


def _build_pbsdsh_script(pbs_options, script_path, config_save_path):
    """Return a full multi-node PBS script that launches torchrun via pbsdsh, not mpiexec.

    Thin wrapper: builds the ``#PBS`` header + module/conda lines from ``pbs_options`` and
    appends :func:`_pbsdsh_launch_block` (which holds the launch logic and its rationale).
    ``credit submit --launcher pbsdsh`` builds the equivalent script through the same block
    helper; this function is the standalone/legacy entry point used by ``launch_script_pbsdsh``.

    Args:
        pbs_options (dict): The ``pbs:`` block from the config.
        script_path (str): Path to the training script (e.g. credit/applications/train_gen2.py).
        config_save_path (str): Path to the config copy the training script should read.

    Returns:
        str: The full PBS batch script.
    """
    num_nodes = pbs_options.get("nodes", 1)
    num_gpus = pbs_options.get("ngpus", 4)
    conda = pbs_options.get("conda", "credit")

    # torchrun resolves from PATH once the env is activated; a full path also works.
    torchrun = f"{conda}/bin/torchrun" if "/" in str(conda) else "torchrun"

    header = f"""#!/bin/bash -l
#PBS -A {pbs_options.get("project", "NAML0001")}
#PBS -N {pbs_options.get("job_name", "credit_v2_pbsdsh")}
#PBS -l walltime={pbs_options.get("walltime", "01:00:00")}
#PBS -l select={num_nodes}:ncpus={pbs_options.get("ncpus", 64)}:ngpus={num_gpus}:mem={pbs_options.get("mem", "480GB")}
#PBS -q {pbs_options.get("queue", "develop@desched1")}
#PBS -j oe
#PBS -k eod
#PBS -r n

module --force purge
module load {PBSDSH_DERECHO_MODULES}
conda activate {conda}
"""
    block = _pbsdsh_launch_block(
        torchrun=torchrun,
        script_path=script_path,
        config_path=config_save_path,
        num_gpus=num_gpus,
        logdir_parent=os.path.dirname(str(config_save_path)) or ".",
    )
    return header + block


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
