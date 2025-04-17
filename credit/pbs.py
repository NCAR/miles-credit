import re
import os
import yaml
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)


def launch_script(config_file, script_path, launch=True):
    """Generates and optionally launches a PBS script for a single-node job.

    Args:
        config_file (str): Path to the YAML configuration file.
        script_path (str): Path to the script that will be executed by the PBS job.
        launch (bool, optional): If True, the PBS job will be submitted to the queue. Defaults to True.
    """

    # Load the configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Extract PBS options from the config
    pbs_options = config["pbs"]

    save_loc = os.path.expandvars(config["save_loc"])
    config_save_path = os.path.join(save_loc, "model.yml")

    # Generate the PBS script
    script = f"""#!/bin/bash -l
    #PBS -N {pbs_options['job_name']}
    #PBS -l select=1:ncpus={pbs_options['ncpus']}:ngpus={pbs_options['ngpus']}:mem={pbs_options['mem']}
    #PBS -l walltime={pbs_options['walltime']}
    #PBS -l gpu_type={pbs_options['gpu_type']}
    #PBS -A {pbs_options['project']}
    #PBS -q {pbs_options['queue']}
    #PBS -j oe
    #PBS -k eod

    source ~/.bashrc

    conda activate {pbs_options['conda']}

    python {script_path} -c {config_save_path}
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
    #PBS -A {pbs_options.get('project', 'default_project')}
    #PBS -N {pbs_options.get('job_name', 'default_job')}
    #PBS -l walltime={pbs_options.get('walltime', '00:10:00')}
    #PBS -l select={num_nodes}:ncpus={pbs_options.get('ncpus', 1)}:ngpus={num_gpus}:mem={pbs_options.get('mem', '4GB')}
    #PBS -q {pbs_options.get('queue', 'default_queue')}
    #PBS -j oe
    #PBS -k eod

    # Load modules
    module purge
    module load ncarenv/24.12
    module reset
    module load gcc craype cray-mpich cuda cudnn/8.9.7.29-12 conda
    conda activate {pbs_options.get('conda', 'credit')}

    # Export environment variables
    export LSCRATCH=/glade/derecho/scratch/{user}/
    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO

    export CUDA_VISIBLE_DEVICES={cuda_devices}

    export NCCL_SOCKET_IFNAME=hsn
    export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
    export MPICH_OFI_NIC_POLICY=GPU
    export MPICH_GPU_SUPPORT_ENABLED=1

    export NCCL_IB_DISABLE=1
    export NCCL_CROSS_NIC=1
    export NCCL_NCHANNELS_PER_NET_PEER=4

    export MPICH_RDMA_ENABLED_CUDA=1
    export NCCL_NET="AWS Libfabric"
    export NCCL_NET_GDR_LEVEL=PBH

    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_CXI_OPTIMIZED_MRS=false
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072

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
