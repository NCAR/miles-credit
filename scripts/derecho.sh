#!/bin/bash
# Pass your project code at submit time: qsub -A ${PBS_ACCOUNT:?set PBS_ACCOUNT} <this script>
#PBS -N out
#PBS -l walltime=00:10:00
#PBS -l select=16:ncpus=64:ngpus=4:mem=480GB
##PBS -q main
#PBS -q preempt
#PBS -j oe
#PBS -k eod

# see set_gpu_rank as a provided module --> sets GPUs to be unique when using MPIs
# Rory said we might not have to use mpiexec/torchrun once we run that command

# Load modules
module purge
module load nvhpc cuda cray-mpich conda 
conda activate holodec

# Get a list of allocated nodes
nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
#echo $(ssh $head_node hostname -i) > out

export LSCRATCH=/glade/derecho/scratch/schreck/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
# export OMP_NUM_THREADS=16
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1


# Need to set up CUDA DEVICE IDS for all available GPUs
pbs_select=$(printenv | grep PBS_SELECT)
# Extract the values using cut
num_nodes=$(echo "$pbs_select" | cut -d'=' -f2 | cut -d':' -f1)
num_gpus=$(echo "$pbs_select" | grep -oP 'ngpus=\K\d+')
# Calculate the total number of GPUs
total_gpus=$((num_nodes * num_gpus))

# Print the results
echo "Number of nodes: $num_nodes"
echo "Number of GPUs per node: $num_gpus"
echo "Total number of GPUs: $total_gpus"

# Build the device list 
CUDA_DEVICES=""
# Loop from 0 to total_gpus-1 and append each number to the string
for ((i = 0; i < $total_gpus; i++)); do
    CUDA_DEVICES="${CUDA_DEVICES}${i}"
    # Add a comma unless it's the last number
    if [ $i -lt $((total_gpus-1)) ]; then
        CUDA_DEVICES="${CUDA_DEVICES},"
    fi
done

# Log in to WandB if needed
# wandb login 02d2b1af00b5df901cb2bee071872de774781520
pip install .

# Launch MPIs
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" mpiexec -n $num_nodes --ppn 1 --cpu-bind none torchrun --nnodes=$num_nodes --nproc-per-node=$num_gpus --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip applications/trainer_vit2d.py -c config/vit2d-L.yml

#python applications/trainer_vit3d.py -c config/model.yml
# torchrun --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip applications/trainer_vit3d.py -c config/model.yml