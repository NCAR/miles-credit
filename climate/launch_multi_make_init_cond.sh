#!/bin/bash
#PBS -A NAML0001
#PBS -N CAMulator_init_cond_00
#PBS -l walltime=00:25:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l job_priority=premium
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -r n
# Load modules
module purge
module load ncarenv/23.09
module reset
module load gcc craype cray-mpich cuda cudnn/8.8.1.3-12 conda
module list
conda activate /glade/work/wchapman/conda-envs/credit-derecho

# conda conda activate /glade/u/home/schreck/.conda/envs/credit-derecho
# Export environment variables
export LSCRATCH=/glade/derecho/scratch/wchapman/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
echo "=== Job layout ==="
echo "Nodes:    $(wc -l < $PBS_NODEFILE)"
echo "GPUs/node: 4"
echo "Total GPUs: $(( $(wc -l < $PBS_NODEFILE) * 4 ))"
# Log in to WandB if needed
# wandb login 02d2b1af00b5df901cb2bee071872de774781520
# Launch MPIs

nodes=( $( cat $PBS_NODEFILE ) )
echo nodes: $nodes

# Find headnode's IP:
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=1234
echo "Computed MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

mpiexec -n $(( ${#nodes[@]} * 4 )) --ppn 4 --cpu-bind none python ./Make_Climate_Initial_Conditions.py -c ./be21_coupled-v2025.2.0_small_future.yml
