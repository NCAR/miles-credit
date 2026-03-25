#!/bin/bash
#PBS -A NMMM0015
#PBS -N wx_mse_warmup_medium
#PBS -l walltime=12:00:00
#PBS -l select=8:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
#PBS -k eod
# Load modules
module purge
module load ncarenv/24.12
module reset
module load gcc craype cray-mpich cuda cudnn/8.9.7.29-12 conda
conda activate /glade/work/bagherio/conda-envs/credit-derecho
# Export environment variables
export LSCRATCH=/glade/derecho/scratch/bagherio/
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
echo "Number of nodes: 8"
echo "Number of GPUs per node: 4"
echo "Total number of GPUs: 32"
# Log in to WandB if needed
# wandb login 02d2b1af00b5df901cb2bee071872de774781520
# Launch MPIs
nodes=( $( cat $PBS_NODEFILE ) )
echo nodes: $nodes
# Find headnode's IP:
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
MASTER_ADDR=$head_node_ip MASTER_PORT=1234 mpiexec -n 32 --ppn 4 --cpu-bind none python /glade/derecho/scratch/bagherio/cloud.dir/miles-credit/applications/goes_train.py -c /glade/derecho/scratch/bagherio/cloud.dir/goes_10km_train/wx_mse_warmup_medium/model.yml --backend nccl
