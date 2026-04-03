#!/bin/bash
#PBS -A NAML0001
#PBS -N credit_v2
#PBS -l walltime=12:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -r n

module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 \
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

conda activate /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221

REPO=/glade/work/schreck/repos/miles-credit-main
export PYTHONPATH=${REPO}:${PYTHONPATH}

CONFIG=${CONFIG:-${REPO}/config/wxformer_1dg_6hr_v2.yml}

export LSCRATCH=/glade/derecho/scratch/schreck/
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

nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')

pbs_select=$(printenv | grep PBS_SELECT)
num_nodes=$(echo "$pbs_select" | cut -d'=' -f2 | cut -d':' -f1)
num_gpus=$(echo "$pbs_select" | grep -oP 'ngpus=\K\d+')
total_gpus=$(( num_nodes * num_gpus ))

echo "Nodes     : $num_nodes"
echo "GPUs/node : $num_gpus"
echo "Total GPUs: $total_gpus"
echo "Config    : $CONFIG"
echo "Head node : $head_node_ip"

cd ${REPO}
RDZV_PORT=$(( RANDOM % 10000 + 20000 ))

mpiexec -n ${total_gpus} --ppn ${num_gpus} \
    torchrun \
        --nnodes=${num_nodes} \
        --nproc-per-node=${num_gpus} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${head_node_ip}:${RDZV_PORT} \
    ${REPO}/applications/train_v2.py -c ${CONFIG}
