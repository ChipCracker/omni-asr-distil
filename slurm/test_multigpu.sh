#!/bin/bash
#SBATCH --job-name=test-mgpu
#SBATCH --output=logs/test-mgpu_%j.out
#SBATCH --error=logs/test-mgpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --partition=p4
#SBATCH --time=0:05:00
#SBATCH --qos=gpuultimate

cd /nfs1/scratch/students/witzlch88229/projects/omni-asr-distil || exit 1
mkdir -p logs
source .venv/bin/activate

srun python -c "
import torch, torch.distributed as dist, os
dist.init_process_group('nccl')
r = dist.get_rank()
t = torch.ones(1, device=f'cuda:{os.environ.get(\"LOCAL_RANK\", 0)}') * r
dist.all_reduce(t)
print(f'Rank {r}: all_reduce = {t.item()}')
dist.destroy_process_group()
"
