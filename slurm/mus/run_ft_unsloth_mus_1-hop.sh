#!/bin/bash
#SBATCH --job-name=ft_mus_1hop_llama3-8b
#SBATCH --error=log/e.%x.%j
#SBATCH --output=log/o.%x.%j
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --account=scw1997

set -eu

module purge
module load anaconda
module list

source activate
source activate myenv

WORKDIR=/scratch/$USER/faithful_reasoning
cd ${WORKDIR}

for number in 25 50 100
do
  python3 ft_unsloth_mus.py \
  --dataset_name 2sat_${number}mixVars_${number}fixCls_1-hop_100K_muses_500_OR \
  --dataset_dir dataset/MUS/${number}vars_${number}cls \
  --model_id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --output_dir output \
  --prompt_type 5 \
  --batch_size 2 \
  --epochs 2
done




