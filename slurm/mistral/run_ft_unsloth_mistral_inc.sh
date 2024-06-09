#!/bin/bash
#SBATCH --job-name=ft_inc_mistral
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

WORKDIR=/scratch/$USER/LlamaFinetuning
cd ${WORKDIR}

for var in 25 50 100
do
  for hop in 1 4
  do
    python3 ft_unsloth_mistral.py \
    --dataset_name 2sat_${var}mixVars_${var}fixCls_${hop}-hop_100K_inconsistencies_500_OR \
    --dataset_dir dataset/Inconsistencies/${var}vars_${var}cls \
    --model_id unsloth/mistral-7b-instruct-v0.2-bnb-4bit \
    --output_dir output \
    --prompt_type 2 \
    --batch_size 2 \
    --epochs 2
  done
done



