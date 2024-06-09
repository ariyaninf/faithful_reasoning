#!/bin/bash
#SBATCH --job-name=zeroshot_llama3_mus
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

for var in 25 50
do
  for hop in 1 4
  do
    python3 zeroshot_llama-3_mus.py \
    --dataset_name 2sat_${var}mixVars_${var}fixCls_${hop}-hop_100K_muses_500_OR \
    --dataset_dir dataset/MUS/${var}vars_${var}cls \
    --model_name unsloth/llama-3-8b-Instruct-bnb-4bit \
    --output_dir output/zero_shot \
    --prompt_type 5
  done
done



