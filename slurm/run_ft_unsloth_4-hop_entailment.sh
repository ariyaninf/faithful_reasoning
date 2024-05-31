#!/bin/bash
#SBATCH --job-name=ft_4-hop_entailment
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

python3 ft_unsloth_llama3.py \
  --dataset_name 2sat_50mixVars_50fixCls_0.3_4-hop_10K_500_OR \
  --dataset_di dataset/Entailments/50_vars_50_cls \
  --model_id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --output_dir output \
  --prompt_type 1 \
  --batch_size 2 \
  --epochs 2

