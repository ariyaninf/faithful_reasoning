#!/bin/bash
#SBATCH --job-name=ft_mixtral_8x78
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

python3 ft_qlora.py \
  --dataset_name 2sat_15_mixVars_50_mixCls_100K_500_OR \
  --dataset_dir dataset/Entailments_v4.2/ \
  --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --output_dir output/finetuned_model/ \
  --prompt_type 6 \
  --batch_size 4 \
  --epochs 2




