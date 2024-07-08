#!/bin/bash
#SBATCH --job-name=ft_derivation_25p_4hop
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

python3 ft_gen_derivations.py \
  --dataset_name 2sat_25mixVars_25fixCls_4-hop_10K_500_OR \
  --dataset_dir dataset/Derivations/25vars_25cls \
  --model_id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --output_dir output/gen_Derivations \
  --prompt_type 5 \
  --batch_size 2 \
  --epochs 2 \
  --save_steps 1250 \
  --max_seq_length 8192




