#!/bin/bash
#SBATCH --job-name=ft_entailed_100K
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

python3 ft_genProofs_causalLM.py \
  --dataset SENT_inferred_2sat_5fixVars_5fixCls_entailed_100K_100_OR \
  --dataset_dir dataset/Synthetic_CounterEx/positive_examples \
  --output_dir output/trained_genProof \
  --model_name Llama-2-13b-chat-hf \
  --model_dir meta-llama \
  --format_type 5 \
  --load_in_bit 4 \
  --batch_size 32 \
  --epochs 3 \
  --save_steps 3000 \
  --max_seq_length 256



