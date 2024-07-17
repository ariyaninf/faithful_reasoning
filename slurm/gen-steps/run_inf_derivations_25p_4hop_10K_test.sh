#!/bin/bash
#SBATCH --job-name=inf_gen_steps_25p4h_10K_test
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

python3 inf_gen_derivations.py \
  --dataset_name 2sat_25mixVars_25fixCls_4-hop_10K_INC_500_OR_test \
  --dataset_dir dataset/Derivations_INC \
  --model_id output/gen_Derivations/unsloth/llama-3-8b-Instruct-bnb-4bit/2sat_25mixVars_25fixCls_4-hop_10K_500_OR_prompt_5/lora_model \
  --output_dir output/test_Derivations \
  --prompt_type 5 \
  --max_new_tokens 500




