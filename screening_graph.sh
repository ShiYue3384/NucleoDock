#!/bin/bash
#SBATCH -J NucleoDock
#SBATCH -p gpu
#SBATCH --qos=high
#SBATCH -c 16                 # 申请 32 核
#SBATCH -t 72:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 
#SBATCH -o CarsiDock_NA.%j.out
#SBATCH -e CarsiDock_NA.%j.err

python  inference_graph.py \
  --config "RNA_graph_presentation.yml" \
  --input "inputs" \
  --ligands "inputs/candidates.txt" \
  --ckpt_path "checkpoints/graph_99.ckpt" \
  --output_dir "outputs/screening_result" \
  --num_threads 1 \
  --cuda_convert

##SBATCH -w gpu04