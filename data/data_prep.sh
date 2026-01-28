#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python data/prepare_data.py \
  --pdb_file "inputs/receptor.pdb" \
  --reflig "inputs/ref_ligand.sdf" \
  --output_dir "inputs2" \
  --nt_model "data/models/nt_v2" \
  --rnafm_model "data/models/rnafm" \