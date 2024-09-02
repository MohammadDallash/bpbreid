#!/bin/bash

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh

# # Activate the 'dl' environment and run Python scripts
conda activate dl
python track1.py
python track2.py

# # Activate the 'bpbreid' environment and run the remaining scripts
conda activate bpbreid

python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_dukemtmc_infer.yaml
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_infer.yaml
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_occ_duke_infer.yaml
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_p_dukemtmc_infer.yaml

# #Merge
python merge.py
