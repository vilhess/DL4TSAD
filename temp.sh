#!/bin/bash

# Lancer les simulations avec GAT
python multi_simu.py dataset=smd model=gat
python multi_simu.py dataset=smap model=gat

# Lancer les simulations avec PatchTST (sans ReVIN)
python multi_simu.py dataset=swat model=patchtst dataset_model.revin=0
python multi_simu.py dataset=msl model=patchtst dataset_model.revin=0
python multi_simu.py dataset=smd model=patchtst dataset_model.revin=0
python multi_simu.py dataset=smap model=patchtst dataset_model.revin=0

# Lancer les simulations avec PatchAD
python multi_simu.py dataset=msl model=patchad
python multi_simu.py dataset=smd model=patchad
python multi_simu.py dataset=smap model=patchad
