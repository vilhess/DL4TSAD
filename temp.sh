#!/bin/bash


python multi_simu.py dataset=smd model=patchad
python multi_simu.py dataset=smap model=patchad

python multi_simu.py dataset=smd model=lstm dataset_model.revin=0