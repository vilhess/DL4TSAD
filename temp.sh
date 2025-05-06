#!/bin/bash

python multi_simu.py dataset=smap model=lstm dataset_model.revin=0
python multi_simu.py dataset=smd model=lstm dataset_model.revin=0


python multi_simu.py dataset=smd model=anotrans
python multi_simu.py dataset=smap model=anotrans

