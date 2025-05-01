#!/bin/bash

python multi_simu.py dataset=smd model=lstm
python multi_simu.py dataset=smap model=lstm

python multi_simu.py dataset=smap model=doc