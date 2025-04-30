#!/bin/bash

python multi_simu.py dataset=smap model=tranad 
python multi_simu.py dataset=smd model=tranad

python multi_simu.py dataset=smap model=lstm 
python multi_simu.py dataset=smd model=lstm


