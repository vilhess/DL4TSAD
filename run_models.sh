#!/bin/bash

# Define datasets to iterate over
datasets=("nyc_taxi" "ec2_request_latency_system_failure" "swat" "msl" "smap" "smd")

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"

    # Define an associative array of model names and their corresponding scripts
    declare -A models=(
        #["AELSTM"]="python -m trainers.aelstm dataset=$DATASET"
        #["DOC"]="python -m trainers.doc dataset=$DATASET"
        #["PatchTST"]="python -m trainers.patchtst dataset=$DATASET"
        #["USAD"]="python -m trainers.usad dataset=$DATASET"
        #["TRANAD"]="python -m trainers.tranad dataset=$DATASET"
        #["TRANSAM"]="python -m trainers.transam dataset=$DATASET"
        #["PATCHTRAD"]="python -m trainers.patchtrad dataset=$DATASET"
        #["DCDETECTOR"]="python -m trainers.dcdetector dataset=$DATASET"
        #["ANOTRANS"]="python -m trainers.anotrans dataset=$DATASET"
        #["PATCHAD"]="python -m trainers.patchad dataset=$DATASET"
        #["DROCC"]="python -m trainers.drocc dataset=$DATASET"
        #["LSTM"]="python -m trainers.lstm dataset=$DATASET"
        #["MADGAN"]="python -m trainers.madgan dataset=$DATASET"
        ["FEDformer"]="python -m trainers.fedformer dataset=$DATASET"
    )

    # Define an array to specify the execution order
    order=("AELSTM" "DOC" "PatchTST" "USAD" "DROCC" "LSTM" "MADGAN" "TRANAD" "TRANSAM" "PATCHTRAD" "DCDETECTOR" "ANOTRANS" "PATCHAD" "FEDformer")

    # Loop through each model in the specified order
    for model in "${order[@]}"; do
        echo "Running model: $model on dataset: $DATASET"
        ${models[$model]}
        echo "Finished running: $model"
        echo "-----------------------------"
    done
done
