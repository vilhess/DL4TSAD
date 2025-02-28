#!/bin/bash

# Define datasets to iterate over
datasets=("nyc_taxi" "ec2" "swat" "msl" "smap" "smd")

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"

    # Define an associative array of model names and their corresponding scripts
    declare -A models=(
        ["AELSTM"]="python main.py dataset=$DATASET model=aelstm"
        ["DOC"]="python main.py dataset=$DATASET model=doc"
        ["PatchTST"]="python main.py dataset=$DATASET model=patchtst"
        ["USAD"]="python main.py dataset=$DATASET model=usad"
        ["DROCC"]="python main.py dataset=$DATASET model=drocc"
        ["LSTM"]="python main.py dataset=$DATASET model=lstm"
        ["MADGAN"]="python main.py dataset=$DATASET model=madgan"
        ["TRANAD"]="python main.py dataset=$DATASET model=tranad"
        ["TRANSAM"]="python main.py dataset=$DATASET model=transam"
        ["PATCHTRAD"]="python main.py dataset=$DATASET model=patchtrad"
        ["DCDETECTOR"]="python main.py dataset=$DATASET model=dcdetector"
        ["ANOTRANS"]="python main.py dataset=$DATASET model=anotrans"
        ["PATCHAD"]="python main.py dataset=$DATASET model=patchad"
    )

    # Define an array to specify the execution order
    order=("AELSTM" "DOC" "PatchTST" "USAD" "DROCC" "LSTM" "MADGAN" "TRANAD" "TRANSAM" "PATCHTRAD" "DCDETECTOR" "ANOTRANS" "PATCHAD")

    # Loop through each model in the specified order
    for model in "${order[@]}"; do
        echo "Running model: $model on dataset: $DATASET"
        
        # Check if the model exists in the associative array before running
        if [[ -n ${models[$model]} ]]; then
            ${models[$model]}
            echo "Finished running: $model"
        else
            echo "Model $model not found for dataset: $DATASET"
        fi
        
        echo "-----------------------------"
    done
done
