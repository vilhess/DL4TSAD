#!/bin/bash

# Define datasets to iterate over
datasets=( "smap") #  "nyc_taxi" "ec2_request_latency_system_failure" "swat" "msl" "smd" 

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"

    # Define an associative array of model names and their corresponding scripts
    declare -A models=(
        #["AELSTM"]="python multi_simu.py dataset=$DATASET model=aelstm"
        #["DOC"]="python multi_simu.py dataset=$DATASET model=doc"
        ["PatchTST_REV"]="python multi_simu.py dataset=$DATASET model=patchtst dataset_model.revin=1 " 
        ["PatchTST"]="python multi_simu.py dataset=$DATASET model=patchtst dataset_model.revin=0" 
        #["USAD"]="python multi_simu.py dataset=$DATASET model=usad"
        #["DROCC"]="python multi_simu.py dataset=$DATASET model=drocc"
        #["LSTM_REV"]="python multi_simu.py dataset=$DATASET model=lstm dataset_model.revin=1" 
        #["LSTM"]="python multi_simu.py dataset=$DATASET model=lstm dataset_model.revin=0"  
        #["MADGAN"]="python multi_simu.py dataset=$DATASET model=madgan"
        #["TRANAD"]="python multi_simu.py dataset=$DATASET model=tranad"
        #["PATCHTRAD"]="python multi_simu.py dataset=$DATASET model=patchtrad"
        #["DCDETECTOR"]="python multi_simu.py dataset=$DATASET model=dcdetector"
        #["ANOTRANS"]="python multi_simu.py dataset=$DATASET model=anotrans"
        #["PATCHAD"]="python multi_simu.py dataset=$DATASET model=patchad"
        #["CATCH"]="python multi_simu.py dataset=$DATASET model=catch"
        #["GAT"]="python multi_simu.py dataset=$DATASET model=gat"
        #["JEPATCHTRAD"]="python multi_simu.py dataset=$DATASET model=jepatchtrad"
        #["MOMENT"]="python multi_simu.py dataset=$DATASET model=moment"
    )

    # Define an array to specify the execution order
    order=("AELSTM" "DOC" "PatchTST_REV" "USAD" "LSTM" "LSTM_REV" "TRANAD" "PATCHTRAD" "PATCHAD" "ANOTRANS" "DCDETECTOR" "MADGAN" "CATCH" "GAT" "JEPATCHTRAD"  "PatchTST" "DROCC" "MOMENT")

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
