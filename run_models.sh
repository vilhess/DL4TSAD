#!/bin/bash

# Define datasets to iterate over
datasets=("nyc_taxi" "ec2_request_latency_system_failure" "swat"  "smd" "msl" "smap") # 

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"

    # Define an associative array of model names and their corresponding scripts
    declare -A models=(
        #["AELSTM"]="python main.py dataset=$DATASET model=aelstm"
        #["DOC"]="python main.py dataset=$DATASET model=doc"
        ["PatchTST_REV"]="python main.py dataset=$DATASET model=patchtst dataset_model.revin=1 " 
        ["PatchTST"]="python main.py dataset=$DATASET model=patchtst dataset_model.revin=0" 
        #["USAD"]="python main.py dataset=$DATASET model=usad"
        #["DROCC"]="python main.py dataset=$DATASET model=drocc"
        #["LSTM_REV"]="python main.py dataset=$DATASET model=lstm dataset_model.revin=1" 
        #["LSTM"]="python main.py dataset=$DATASET model=lstm dataset_model.revin=0"  
        #["MADGAN"]="python main.py dataset=$DATASET model=madgan"
        #["TRANAD"]="python main.py dataset=$DATASET model=tranad"
        #["PATCHTRAD"]="python main.py dataset=$DATASET model=patchtrad"
        #["DCDETECTOR"]="python main.py dataset=$DATASET model=dcdetector"
        #["ANOTRANS"]="python main.py dataset=$DATASET model=anotrans"
        #["PATCHAD"]="python main.py dataset=$DATASET model=patchad"
        #["CATCH"]="python main.py dataset=$DATASET model=catch"
        #["GAT"]="python main.py dataset=$DATASET model=gat"
        #["JEPATCHTRAD"]="python main.py dataset=$DATASET model=jepatchtrad"
        #["MOMENT"]="python main.py dataset=$DATASET model=moment"
        #["TIME_MIXER"]="python main.py dataset=$DATASET model=timemixer"
        #["GPT4TS"]="python main.py dataset=$DATASET model=gpt4ts"
    )   

    # Define an array to specify the execution order
    order=("AELSTM" "DOC" "PatchTST" "PatchTST_REV" "USAD" "LSTM" "LSTM_REV" "TRANAD" "PATCHTRAD" "PATCHAD" "ANOTRANS" "DCDETECTOR" "MADGAN" "DROCC" "CATCH" "GAT" "JEPATCHTRAD" "MOMENT" "TIME_MIXER" "GPT4TS")

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
