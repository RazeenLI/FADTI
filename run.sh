#!/bin/bash

# nohup ./run.sh > run2.log 2>&1 & 
mkdir -p logs
models=("mean" "median" "knn" "csdi" "fadti" "mtsci" "timemixerpp" "saits" "timesnet" "brits" "timemixer") 
datas=("ett" "weather" "metr_la" "yeast") # "ett" "weather" "metr_la" "yeast"
nfolds=(0 1 2 3 4) 
missrates=(0.1 0.5)
misspatterns=("point" "time")
nsample=100
device="cuda:5"

for model in "${models[@]}"; do
    for data in "${datas[@]}"; do
         for nfold in "${nfolds[@]}"; do
            for missrate in "${missrates[@]}"; do
                for misspattern in "${misspatterns[@]}"; do
                    output_file="logs/${model}_${data}_${nfold}.out"
                    echo "Running: python run_all.py --model \"$model\" --data \"$data\" --nsample $nsample --device \"$device\" --nfold $nfold --misspattern \"$misspattern\" --missrate $missrate > $output_file"
                    python run_all.py --model "$model" --data "$data" --nsample $nsample --device "$device" --nfold $nfold --misspattern "$misspattern" --missrate $missrate > "$output_file"
                done
            done
         done
    done
done
