#!/bin/bash

mkdir -p logs
models=("mean" "median" "knn" "csdi" "fadti" "mtsci" "timemixerpp" "saits" "timesnet" "brits" "timemixer") 
datas=("ett" "weather" "metr_la" "ecoli")
nfolds=(0 1 2 3 4) 
missrates=(0.1 0.5)
misspatterns=("point" "time")
nsample=100
device="${DEVICE:-cuda:0}"
python_bin="${PYTHON_BIN:-python3}"

for model in "${models[@]}"; do
    for data in "${datas[@]}"; do
         for nfold in "${nfolds[@]}"; do
            for missrate in "${missrates[@]}"; do
                for misspattern in "${misspatterns[@]}"; do
                    output_file="logs/${model}_${data}_${nfold}.out"
                    echo "Running: $python_bin run_all.py --model \"$model\" --data \"$data\" --nsample $nsample --device \"$device\" --nfold $nfold --misspattern \"$misspattern\" --missrate $missrate > $output_file"
                    "$python_bin" run_all.py --model "$model" --data "$data" --nsample "$nsample" --device "$device" --nfold "$nfold" --misspattern "$misspattern" --missrate "$missrate" > "$output_file"
                done
            done
         done
    done
done
