#!/bin/bash
mkdir -p logs
# 示例参数数组，可根据需要修改
methods=("ftcsdi" "ftcsdi_frsst" "ftcsdi_fsst")
# ("csdi_ori" "csdi" "ftcsdi" "saits")
datas=("ett")
nfolds=(0)
# nfolds=(0 1 2 3 4)
missrates=(0.1 0.5)
misspatterns=("point" "time")
# nsample 和 device 可以直接定义
nsample=100
device="cuda:6"

# 外层循环遍历所有参数组合
for model in "${models[@]}"; do
    for data in "${datas[@]}"; do
         for nfold in "${nfolds[@]}"; do
            for missrate in "${missrates[@]}"; do
                for misspattern in "${misspatterns[@]}"; do
                    # 构造输出文件名，可以根据需求修改命名规则
                    output_file="logs/${model}_${data}_${nfold}.out"
                    echo "Running: python run.py --model \"$model\" --data \"$data\" --nsample $nsample --device \"$device\" --nfold $nfold --misspattern \"$misspattern\" --missrate $missrate > $output_file"
                    python run.py --model "ftcsdi" --data "$data" --nsample $nsample --device "$device" --nfold $nfold --misspattern "$misspattern" --missrate $missrate > "$output_file"
                done
            done
         done
    done
done
