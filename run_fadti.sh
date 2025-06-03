#!/bin/bash

# nohup ./run_fadti.sh > run.log 2>&1 & 
# python run.py --model "fadti" --ffttype "dft" --timetype "attn" --data "ett" --nsample 100 --nfold 0 --missrate 0.1 --misspattern "point" --device "cuda:6" --modelfolder "ftcsdi_ett_point_0.1_20250404_085422"

mkdir -p logs
# 示例参数数组，可根据需要修改
# ("csdi_ori" "csdi" "ftcsdi" "saits")
ffttypes=("dft" "stft" "frsst")
timetypes=("attn" "conv")
# timetypes=("attn")
datas=("ett" "weather")
# ett weather
nfolds=(0 1 2 3 4)
missrates=(0.1 0.5)
misspatterns=("point" "time")
# nsample 和 device 可以直接定义
nsample=100
device="cuda:6"

# 外层循环遍历所有参数组合
for data in "${datas[@]}"; do
  for ffttype in "${ffttypes[@]}"; do
    for timetype in "${timetypes[@]}"; do
      for nfold in "${nfolds[@]}"; do
        for missrate in "${missrates[@]}"; do
          for misspattern in "${misspatterns[@]}"; do

            # 构造日志文件名
            output_file="logs/fadti_${ffttype}_${timetype}_${data}_fold${nfold}_${misspattern}_mr${missrate}.out"

            echo "Running: python run.py \
--model fadti \
--ffttype ${ffttype} \
--timetype ${timetype} \
--data ${data} \
--nsample ${nsample} \
--device ${device} \
--nfold ${nfold} \
--misspattern ${misspattern} \
--missrate ${missrate} \
> ${output_file}"

            # 真正执行
            python run.py \
              --model "fadti" \
              --ffttype "${ffttype}" \
              --timetype "${timetype}" \
              --data "${data}" \
              --nsample "${nsample}" \
              --device "${device}" \
              --nfold "${nfold}" \
              --misspattern "${misspattern}" \
              --missrate "${missrate}" \
              > "${output_file}"

          done
        done
      done
    done
  done
done