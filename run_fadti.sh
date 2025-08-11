#!/bin/bash

# nohup ./run_fadti.sh > run2.log 2>&1 & 
# python run.py --model "fadti" --ffttype "dft" --timetype "attn" --data "ett" --nsample 100 --nfold 0 --missrate 0.1 --misspattern "point" --device "cuda:6" --modelfolder "ftcsdi_ett_point_0.1_20250404_085422"

mkdir -p logs
ffttypes=("none" "dft" "stft" "frsst") # "none" "dft" 
timetypes=("attn" "conv")
datas=("ett" "weather" "yeast" "metr_la") # "ett" "weather" "yeast" "metr_la"
nfolds=(0 1)
missrates=(0.1 0.5)
misspatterns=("point" "time")
nsample=100
device="cuda:6"

for data in "${datas[@]}"; do
  for ffttype in "${ffttypes[@]}"; do
    for timetype in "${timetypes[@]}"; do
      for nfold in "${nfolds[@]}"; do
        for missrate in "${missrates[@]}"; do
          for misspattern in "${misspatterns[@]}"; do
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