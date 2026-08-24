#!/bin/bash

mkdir -p logs
ffttypes=("none" "dft" "stft" "frsst") # "none" "dft" 
timetypes=("attn" "conv")
datas=("ett" "weather" "yeast" "metr_la")
nfolds=(0 1)
missrates=(0.1 0.5)
misspatterns=("point" "time")
nsample=100
device="${DEVICE:-cuda:0}"
python_bin="${PYTHON_BIN:-python3}"

for data in "${datas[@]}"; do
  for ffttype in "${ffttypes[@]}"; do
    for timetype in "${timetypes[@]}"; do
      for nfold in "${nfolds[@]}"; do
        for missrate in "${missrates[@]}"; do
          for misspattern in "${misspatterns[@]}"; do
            output_file="logs/fadti_${ffttype}_${timetype}_${data}_fold${nfold}_${misspattern}_mr${missrate}.out"
            echo "Running: $python_bin run.py \
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

            "$python_bin" run.py \
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
