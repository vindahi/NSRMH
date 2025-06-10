#!/bin/bash
set -e
METHOD='FSFH'
bits=(64 128)
noises=(0.0 0.4 0.8)

for i in ${bits[*]}; do
for j in ${noises[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python main.py --hash_dim $i \
                                                     --dataset wiki \
                                                     --noise_rate $j \
                                                     --epoch 200 \
                                                     --classes 10 \
                                                     --image_dim 128 \
                                                     --text_dim 10 \

  echo "**********End ${METHOD} algorithm**********"
done
done