# #!/bin/bash
# set -e
# METHOD='FSFH'
# bits=(16 32 64 128)
# mlpdrops=(0.001 0.002 0.003 0.004 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# drops=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

# for i in ${mlpdrops[*]}; do
# for j in ${drops[*]}; do
#   echo "**********Start ${METHOD} algorithm**********"
#   CUDA_VISIBLE_DEVICES=0 python test.py --hash_dim 16 \
#                                                      --dataset flickr \
#                                                      --noise_rate 0.4 \
#                                                      --epoch 150 \
#                                                      --classes 24 \
#                                                      --mlpdrop $i \
#                                                      --dropout $j \

#   echo "**********End ${METHOD} algorithm**********"
# done
# done



#!/bin/bash
set -e
METHOD='FSFH'
# bits=(16 32 64 128)
# noises=(0.4 0.8)
bits=(16)
noises=(0.4)
paramcen=(0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0 10 100 1000)

for i in ${bits[*]}; do
for j in ${noises[*]}; do
for k in ${paramcen[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python test.py --hash_dim $i \
                                                     --dataset flickr \
                                                     --noise_rate $j \
                                                     --epoch 150 \
                                                     --classes 24 \
                                                     --param_sup $k \


  echo "**********End ${METHOD} algorithm**********"
done
done
done