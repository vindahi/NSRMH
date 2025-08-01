
#!/bin/bash
set -e
bits=(16 32 64 128)
noises=(0.4 0.8)

for i in ${bits[*]}; do
for j in ${noises[*]}; do
  CUDA_VISIBLE_DEVICES=0 python test.py --hash_dim $i \
                                                     --dataset flickr \
                                                     --noise_rate $j \
                                                     --epoch 150 \
                                                     --classes 24 \


done
done
