#!/bin/bash
#exec > output_dCNN_v3.txt 2>&1
export CUDA_VISIBLE_DEVICES=0

model_name=DoubleCNN

des='DoubleCNN-M'
use_norm_list=(1 0)
dCNN_use_norm_list=(true false)
dCNN_mode_list=('res' 'weight')

for use_norm in "${use_norm_list[@]}"; do
    for dCNN_use_norm in "${dCNN_use_norm_list[@]}"; do
        for dCNN_mode in "${dCNN_mode_list[@]}"; do
            model_id='$weather_use_norm_${use_norm}_dCNN_mode_${dCNN_mode}_dCNN_use_norm_${dCNN_use_norm}'
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/weather/ \
                --data_path weather.csv \
                --model_id "${model_id}" \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len 96 \
                --enc_in 21 \
                --c_out 21 \
                --des $des \
                --batch_size 16 \
                --itr 1 \
                --use_norm $use_norm \
                --dCNN_mode $dCNN_mode \
                --dCNN_use_norm $dCNN_use_norm
        done
    done
done