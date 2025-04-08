#!/bin/bash


echo "=============================================="
echo "Training dataset: Flight"
echo "=============================================="

# for pre_len in 96 192 336
# do
#     echo "--------------------------------------------------"
#     echo "Training with pre_len=$pre_len "
#     echo "--------------------------------------------------"
    
#     # 调用Python训练脚本并传递参数
#     python main.py \
#         --early_stop \
#         --data "Flight" \
#         --embed_size 128 \
#         --hidden_size 512 \
#         --seq_len 96 \
#         --pre_len "$pre_len" \
#         --batch_size 2 \
#         --feature_blocks 16 \
#         --window_size 32 \
#         --stride 16 \
#         --epochs 50 \
#         --lr 1e-5 \
#         --decay_step 5 \
#         --decay_rate 0.5
    
#     echo "Completed: Flight (pre_len=$pre_len )"

# done

pre_len=720
echo "--------------------------------------------------"
echo "Training with pre_len=$pre_len "
echo "--------------------------------------------------"

# 调用Python训练脚本并传递参数
python main.py \
    --early_stop \
    --data "Flight" \
    --embed_size 128 \
    --hidden_size 512 \
    --seq_len 96 \
    --pre_len "$pre_len" \
    --batch_size 2 \
    --feature_blocks 8 \
    --window_size 16 \
    --stride 8 \
    --epochs 50 \
    --lr 1e-5 \
    --decay_step 5 \
    --decay_rate 0.5

echo "Completed: Flight (pre_len=$pre_len )"


