#!/bin/bash


echo "=============================================="
echo "Training dataset: Exchange"
echo "=============================================="

for pre_len in 96 192 336 720
do
    echo "--------------------------------------------------"
    echo "Training with pre_len=$pre_len "
    echo "--------------------------------------------------"
    
    # 调用Python训练脚本并传递参数
    python main.py \
        --early_stop \
        --data "exchange" \
        --embed_size 128 \
        --hidden_size 512 \
        --seq_len 96 \
        --pre_len "$pre_len" \
        --batch_size 32 \
        --feature_blocks 8 \
        --window_size 8 \
        --stride 4 \
        --epochs 50 \
        --lr 1e-5 \
        --decay_step 5 \
        --decay_rate 0.5
    
    echo "Completed: Exchange (pre_len=$pre_len )"

done

echo "=============================================="
echo "Finished all combinations for Exchange "
echo "=============================================="
