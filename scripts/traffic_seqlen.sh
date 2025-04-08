#!/bin/bash


echo "=============================================="
echo "Training dataset: Traffic"
echo "=============================================="

for seq_len in 48 144 192 336 504 720
do
    echo "--------------------------------------------------"
    echo "Training with seq_len=$seq_len "
    echo "--------------------------------------------------"
    
    # 调用Python训练脚本并传递参数
    python main.py \
        --early_stop \
        --data "traffic" \
        --embed_size 256 \
        --hidden_size 512 \
        --seq_len "$seq_len" \
        --pre_len 336 \
        --batch_size 2 \
        --feature_blocks 8 \
        --window_size 4 \
        --stride 2 \
        --epochs 50 \
        --lr 1e-5 \
        --decay_step 5 \
        --decay_rate 0.5
    
    echo "Completed: Traffic (seq_len=$seq_len )"

done

echo "=============================================="
echo "Finished all combinations for Traffic "
echo "=============================================="
