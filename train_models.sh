#!/bin/bash

# 定义数据集列表
datasets=("exchange")

# 定义参数组合
embed_sizes=(8)
hidden_sizes=(512)
seq_lens=(96)
pre_lens=(96 192 336 720)

# 循环遍历每个数据集
for dataset in "${datasets[@]}"; do
    echo "=============================================="
    echo "Training dataset: $dataset"
    echo "=============================================="

    # 遍历所有embed_size和seq_len组合
    for embed_size in "${embed_sizes[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for seq_len in "${seq_lens[@]}"; do
                for pre_len in "${pre_lens[@]}"; do
                    echo "--------------------------------------------------"
                    echo "Training with embed_size=$embed_size, hidden_size=$hidden_size, seq_len=$seq_len, pre_len=$pre_len "
                    echo "--------------------------------------------------"
                    
                    # 调用Python训练脚本并传递参数
                    python main.py \
                        --early_stop \
                        --data "$dataset" \
                        --embed_size "$embed_size" \
                        --hidden_size "$hidden_size" \
                        --seq_len "$seq_len" \
                        --pre_len "$pre_len" \
                        --batch_size 2 \
                        --feature_blocks 8 \
                        --window_size 32 \
                        --stride 16 \
                        --epochs 50 \
                        --lr 1e-5 \
                        --decay_step 5 \
                        --decay_rate 0.5
                    
                    echo "Completed: $dataset (embed=$embed_size, hidden=$hidden_size, seq=$seq_len, pre_len=$pre_len )"
                done
            done
        done
    done

    echo "=============================================="
    echo "Finished all combinations for $dataset"
    echo "=============================================="
done

echo "All datasets and parameter combinations trained!"