import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.data_loader import UnifiedTimeSeriesDataset
from utils.utils import evaluate_metrics
from model.FourierSSM_v1 import FourierSSM

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ================= 参数配置 =================
parser = argparse.ArgumentParser(description='FourierSSM Noise Robustness Test')
parser.add_argument('--data', type=str, default='traffic', help='Dataset name')
parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
parser.add_argument('--pre_len', type=int, default=96, help='Prediction length')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--embed_size', type=int, default=256, help='Embedding dimension')
parser.add_argument('--hidden_size', type=int, default=512, help='Hidden dimension')
parser.add_argument('--feature_blocks', type=int, default=8, help='Feature blocks')
parser.add_argument('--window_size', type=int, default=4, help='STFT window size')
parser.add_argument('--stride', type=int, default=2, help='STFT stride')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型路径生成 =================
def get_model_path():
    """生成标准化的模型存储路径"""
    model_name = f"best_{args.data}_{args.seq_len}_{args.pre_len}_" \
                f"{args.batch_size}_{args.hidden_size}_{args.embed_size}_" \
                f"{args.feature_blocks}_{args.window_size}.pth"
    return os.path.join("output", args.data, model_name)

# ================= 数据加载 =================
def load_test_data():
    data_map = {
        'solar': 'data/solar.csv',
        'traffic': 'data/traffic.csv',
        'electricity': 'data/electricity.csv',
        'exchange': 'data/exchange_rate.csv',
        'metr': 'data/metr.csv',
        'PeMS07': 'data/PeMS07.csv',
        'Flight': 'data/Flight.csv',
        'weather': 'data/weather.csv'
    }
    
    # 加载归一化器
    train_set = UnifiedTimeSeriesDataset(
        data_map[args.data], 
        'train', 
        args.seq_len, 
        args.pre_len, 
        scale=True
    )
    
    # 创建测试集
    test_set = UnifiedTimeSeriesDataset(
        data_map[args.data],
        'test',
        args.seq_len,
        args.pre_len,
        scaler=train_set.scaler,
        scale=True
    )
    return DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# ================= 噪声处理 =================
def add_gaussian_noise(x, snr_db):
    signal_power = torch.mean(x**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return x + torch.randn_like(x) * torch.sqrt(noise_power)

# ================= 评估函数 =================
def noise_evaluation(model, loader):
    model.eval()
    # snr_levels = np.arange(100, -1, -10)  # 100dB到0dB
    snr_levels = [100, 64, 32, 16, 8, 4, 2, 1, 0]
    results = {}
    
    for snr in tqdm(snr_levels, desc="Testing SNR levels"):
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x_noisy = add_gaussian_noise(x, snr).float().to(device)
                y = y.float().to(device)
                output = model(x_noisy)
                
                preds.append(output.cpu().numpy())
                trues.append(y.cpu().numpy())
        
        # 计算指标
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, rmse = evaluate_metrics(trues, preds)
        results[snr] = rmse
        print(f"SNR: {snr:3d} dB | RMSE: {rmse:.4f}")
    
    return results

# ================= 主流程 =================
def main():
    # 自动定位模型
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at:\n{model_path}\n"
                               "Check if training parameters match!")
    
    # 初始化模型
    test_loader = load_test_data()
    model = FourierSSM(
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        feature_size=test_loader.dataset.full_data.shape[1],
        feature_blocks=args.feature_blocks,
        window_size=args.window_size,
        stride=args.stride
    ).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded pretrained model from {model_path}")

    # 执行测试
    results = noise_evaluation(model, test_loader)

    # ================= 保存结果 =================
    output_dir = os.path.join("noise_robustness", args.data)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存可视化结果
    plt.figure(figsize=(10, 6))
    snr_values = sorted(results.keys(), reverse=True)
    rmse_values = [results[snr] for snr in snr_values]
    
    plt.plot(snr_values, rmse_values, 'g-o', linewidth=2, markersize=8)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title(f"Noise Robustness: {args.data.upper()}", fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 101, 10))
    plt.savefig(os.path.join(output_dir, f"{args.data}_{args.pre_len}_{args.window_size}_robustness.pdf"), bbox_inches='tight')
    
    # 保存CSV数据
    csv_path = os.path.join(output_dir, f"{args.data}_{args.pre_len}_{args.window_size}.csv")
    np.savetxt(csv_path,
               np.column_stack([snr_values, rmse_values]),
               delimiter=',',
               header='SNR(dB),RMSE',
               fmt='%.4f',
               comments='')
    
    print(f"\nResults saved to:\n{output_dir}")
    print(f" - Visualization: {args.data}_{args.pre_len}_{args.window_size}_robustness.pdf")
    print(f" - Raw data: {args.data}_{args.pre_len}_{args.window_size}.csv")

if __name__ == "__main__":
    main()