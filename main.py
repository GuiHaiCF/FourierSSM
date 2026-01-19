import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import UnifiedTimeSeriesDataset
from utils.utils import evaluate_metrics
from tqdm import tqdm
import time
import os
import numpy as np
import signal
from FourierSSM.model.FourierSSM import FourierSSM  

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 固定随机种子（保证可重复性）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ================= 参数配置 =================
parser = argparse.ArgumentParser(description='FourierSSM for multivariate time series forecasting')

# 数据集参数
parser.add_argument('--data', type=str, default='exchange', help='data set')
parser.add_argument('--seq_len', type=int, default=96, help='input length')
parser.add_argument('--pre_len', type=int, default=96, help='predict length')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')

# 模型参数
parser.add_argument('--embed_size', type=int, default=128, help='embed dimensions')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden dimensions')
parser.add_argument('--feature_blocks', type=int, default=8, help='mul-scale feature ')
parser.add_argument('--window_size', type=int, default=32, help='window size for STFT')
parser.add_argument('--stride', type=int, default=16, help='stride for STFT')

# 优化参数
parser.add_argument('--epochs', type=int, default=50, help='train epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='optimizer learning rate')
parser.add_argument('--decay_step', type=int, default=5, help='Learning rate decay step')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Learning rate decay rate')
parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')

args = parser.parse_args()
print(f'Training configs: {args}')

def handle_signal(signum, frame):
    """信号处理函数：捕获中断信号触发异常"""
    raise KeyboardInterrupt(f"Received termination signal {signum}")

def main():
    # 注册信号处理（捕获Ctrl+C和kill信号）
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        # ================= 设备设置 =================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # ================= 数据准备 =================
        data_paths = {
            'electricity': 'data/electricity.csv',
            'solar': 'data/solar.csv',
            'metr': 'data/metr.csv',
            'PeMS07':'data/PeMS07.csv',
            'Flight':'data/Flight.csv',
            'exchange':'data/exchange_rate.csv',
            'weather':'data/weather.csv',
            'traffic': 'data/traffic.csv'
        }
        
        # 创建数据集实例
        train_set = UnifiedTimeSeriesDataset(data_paths[args.data], 'train', args.seq_len, args.pre_len, scale=True)
        val_set = UnifiedTimeSeriesDataset(data_paths[args.data], 'val', args.seq_len, args.pre_len, scaler=train_set.scaler)
        test_set = UnifiedTimeSeriesDataset(data_paths[args.data], 'test', args.seq_len, args.pre_len, scaler=train_set.scaler)

        # ================= 数据加载 =================
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        
        # ================= 模型初始化 =================
        model = FourierSSM(
            seq_len=args.seq_len,
            pre_len=args.pre_len,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            feature_size=train_set.get_feature_dim(),
            feature_blocks = args.feature_blocks, 
            window_size=args.window_size,
            stride=args.stride,
        ).to(device)

        #打印模型总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Model Info] Total Parameters: {total_params}\n")

        # ================= 优化器配置 =================
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.decay_step, 
            gamma=args.decay_rate
        )

        # ================= 断点续训配置 =================
        checkpoint_dir = os.path.join("output", args.data, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 
                                       f"checkpoint_{args.seq_len}_{args.pre_len}_{args.batch_size}_{args.hidden_size}_{args.embed_size}_{args.feature_blocks}_{args.window_size}.pth")
        best_model_path = os.path.join("output", 
                                       args.data, f"best_{args.data}_{args.seq_len}_{args.pre_len}_{args.batch_size}_{args.hidden_size}_{args.embed_size}_{args.feature_blocks}_{args.window_size}.pth")
        
        # 尝试加载检查点
        start_epoch = 0
        best_loss = float('inf')
        patience = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            patience = checkpoint['patience']
            print(f"\nLoaded checkpoint from epoch {start_epoch}")

        # ================= 训练循环 =================
        for epoch in range(start_epoch, args.epochs):
            try:
                model.train()
                train_loss = 0
                epoch_start = time.time()

                        # 创建带进度条的训练迭代器
                progress_bar = tqdm(enumerate(train_loader), 
                                    total=len(train_loader), 
                                    desc=f'Epoch {epoch+1}/{args.epochs}',
                                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
                # 训练批次迭代
                for batch_idx, (x, y) in progress_bar:
                    x = x.float().to(device, non_blocking=True)
                    y = y.float().to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    # 实时更新进度条信息
                    avg_loss = train_loss / (batch_idx + 1)  # 计算平均损失
                    progress_bar.set_postfix({
                        'batch_loss': f"{loss.item():.4f}",
                        'avg_loss': f"{avg_loss:.4f}"
                    })

                # 学习率调整
                if scheduler and (epoch+1) % args.decay_step == 0:
                    scheduler.step()

                # ================= 验证阶段 =================
                val_loss, val_mape, val_mae, val_rmse, val_smape = evaluate(model, val_loader, criterion, device)
                
                # 早停逻辑
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience += 1

                # 打印日志
                print(f'Epoch {epoch+1:03d} | Time: {time.time()-epoch_start:.1f}s | '
                      f'Train Loss: {train_loss/len(train_loader):.4f} | '
                      f'Val Loss: {val_loss:.4f} | SMAPE: {val_smape:.2%} | '
                      f'MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}')

                # 保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'patience': patience
                }
                torch.save(checkpoint, checkpoint_path)

                # 早停检查
                if args.early_stop and patience >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            except KeyboardInterrupt:
                print("\nTraining interrupted. Saving checkpoint...")
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'patience': patience
                }
                torch.save(checkpoint, checkpoint_path)
                break

        # ================= 最终测试 =================
        print("\n=== Starting Final Test ===")
        model.load_state_dict(torch.load(best_model_path))
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        test_loss, test_mape, test_mae, test_rmse, test_smape = evaluate(model, test_loader, criterion, device)
        
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f} | SMAPE: {test_smape:.4f}")
        print(f"MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")

        result_file = os.path.join("output", args.data, f"result_{args.data}.txt")
        f = open(result_file, 'a')
        f.write('=' * 80 + '\n')
        f.write(f'Experiment Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('seq_len:{}, pre_len:{}\n'.format(args.seq_len, args.pre_len))
        f.write('data:{}, batch_size:{}, embed_size:{}, hidden_size:{}, feature_blocks:{}\n'.format(args.data, args.batch_size, args.embed_size, args.hidden_size, args.feature_blocks))
        f.write('Test Loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4%}, SMAPE: {:.4f} '.format(test_loss, test_mae, test_rmse, test_mape, test_smape))
        f.write('\n')
        f.write('=' * 80 + '\n\n')
        f.close()


        
        # # 保存结果
        # with open("results_fourier.txt", "a") as f:
        #     f.write('seq_len:{}, pre_len:{}\n'.format(args.seq_len, args.pre_len))
        #     f.write(f'data:{args.data}, batch_size:{args.batch_size}, embed_size:{args.embed_size}, hidden_size:{args.hidden_size}, feature_blocks:{args.feature_blocks}, window_size:{args.window_size}\n')
        #     f.write(f'Test Loss: {test_loss:.4f}, MAPE: {test_mape:.4%}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}\n')
        #     f.write('\n')
        #     f.close()

    finally:
        # ================= 资源清理 =================
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        print("Resources cleaned up successfully")

def evaluate(model, loader, criterion, device):
    """模型评估函数"""
    model.eval()
    loss_total, preds, trues = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.float().to(device)
            output = model(x)
            loss = criterion(output, y)
            loss_total += loss.item()
            preds.append(output.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    # 指标计算
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mape, mae, rmse, smape = evaluate_metrics(trues, preds)
    return loss_total/len(loader), mape, mae, rmse, smape

if __name__ == '__main__':
    main()