import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data.data_loader import UnifiedTimeSeriesDataset  

# 预定义数据集路径
DATA_PATHS = {
    'exchange': 'data/exchange_rate.csv',
    'ECG': 'data/ECG.csv',
    'electricity': 'data/electricity.csv',
    'solar': 'data/solar.csv',
    'metr': 'data/metr.csv',
    'PeMS07': 'data/PeMS07.csv',
    'Flight': 'data/Flight.csv',
    'weather': 'data/weather.csv',
    'traffic': 'data/traffic.csv'
}

# 固定随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# 噪声添加函数
def add_gaussian_noise(data, snr):
    """添加高斯白噪声到原始数据"""
    if snr == float('inf'):
        return data
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

# 带噪声的数据集类
class NoisyTimeSeriesDataset(UnifiedTimeSeriesDataset):
    def __init__(self, data_path, snr, *args, **kwargs):
        self.snr = snr
        super().__init__(data_path, *args, **kwargs)
    
    def _load_and_preprocess(self, data_path):
        """重写预处理流程添加噪声"""
        raw_df = pd.read_csv(data_path)
        raw_values = raw_df.iloc[:, 1:].values  # 去除时间戳列
        
        # 仅训练集添加噪声
        if self.flag == 'train' and self.snr != float('inf'):
            noisy_data = add_gaussian_noise(raw_values, self.snr)
        else:
            noisy_data = raw_values
        
        # 原始预处理流程
        processed_df = pd.DataFrame(noisy_data).interpolate(method='linear').ffill().bfill()
        zero_columns = processed_df.columns[(processed_df == 0).all()]
        return processed_df.drop(columns=zero_columns).values

# 模型训练评估流程
class NoiseRobustnessTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("Noise", exist_ok=True)
        
        # 确定最终数据路径
        if self.args.data_path:
            self.final_data_path = self.args.data_path
        else:
            self.final_data_path = DATA_PATHS.get(self.args.data, None)
            if self.final_data_path is None:
                raise ValueError(f"未知数据集名称: {self.args.data}")
        
    def _build_model(self, feature_dim):
        """创建模型实例"""
        from model.FourierSSM_v1 import FourierSSM  # 确保模型可导入
        return FourierSSM(
            seq_len=self.args.seq_len,
            pre_len=self.args.pre_len,
            embed_size=self.args.embed_size,
            hidden_size=self.args.hidden_size,
            feature_size=feature_dim,
            feature_blocks=self.args.feature_blocks,
            window_size=self.args.window_size,
            stride=self.args.stride
        ).to(self.device)
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        """训练单个epoch"""
        model.train()
        total_loss = 0
        for x, y in loader:
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def _evaluate(self, model, loader, criterion):
        """模型评估"""
        model.eval()
        loss_total, preds, trues = 0, [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.float().to(self.device), y.float().to(self.device)
                output = model(x)
                loss = criterion(output, y)
                loss_total += loss.item()
                preds.append(output.cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        rmse = np.sqrt(np.mean((preds - trues)**2))
        return loss_total/len(loader), rmse
    
    def run_experiment(self, snr_values):
        """运行完整测试流程"""
        results = []
        checkpoint_path = f"Noise/{self.args.data}_checkpoint.pth"
        results_csv = f"Noise/{self.args.data}_results.csv"
        best_model_path = f"Noise/{self.args.data}_best_model.pth"
        
        # 断点恢复
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            snr_values = checkpoint['remaining_snrs']
            results = checkpoint['results']
            print(f"从检查点恢复。剩余SNR: {snr_values}")
        
        for snr in snr_values.copy():
            print(f"\n=== 测试SNR: {snr} dB ({self.args.data}) ===")
            
            # 创建数据集
            train_set = NoisyTimeSeriesDataset(
                self.final_data_path, snr, 'train', 
                self.args.seq_len, self.args.pre_len, 
                scale=True
            )
            val_set = NoisyTimeSeriesDataset(
                self.final_data_path, snr, 'val',
                self.args.seq_len, self.args.pre_len,
                scaler=train_set.scaler
            )
            test_set = NoisyTimeSeriesDataset(
                self.final_data_path, snr, 'test',
                self.args.seq_len, self.args.pre_len,
                scaler=train_set.scaler
            )
            
            # 初始化模型
            model = self._build_model(train_set.get_feature_dim())
            criterion = nn.MSELoss().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            
            best_rmse = float('inf')
            patience = 0
            
            # 训练循环
            for epoch in range(self.args.epochs):
                train_loss = self._train_epoch(model, DataLoader(
                    train_set, batch_size=self.args.batch_size, shuffle=True), 
                    criterion, optimizer)
                
                val_loss, val_rmse = self._evaluate(model, DataLoader(
                    val_set, batch_size=self.args.batch_size), criterion)
                
                # 早停机制
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    patience = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        print(f"epoch {epoch} 早停")
                        break
                
                print(f"{self.args.data} SNR {snr}dB | Epoch {epoch+1}/{self.args.epochs} | "
                      f"训练损失: {train_loss:.4f} | 验证RMSE: {val_rmse:.4f}")
            
            # 最终测试
            model.load_state_dict(torch.load(best_model_path))
            _, test_rmse = self._evaluate(model, DataLoader(
                test_set, batch_size=self.args.batch_size), criterion)
            results.append({'SNR': snr, 'RMSE': test_rmse})
            
            # 更新检查点
            snr_values.remove(snr)
            torch.save({
                'remaining_snrs': snr_values,
                'results': results
            }, checkpoint_path)
            
            # 保存结果
            pd.DataFrame(results).to_csv(results_csv, index=False)
            self._plot_results()
        
        os.remove(checkpoint_path)  # 清理最终检查点
        return results
    
    def _plot_results(self):
        """绘制结果曲线"""
        df = pd.read_csv(f"Noise/{self.args.data}_results.csv")
        plt.figure(figsize=(10, 6))
        plt.plot(df['SNR'], df['RMSE'], 'bo-', markersize=8)
        
        # 设置混合坐标轴
        # plt.xscale('symlog', linthresh=4)
        plt.xticks([0,1,2,4,8,16,32,64,128], 
                   ['0','1','2','4','8','16','32','64','128','Clean'])
        
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title(f'{self.args.data} 噪声鲁棒性分析', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"Noise/{self.args.data}_noise_robustness_curve.png")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='exchange', 
                       help='data set')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pre_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--feature_blocks', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    args = parser.parse_args()
    
    # 配置测试参数
    snr_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, float('inf')]
    
    tester = NoiseRobustnessTester(args)
    results = tester.run_experiment(snr_values)
    print("\n最终结果:")
    print(pd.DataFrame(results))