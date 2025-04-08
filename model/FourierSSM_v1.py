import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class ComplexRelu(nn.Module):
    """复数ReLU激活函数,分别对实部和虚部应用ReLU"""
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    
class STFTProcessor(nn.Module):
    """STFT处理模块(支持多尺度特征)"""
    def __init__(self, embed_size, hidden_size, feature_blocks,
                 window_size, stride):
        super().__init__()
        self.E = embed_size             # 嵌入维度
        self.N = hidden_size            # 隐藏层维度
        self.M = feature_blocks         # 特征划分块数
        self.W = window_size            # 窗口长度
        self.S = stride                 # 帧移
        self.F = self.W // 2 + 1        # 频率分量数
        self.E_m = self.E // self.M     # 子块特征数

        #动态参数生成网络
        self.mlp = nn.Sequential(
            nn.Linear(self.E_m, 128),           # 输入实部
            nn.GELU(),
            nn.Linear(128, 2 * self.E_m * self.E_m) # 输出实/虚参数
        )

        #全局复数对角矩阵
        self.A = nn.Parameter(torch.randn(hidden_size, dtype=torch.cfloat))

        #可迭代偏置参数
        self.biases = nn.Parameter(torch.randn(self.E_m, dtype=torch.cfloat))

    def stft_transform(self, x):
        """
        STFT变换
        输入: [B, T, E]
        输出: [B, T', F, E] (T'为时间帧数,F为频率分量)
        """

        B,T,E = x.shape

        # 分帧处理
        frames = x.unfold(1, self.W, self.S)        # [B, T', E, W]
        frames = frames.permute(0, 1, 3, 2)         # [B, T', W, E]
        # 加汉明窗
        window = torch.hann_window(self.W, device=x.device).view(1,1,self.W,1)
        windowed = frames * window
        # 快速傅里叶变换
        stft = fft.rfft(windowed, dim=2)            #[B, T', F, E]
        return stft
    
    def forward(self, x):
        """
        前向传播
        输入:[B, T, E]
        输出:[B, T, E]  (时域重建信号)
        """
        stft = self.stft_transform(x)               #[B, T', F, E]
        B, T_prime, Fre ,E = stft.shape

        #多尺度特征划分
        stft = stft.reshape(B*self.M, T_prime, Fre, self.E_m)   # [B*M, T', F, E_m]

        # 预计算频率响应分母项 (N, F), 从 0 到 F-1，归一化到 [0, 0.5]
        freqs = torch.arange(self.F, device=x.device)/self.W
        denom = 1 - self.A.unsqueeze(0) * torch.exp(-1j * 2 * torch.pi * freqs.unsqueeze(1))     #[F, N]
        denom = denom.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, F, N, 1, 1]

        X = stft
        
        #动态参数生成（压缩batch和频率维度）
        compressed = X.mean(dim=(0,2))                      # [T', E_m]
        params = self.mlp(compressed.real)                  # [T', 2*E_m*E_m]
        real, imag = params.chunk(2, dim=-1)                # 分割为：[T', E_m*E_m] [T', E_m*E_m]
        kernel = torch.complex(real, imag)                  # [T', E_m*E_m]

        U = kernel.view(T_prime, self.E_m, self.E_m)        # [T', E_m, E_m]
        U = U.unsqueeze(1).unsqueeze(2)                     # [T', 1, 1, E_m, E_m]            

        # 计算单频点响应
        terms = U / denom                                   # 广播后 [T', F, N, E_m, E_m]

        # 沿n维度求和
        G = terms.sum(dim=2)                                # [T', F, E_m, E_m]

        # 矩阵乘法
        X = torch.einsum('tfae,btfe->btfa', G, X)           # [B*M, T', F, E_m]

        #添加偏置项
        bias = self.biases.view(1,1,1,-1)                   # [1, 1, 1, E_m]
        X = X + bias                                        # [B*M, T', F, E_m]

        #非线性激活
        X = ComplexRelu()(X)                                # [B*M, T', F, E_m]

        # 多尺度融合与逆变换
        fused = X.reshape(B, T_prime, Fre, self.E)              # [B, T', F, E]
        istft = fft.irfft(fused, n=self.W, dim=2)               # [B, T', W, E]

        #重叠相加重建信号
        return self.overlap_add(istft)                          # [B, T, E]

    def overlap_add(self, frames):
        """重叠相加法重建时域信号"""
        B, T_prime, W, E = frames.shape
        output = torch.zeros(B, (T_prime-1)*self.S + W, E, device=frames.device)
        for t in range(T_prime):
            start = t * self.S
            output[:, start:start+W] += frames[:, t]
        return output

class DFTProcessor(nn.Module):
    """DFT处理模块(全局频域特征)"""
    def __init__(self, embed_size, hidden_size, feature_blocks):
        super().__init__()
        self.E = embed_size                     #嵌入维度
        self.N = hidden_size                    #隐藏层维度
        self.M = feature_blocks                 #特征划分块数

        self.E_m = self.E//self.M               # 子块特征数

        #可迭代偏置参数
        self.biases = nn.Parameter(torch.randn(self.E_m, dtype=torch.cfloat))

        self.A = nn.Parameter(torch.randn(hidden_size, dtype=torch.cfloat))             # [N]
        self.B = nn.Parameter(torch.randn(hidden_size , self.E_m, dtype=torch.cfloat))  # [N, E_m]
        self.C = nn.Parameter(torch.randn(self.E_m, hidden_size , dtype=torch.cfloat))  # [E_m, N]


    def dft_transform(self, x):
        """
        DFT变换
        输入:[B,T,E]
        输出:[B,F,E](F=⌈T/2⌉+1)
        """
        return fft.rfft(x, dim=1)  # 沿时间维度变换
    
    def forward(self, x):
        """
        前向传播
        输入: [B, T, E]
        输出: [B, T, E]
        """
        dft = self.dft_transform(x)     # [B,F,E]
        B, Fre, E = dft.shape

        #多尺度特征划分
        dft = dft.reshape(B*self.M, Fre, self.E_m)     # [B*M, F, E_m]

        X = dft

        # 计算频域响应矩阵G
        freqs = torch.linspace(0, 0.5, Fre, device=x.device)
        exp_term = torch.exp(-2j * torch.pi * freqs)            # [F]
        denominator = 1 - self.A.unsqueeze(1) * exp_term             # [N, F]
        inv_denominator = 1.0 / denominator                     # [N, F]

        # 向量化计算G矩阵 [F, E_m, E_m]
        G = torch.einsum('mn,nf,ne->mfe', self.C, inv_denominator, self.B)
        G = G.permute(1, 0, 2)  # [F, E_m, E_m]

        # 频域相乘
        X = torch.einsum('bfe,fae->bfa', X, G)   # [B*M, F, E_m]


        #添加偏置项
        bias = self.biases.view(1,1,-1)                     # [1, 1, E_m]
        X = X + bias                                        # [B*M, F, E_m]

        #非线性激活
        X = ComplexRelu()(X)                                # [B*M, F, E_m]

        # 多尺度融合与逆变换
        fused = X.reshape(B, Fre, self.E)                       # [B, F, E]

        # 逆变换
        return fft.irfft(fused, n=x.size(1), dim=1)             # [B, T, E]

class FourierSSM(nn.Module):
    """傅里叶状态空间模型"""
    def __init__(self, seq_len, pre_len, feature_size, embed_size, hidden_size, feature_blocks,
                 K=3, window_size=32, stride=16):
        super().__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.D = feature_size
        self.E = embed_size
        self.N = hidden_size
        self.M = feature_blocks
        self.K = K
        self.window_size = window_size
        self.stride = stride

        # 输入输出投影
        self.embed_in = nn.Linear(self.D, self.E)
        self.embed_out = nn.Linear(self.E, self.D)

        # 频域处理模块
        self.stft = STFTProcessor(embed_size, hidden_size, feature_blocks, 
                                  window_size, stride,)
        self.dft = DFTProcessor(embed_size, hidden_size, feature_blocks)

        # 可学习混合系数
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 初始权重0.3

        self.ffn = nn.Sequential(
            # 输入形状 [B, T*D]
            nn.Linear(self.seq_len * self.D, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.N),
            nn.Softplus(),
            nn.Linear(self.N, self.pre_len * self.D),
            nn.Unflatten(-1, (self.pre_len, self.D))
            # 恢复形状 [B, pre_len, D]
        )
    
    def forward(self,x):
        """
        输入: [B, T, D]
        输出: [B, pred_steps, D]
        """  
        # 嵌入投影
        x_emb = self.embed_in(x)  # [B, T, E]

        # 双分支处理
        stft_out = self.stft(x_emb)  # [B, T, E]
        dft_out = self.dft(x_emb)   # [B, T, E]

        # 自适应融合
        alpha = torch.sigmoid(self.alpha)
        output = alpha * stft_out + (1-alpha) * dft_out     # [B, T, E]

        # 输出投影与预测
        x_out = self.embed_out(output)              # [B, T, D]

        x_flat = x_out.reshape(x_out.size(0), -1)   # [B, T, D] -> [B, T*D]

        return self.ffn(x_flat)                     # [B, pre_len, D]         


        



