import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP


class FourierSSM(nn.Module):
    def __init__(self, seq_len, pre_len, embed_size, hidden_size, feature_size, rank,
                 K=3, window_size=32, stride=16):
        """
        FourierSSM 核心模型架构
        seq_len: 输入序列长度
        pre_len: 预测序列长度
        embed_size: 嵌入维度
        hidden_size: 隐状态维度(A矩阵对角线元素数)
        feature_size: 输入特征维度
        K: 频域迭代次数
        window_size: STFT窗口长度
        stride: STFT帧移步长
        n_fft: FFT点数(默认等于窗口长度)
        """
        super().__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.E = embed_size
        self.D = feature_size
        self.H = hidden_size
        self.r = rank
        self.K = K
        self.window_size = window_size
        self.stride = stride
        self.n_fft = window_size 

        # 自动计算独立频域参数
        self.F = self.n_fft // 2 + 1                                # 频率分量数 F
        self.T_prime = (seq_len - window_size) //stride + 1         # 时间帧数 T'    

        self.alpha = nn.Parameter(torch.randn(self.T_prime, 1, 1, 1, dtype=torch.float))  

        #初始化 V_B,V_C (形状 E × r)
        self.V_B  = nn.Parameter(torch.randn(self.E, self.r, dtype=torch.cfloat),requires_grad=True)           
        self.V_C  = nn.Parameter(torch.randn(self.E, self.r, dtype=torch.cfloat),requires_grad=True)


        # 1. 输入嵌入层 (D -> E)
        self.embed_proj = nn.Linear(self.D, self.E)

        # 2.状态转移矩阵 (复数对角矩阵，H维向量存储对角线元素 )
        self.A = nn.Parameter(torch.randn(self.H, dtype=torch.cfloat))

        # 3.动态参数生成网络(MLP)
        self.param_net = nn.Sequential(
            # 输入: (B, T', 2*E*F)
            nn.Linear(2*self.E*self.F, 128),            
            nn.LayerNorm(128),
            nn.SiLU(),                                   
            nn.Linear(128, 64, bias=False),             
            nn.LayerNorm(64),
            nn.Linear(64, 4*self.H*self.r),  
            nn.Dropout(0.1)
            # 输出: (B, T',4*H*r)
        )


        # 4.复数偏置参数 (K次迭代独立)
        self.biases = nn.ParameterList([
            nn.Parameter(torch.randn(self.T_prime, self.E, self.F, dtype=torch.cfloat))             #(T', E, F)
            for _ in range(self.K)
        ])

        # 5.逆嵌入投影层（E -> D）
        self.inv_embed = nn.Linear(self.E, self.D)

        # 6.时间预测层(seq_len -> pre_len)
        self.ffn = nn.Sequential(
            # 输入形状 [B, T*D]
            nn.Linear(self.seq_len * self.D, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.H),
            nn.Softplus(),
            nn.Linear(self.H, self.pre_len * self.D),
            nn.Unflatten(-1, (self.pre_len, self.D))
            # 恢复形状 [B, pre_len, D]
        )

        # 参数校验
        self._validate_params()

    #生成动态参数B，C（复数）
    def _complex_param_generator(self, params):
        """将实数参数分解为复数B,C矩阵"""
        B_real, B_imag, C_real, C_imag = torch.split(
            params, 
            [self.H*self.r, self.H*self.r, self.H*self.r, self.H*self.r],
            dim=-1
        )
        B = torch.complex(
            B_real.view(-1, self.T_prime, self.H, self.r),
            B_imag.view(-1, self.T_prime, self.H, self.r)
        )
        C = torch.complex(
            C_real.view(-1, self.T_prime, self.H, self.r),
            C_imag.view(-1, self.T_prime, self.H, self.r)
        )
        return B, C

    def _validate_params(self):
        """参数校验"""
        assert self.n_fft >= self.window_size, "FFT点数不能小于窗口长度"
        assert self.stride <= self.window_size, "帧移不能超过窗口长度"

    def stft(self, x):
        """
        短时傅里叶变换

        x: 输入张量(B, T, E)
        return: STFT复数张量 (B, T', F, E)
        """

        B, T, E = x.shape
        x = x.permute(0, 2, 1)  # (B, E, T)
        x = x.reshape(-1, T)     # (B*E, T) 合并批次和嵌入维度
        
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.stride,
            win_length=self.window_size,
            window=torch.hamming_window(self.window_size, device=x.device),
            return_complex=True,
            center=False                        #禁止补零。 当信号长度不足以形成完整的时间帧时，STFT 会在末尾补零，以确保完整的窗口计算。
        )  # (B*E, F, T')
        
        stft = stft.reshape(B, E, self.F, -1)  # (B, E, F, T')
        return stft.permute(0, 3, 2, 1)        # (B, T', F, E)
    
    def istft(self, stft):
        """
        逆短时傅里叶变换

        stft: 复数张量(B, T', F, E)
        return: 时域信号(B, T, E)
        """
        B, T_prime, F, E = stft.shape
        stft = stft.permute(0, 3, 2, 1)  # (B, E, F, T')
        stft = stft.reshape(B*E, F, T_prime)  # (B*E, F, T')
        
        x = torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.stride,
            win_length=self.window_size,
            window=torch.hamming_window(self.window_size, device=stft.device),
            center=False
        )  # (B*E, T)
        
        x = x.reshape(B, E, -1)  # (B, E, T)
        return x.permute(0, 2, 1)  # (B, T, E)
    

    def forward(self, x):
        """
        前相传播

        x: 输入张量(B, T, D)
        return: 预测张量(B, T_pre, D)
        """

        #1.嵌入投影(B,T,D)->(B,T,E)
        x = self.embed_proj(x)

        #2.STFT转换（B, T, E）-> (B, T', F, E) 
        X_stft = self.stft(x)
        B, T_prime, Fre_n, E = X_stft.shape

        #3.动态参数生成（分解实部虚部并拼接）
        X_cat = torch.cat([X_stft.real, X_stft.imag], dim=-1)                                   # (B, T', F, 2E)
        X_cat = X_cat.reshape(B, self.T_prime, -1)                                              # (B, T', 2E*F) 

        params = self.param_net(X_cat)                                                          # (B, T', 4H*r)

        # 生成复数U_b,U_c矩阵
        U_b, U_c = self._complex_param_generator(params)                                        # (B, T', H, r) 复数
        # 计算V1⊗V2的Kronecker积
        V_kron = torch.kron(self.V_C, self.V_B)                                                 # 形状(E², r²)

    
        #4.频域迭代更新
        X = X_stft.permute(0,3,1,2)                                                             #(B, E, T', F )

        for k in range(self.K):
            residual = X                                                                        #(B, E, T', F )

            
            # 预计算频率响应分母项 (H, F), 从 0 到 F-1，归一化到 [0, 0.5]
            freq = torch.arange(self.F, device=x.device)/self.n_fft
            denom = 1 - self.A.unsqueeze(1) * torch.exp(-1j * 2 * torch.pi * freq)              #(H,F)       

            X_out = torch.zeros_like(X)                                                         #(B, E, T', F )      

            #使用分解式累加代替完整H矩阵
            for t in range(self.T_prime):
                # 初始化当前时间帧的累积量
                H_mix = torch.zeros(B, self.E, self.E, self.F, 
                                dtype=torch.cfloat, device=x.device)
                
                # 按时间帧逐帧计算
                for n in range(self.H):
                    # 计算当前时间帧的外积                                                         
                    outer = torch.einsum('bi,bj->bij', U_b[:, t, n, :], U_c[:, t, n, :])        # (B,r,r)

                    # 频率调制
                    modulated = outer.unsqueeze(-1) * (1 / denom[n]).view(1,1,1,-1)             # (B, r, r, F)
      
  
                    # 矩阵乘法优化
                    modulated_flat = modulated.reshape(B*self.F, self.r**2)                    # (B*F, r²)
                    projected = torch.matmul(modulated_flat, V_kron.T)                          # (B*F, E²)
                    projected = projected.reshape(B, self.F, self.E, self.E).permute(0,2,3,1)   # (B, E, E, F)

                    
                    # 直接累加到混合传递函数
                    H_mix += projected

                # 滑动平均公式，实时计算全局平均
                if t == 0:
                    H_avg = H_mix.detach().clone()
                else:
                    H_avg = (H_avg * t + H_mix) / (t + 1)

                # 混合传递函数
                H_mix = H_avg + self.alpha[t] * H_mix
                
                # 矩阵乘法
                X_t = torch.einsum('beef,bef->bef', H_mix, X[:, :, t])            #(B, E, F)
                
                X_out[:, :, t, :] = X_t

            #残差连接
            X = X_out + residual                                                                #(B, E, T', F )

            #添加偏置
            X = X + self.biases[k].permute(1,0,2).unsqueeze(0)                                  # (B, E, T', F)
            X = torch.complex(F.relu(X.real), F.relu(X.imag))


        #4.逆STFT(B, T', F, E) -> (B, T, E)
        x_out = self.istft(X.permute(0,2,3,1))

        # 5. 逆嵌入投影 (B, T, E) -> (B, T, D)
        x_out = self.inv_embed(x_out)

        #(B, T, D) -> (B, T*D)
        x_flat = x_out.reshape(x_out.size(0), -1)

        # 6. 时间预测 (B, T*D) -> (B, pre_len, D)
        return self.ffn(x_flat)

    


    









