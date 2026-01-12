#!/usr/bin/env python3
# coding: utf-8

"""
电力系统机组组合预测模型（Transformer版本）
基于原始工作修改：Arun Ramesh, 休斯顿大学
修改内容：改用Transformer架构实现时序特征提取
"""

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from timeit import default_timer as timer
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os  # 新增os模块用于路径处理

# 设置随机种子保证可重复性
np.random.seed(1)  # NumPy随机种子
torch.manual_seed(1)  # PyTorch随机种子

# ================== 数据加载与预处理 ==================
# 读取24节点电力系统的负荷需求数据（输入）和机组组合数据（输出）
dfX_24 = pd.read_csv("demand24BusWBCorr24Prd.txt")  # 负荷需求数据
dfY_24 = pd.read_csv("commitment24BusWBCorr24Prd.txt")  # 机组启停状态

# 数据标准化：将输入数据缩放到[0,1]范围
x = dfX_24.to_numpy() / 100  # 最大负荷为100MW[7](@ref)
y = dfY_24.to_numpy()  # 输出为二进制状态（0/1）

# 数据洗牌：打乱样本顺序以避免时序依赖性
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# 数据集划分：80%训练集，20%测试集
split_at = len(x) - len(x) // 5
(x_train, x_test) = x[:split_at], x[split_at:]
(y_train, y_test) = y[:split_at], y[split_at:]

# 转换为PyTorch张量
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.FloatTensor(y_test)

# 创建数据加载器（支持批量训练）
batch_size = 64  # 根据GPU显存调整[8](@ref)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ================== Transformer模型定义 ==================
class PositionalEncoding(nn.Module):
    """位置编码模块，为输入序列添加时序信息

    参数：
        d_model: 嵌入维度
        max_len: 最大序列长度

    实现参考：《Attention Is All You Need》论文
    """

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用余弦
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """前向传播：添加位置编码

        输入形状：[batch_size, features]
        输出形状：[batch_size, 1, d_model]
        """
        x = x.unsqueeze(1)  # 添加序列维度
        x = x + self.pe[:, :x.size(1), :]
        return x


class PowerSystemTransformer(nn.Module):
    """基于Transformer的电力系统机组组合预测模型

    参数：
        input_dim: 输入特征维度
        output_dim: 输出维度（机组数量）
        d_model: 嵌入维度（建议64-256）
        nhead: 注意力头数（需能被d_model整除）
        num_encoder_layers: Transformer编码器层数
        dim_feedforward: 前馈网络维度
        dropout: 随机失活率
    """

    def __init__(self, input_dim, output_dim, d_model=128, nhead=8,
                 num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()  # 将输出映射到[0,1]

    def forward(self, src):
        """前向传播过程

        输入形状：[batch_size, input_dim]
        输出形状：[batch_size, output_dim]
        """
        # 嵌入和位置编码
        src = self.input_embedding(src)  # [batch_size, d_model]
        src = self.pos_encoder(src)  # [batch_size, 1, d_model]

        # Transformer处理
        src = src.permute(1, 0, 2)  # 调整为[seq_len, batch_size, d_model]
        output = self.transformer_encoder(src)  # 编码器处理

        # 输出处理
        output = output[0]  # 取序列第一个位置输出
        output = self.output_layer(output)
        return self.sigmoid(output)


# ================== 模型训练配置 ==================
# 模型初始化
input_dim = x_train.shape[1]  # 输入特征维度
output_dim = y_train.shape[1]  # 输出机组数量
# 修改模型参数减少计算量
model = PowerSystemTransformer(
    input_dim, 
    output_dim, 
    d_model=64,  # 减少嵌入维度
    nhead=4,     # 减少注意力头数
    num_encoder_layers=2,  # 减少编码器层数
    dim_feedforward=128    # 减小前馈网络维度
)

# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练参数
num_epochs = 50  # 完整训练轮次
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
#device = torch.device("cuda")
device = torch.device("cpu")
model.to(device)  # 将模型移至指定设备

# ================== 训练循环 ==================
print(f"训练设备: {device}")
train_losses = []
start_time = timer()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        # 数据移至设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 记录损失
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 每5轮输出训练进度
    if (epoch + 1) % 5 == 0:
        print(f'轮次 [{epoch + 1}/{num_epochs}], 损失: {avg_loss:.4f}')

# 训练计时
training_time = timer() - start_time
print(f"总训练时间: {training_time:.2f}秒")


# ================== 模型评估 ==================
def evaluate_model(model, x_tensor, y_tensor):
    """模型评估函数

    参数：
        model: 训练好的模型
        x_tensor: 输入张量
        y_tensor: 真实标签

    返回：
        accuracy: 准确率（百分比）
        predictions: 预测结果
    """
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor.to(device))
        preds = (outputs >= 0.5).float()
        acc = (preds == y_tensor.to(device)).float().mean().item() * 100
    return acc, preds.cpu().numpy()


# 训练集评估
train_acc, train_preds = evaluate_model(model, x_train_tensor, y_train_tensor)

# 测试集评估
test_acc, test_preds = evaluate_model(model, x_test_tensor, y_test_tensor)

print(f"训练集准确率: {train_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%")


# ================== 结果保存 ==================
# 保存预测结果（二进制状态和概率值）
# 结果保存路径处理（适配Linux）
def save_results(filename, data):
    """保存结果到CSV文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

# 在保存图片前添加目录创建
os.makedirs("results", exist_ok=True)  # 新增这行
# 图片保存路径处理
plt.savefig(os.path.join("results", "training_loss_{}_{}.png".format(device,batch_size)))

# 模型保存路径处理
os.makedirs("models", exist_ok=True)  # 新增这行
torch.save(model.state_dict(), os.path.join("models", "power_system_transformer_model.pth"))


# ================== 模型加载函数 ==================
def load_pretrained_model(model_path, input_dim, output_dim):
    """加载预训练模型

    参数：
        model_path: 模型权重路径
        input_dim: 必须与训练时一致
        output_dim: 必须与训练时一致
    """
    model = PowerSystemTransformer(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


print("训练完成！模型和预测结果已保存。")
# 在模型初始化后添加
torch.backends.quantized.engine = 'qnnpack'  # 启用量化
