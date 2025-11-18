# 电力系统机组组合预测模型（Transformer版本）

## 项目概述

本项目实现了基于Transformer架构的电力系统机组组合预测模型，用于预测24节点电力系统中各机组的启停状态。该模型通过学习历史负荷需求与机组组合之间的关系，实现对未来机组启停状态的准确预测，为电力系统调度提供决策支持。

## 主要功能

- 使用Transformer架构进行时序特征提取和模式识别
- 支持24节点电力系统的机组组合预测
- 完整的数据加载、预处理、模型训练、评估和结果保存流程
- 训练损失可视化和模型性能评估
- 支持模型保存和加载功能

## 项目结构

```
Load/
├── LogRegressionDetailed_24Bus_24Period00.py  # 主要程序文件（Transformer版本）
├── demand24BusWBCorr24Prd.txt                  # 负荷需求数据（输入）
├── commitment24BusWBCorr24Prd.txt              # 机组组合数据（输出）
├── commitment24Bus24PrdTestSample_Transformer.csv  # 预测的机组组合结果
├── probabilities24Bus24PrdTestSample_Transformer.csv  # 预测的概率值
├── power_system_transformer_model.pth          # 训练好的模型权重
├── training_loss_cpu_64.png                    # 训练损失曲线
└── README.md                                   # 项目说明文档
```

## 技术实现

### 模型架构

模型基于Transformer架构实现，主要包含以下组件：

- **输入嵌入层**：将输入特征映射到高维空间
- **位置编码**：为输入添加时序信息
- **Transformer编码器**：包含多头自注意力机制和前馈神经网络
- **输出层**：将特征映射到机组启停状态概率

### 数据处理

- 输入数据：负荷需求数据，范围归一化到[0,1]
- 输出数据：机组启停状态（二进制值：0/1）
- 数据划分：80%训练集，20%测试集
- 批量训练：支持mini-batch训练，提高训练效率

### 训练配置

- 损失函数：二分类交叉熵损失（BCELoss）
- 优化器：Adam优化器
- 评估指标：准确率（Accuracy）
- 训练设备：支持CPU（当前配置）和GPU

## 使用方法

### 环境要求

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

### 运行模型

1. 确保所有数据文件在同一目录下
2. 运行主程序文件：

```bash
python LogRegressionDetailed_24Bus_24Period00.py
```

3. 程序将自动执行以下步骤：
   - 加载和预处理数据
   - 初始化并训练模型
   - 评估模型性能
   - 保存预测结果和模型权重

### 结果说明

- `commitment24Bus24PrdTestSample_Transformer.csv`：预测的机组启停状态（二进制）
- `probabilities24Bus24PrdTestSample_Transformer.csv`：预测的概率值
- `power_system_transformer_model.pth`：训练好的模型权重
- `training_loss_cpu_64.png`：训练损失曲线

## 模型评估

模型使用准确率作为评估指标，训练完成后会输出：
- 训练集准确率
- 测试集准确率
- 总训练时间

## 注意事项

- 当前模型配置为CPU训练，如需使用GPU训练，请修改代码中的设备设置
- 模型参数（如嵌入维度、注意力头数、编码器层数等）可根据实际需求调整
- 数据文件较大，请确保有足够的内存空间

## 参考资料

- 基于原始工作修改：Arun Ramesh, 休斯顿大学
- Transformer架构参考：《Attention Is All You Need》论文

---

*注：本项目为电力系统机组组合预测的Transformer实现版本，用于学术研究和实验目的。*
