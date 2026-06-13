# Neural Network & Deep Learning - Project 2

本项目是《神经网络与深度学习》课程的Project-2，目标是在 CIFAR-10 数据集上训练自定义的图像分类网络，并探究 Batch Normalization (BN) 在模型优化中的作用。

本项目通过手写模块化的 ResNet 架构，对比了不同的网络宽度、激活函数、优化器和正则化策略，以及在 VGG 网络上进行实验探究 BN层的优化效果，并进行了详细的可视化分析。

---

## 环境要求 (Requirements)

本项目代码基于 PyTorch 编写。推荐在 GPU (CUDA) 环境下运行以获得最佳的训练速度。

```bash
# 核心依赖库
python >= 3.8
torch >= 1.10.0
torchvision >= 0.11.0
matplotlib >= 3.4.0
numpy >= 1.20.0
tqdm >= 4.60.0
```

---

## 项目结构 (File Structure)

```text
.
├── README.md                    # 本说明文件
├── my_model/                    
│   ├── model.py                 # 自定义的 SimpleResBlock 模块与模型网络结构
│   ├── engine.py                # 包含数据加载器与训练/验证循环逻辑
│   ├── train.py                 # 自定义参数配置，训练并保存最优模型 
│   ├── run_experiments.py       # 四项对比实验代码
│   └── visualize.py             # 卷积核与特征提取的可视化分析
├── VGG_BatchNorm/
│   ├── data/                    # 数据集存放目录
│   ├── models/
│   │   ├── __init__.py
│   │   └── vgg.py               # 包含基础 VGG-A 与加入 BN 层的 VGG_BatchNorm
│   ├── utils/
│   │   ├── __init__.py
│   │   └── nn.py                           
│   └── VGG_Loss_Landscape.py    # Standard VGG-A 与 VGG_BatchNorm 的实验代码                         

```

---

## 运行指南 (How to Run)

### CIFAR-10 网络训练与对比实验

**1. 运行对比实验**

运行该脚本，会自动对比不同网络宽度、不同激活函数、不同优化器及正则化策略，并输出对比图表。
```bash
cd my_model
python run_experiments.py
```

**2. 训练特定参数配置的模型**

可修改主函数的参数配置，运行该脚本，将训练模型并保存模型权重，同时生成训练曲线图 (`training_curve.png`)。

```bash
python train.py
```

**3. 网络可视化**

运行该脚本，将读取训练好的模型，生成第一层 3x3 卷积核的权重图 (`insight_1_filters.png`)，以及网络浅层到深层（Stage 0 到 Stage 3）特征图（Feature Map）的演化过程 (`insight_2_feature_evolution.png`)。

```bash
python visualize.py
```

### Batch Normalization (BN) 分析实验

运行该脚本，将自动在 `[5e-4, 1e-4, 5e-5, 1e-5]` 等不同学习率下训练 Standard VGG-A 与 VGG_BatchNorm 模型。
代码会在训练过程中捕捉梯度变化，并进行可视化，输出对比分析图：

1. `performance_convergence_comparison.png`: 收敛速度对比。
2. `optimization_loss_landscape.png`: Loss Landscape 波动范围图。
3. `optimization_gradient_predictiveness.png`: 梯度预测性差异图 (对数坐标)。

```bash
cd VGG_BatchNorm
python VGG_Loss_Landscape.py
```
