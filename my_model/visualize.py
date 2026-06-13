# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from model import ModularResNet
# from engine import get_dataloaders

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def visualize_insights(model_path):
#     print("Loading best model for visualization...")
#     model = ModularResNet(filters=[64, 128, 256]).to(DEVICE)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     except Exception as e:
#         print(f"Error loading weights.")
#         return

#     model.eval()

#     # 第一层卷积核滤波器可视化 (Filters)
#     weights = model.conv1.weight.data.cpu().numpy()
#     # 归一化以用于图像显示
#     w_min, w_max = weights.min(), weights.max()
#     weights = (weights - w_min) / (w_max - w_min)
    
#     plt.figure(figsize=(8, 8))
#     # 抽取前 36 个滤波器进行展示
#     for i in range(min(36, weights.shape[0])):
#         plt.subplot(6, 6, i+1)
#         img = np.transpose(weights[i], (1, 2, 0)) # PyTorch格式转为Matplotlib格式
#         plt.imshow(img)
#         plt.axis('off')
#     plt.suptitle("Insight: 1st Layer Convolutional Filters", fontsize=16)
#     plt.tight_layout()
#     plt.savefig('insight_filters.png', dpi=150)
#     print("Saved 'insight_filters.png'.")

#     # 网络特征图解释 (Feature Maps)
#     _, testloader = get_dataloaders(batch_size=1)
#     dataiter = iter(testloader)
    
#     # 随便取一张图片进行可视化
#     for _ in range(5):
#         images, labels = next(dataiter)
#     img = images.to(DEVICE)

#     with torch.no_grad():
#         # 获取第一层卷积激活后的输出
#         out = model.act(model.bn1(model.conv1(img)))
        
#     feature_maps = out.squeeze(0).cpu().numpy()
    
#     plt.figure(figsize=(14, 4))
    
#     # 画原始输入图片 (需反归一化)
#     plt.subplot(1, 6, 1)
#     raw_img = images[0].numpy()
#     raw_img = np.transpose(raw_img, (1, 2, 0))
#     mean, std = np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])
#     raw_img = std * raw_img + mean
#     raw_img = np.clip(raw_img, 0, 1) # 限制到 [0, 1] 范围内
    
#     plt.imshow(raw_img)
#     plt.title(f"Original Input\nLabel: {labels[0].item()}")
#     plt.axis('off')
    
#     # 画前 5 个通道的特征图激活状态
#     for i in range(5):
#         plt.subplot(1, 6, i+2)
#         # 用颜色映射展示特征强度
#         plt.imshow(feature_maps[i], cmap='magma')
#         plt.title(f"Feature Channel {i}")
#         plt.axis('off')
        
#     plt.suptitle("Insight: Internal Feature Representation (Network Interpretation)", fontsize=16)
#     plt.tight_layout()
#     plt.savefig('insight_feature_maps.png', dpi=150)
#     print("Saved 'insight_feature_maps.png'.")

# if __name__ == '__main__':
    
#     visualize_insights("best_model.pth")



import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model import ModularResNet
from engine import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CIFAR-10 类别名称
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 标准化张量的反归一化函数
def unnormalize(img_tensor):
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_insights(model_path):
    print("Loading best model for visualization...")
    
    model = ModularResNet(filters=[64, 128, 256]).to(DEVICE)
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Please train the model first.")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # 第一层卷积核滤波器可视化 (Visualization of Filters)
    print("Generating Filter Visualization...")
    weights = model.conv1.weight.data.cpu().numpy()
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min) # 归一化到 [0, 1] 范围
    
    plt.figure(figsize=(8, 8))
    # 抽取前 36 个滤波器进行展示
    for i in range(min(36, weights.shape[0])):
        plt.subplot(6, 6, i+1)
        img = np.transpose(weights[i], (1, 2, 0)) 
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle("1st Layer 3x3 Convolutional Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig('insight_1_filters.png', dpi=150)
    plt.close()

    # 网络深度特征演化解释 (Network Interpretation)
    print("Generating Feature Map Evolution...")
    _, testloader = get_dataloaders(batch_size=1)
    dataiter = iter(testloader)
    
    # 抽取一张图片
    for _ in range(7):
        images, labels = next(dataiter)
    img = images.to(DEVICE)
    label_name = CLASSES[labels[0].item()]

    # 提取网络在不同深度的中间层特征
    features = {}
    with torch.no_grad():
        # Block 0: Initial Conv (32x32)
        x0 = model.act(model.bn1(model.conv1(img)))
        features['Stage 0 (Initial Conv)'] = x0
        
        # Block 1: Layer 1 (32x32)
        x1 = model.layer1(x0)
        features['Stage 1 (ResBlock 1)'] = x1
        
        # Block 2: Layer 2 (16x16)
        x2 = model.layer2(x1)
        features['Stage 2 (ResBlock 2)'] = x2
        
        # Block 3: Layer 3 (8x8)
        x3 = model.layer3(x2)
        features['Stage 3 (ResBlock 3)'] = x3

    # 绘制特征演化图
    plt.figure(figsize=(16, 6))
    
    # 绘制原始输入图片
    plt.subplot(2, 5, 1)
    plt.imshow(unnormalize(images[0]))
    plt.title(f"Input Image\nLabel: [{label_name}]", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 依次绘制各深度的特征图
    plot_idx = 2
    for stage_name, fmap_tensor in features.items():
        fmap = fmap_tensor.squeeze(0).cpu().numpy()
        
        # 绘制该层所有通道的平均激活热力图(代表网络的注意力中心)
        avg_fmap = np.mean(fmap, axis=0)
        plt.subplot(2, 5, plot_idx)
        plt.imshow(avg_fmap, cmap='viridis')
        plt.title(f"{stage_name}\nAverage Attention", fontsize=10)
        plt.axis('off')
        
        # 绘制该层激活强度最大的单一通道(代表最突出的特征提取)
        max_channel_idx = np.argmax(np.max(fmap, axis=(1,2)))
        plt.subplot(2, 5, plot_idx + 5) # 放在下一排对应的位置
        plt.imshow(fmap[max_channel_idx], cmap='magma')
        plt.title(f"Max Channel: {max_channel_idx}", fontsize=10)
        plt.axis('off')
        
        plot_idx += 1

    plt.suptitle("Feature Map Evolution", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('insight_2_feature_evolution.png', dpi=200)
    plt.close()
    
    print("Visualization complete!")

if __name__ == '__main__':
    
    visualize_insights("best_model.pth")