import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm

from models.vgg import VGG_A, VGG_BatchNorm
from data.loaders import get_cifar_loader

# 训练参数配置
num_workers = 4
batch_size = 128
epochs_n = 30 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")

train_loader = get_cifar_loader(root='./data/', train=True, batch_size=batch_size, num_workers=num_workers)
val_loader = get_cifar_loader(root='./data/', train=False, batch_size=batch_size, num_workers=num_workers)

def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total

def set_random_seeds(seed_value=2026, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=5, label="Model"):
    model.to(device)
    
    step_losses = []
    step_grad_diffs = []
    epoch_losses = []
    epoch_accuracies = []
    
    prev_gradients = None 
    
    for epoch in range(epochs_n):
        model.train()
        epoch_loss_sum = 0.0
        steps_in_epoch = 0
        
        for data in tqdm(train_loader, desc=f"{label} - Epoch {epoch+1}/{epochs_n}", leave=False):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            
            step_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            steps_in_epoch += 1
            
            # 计算梯度预测性
            current_grad_vector = []
            for p in model.parameters():
                if p.grad is not None:
                    current_grad_vector.append(p.grad.data.view(-1))
            
            if len(current_grad_vector) > 0:
                current_grad_tensor = torch.cat(current_grad_vector)
                if prev_gradients is not None:
                    # 如果网络坏死，这里会是0。
                    grad_diff = torch.norm(current_grad_tensor - prev_gradients, p=2).item()
                    step_grad_diffs.append(grad_diff)
                else:
                    step_grad_diffs.append(0.0)
                prev_gradients = current_grad_tensor.clone()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
        avg_epoch_loss = epoch_loss_sum / steps_in_epoch
        epoch_losses.append(avg_epoch_loss)
        
        val_acc = get_accuracy(model, val_loader, device)
        epoch_accuracies.append(val_acc)
        
        test_error = 100.0 - val_acc
        print(f"[{label}] Epoch {epoch+1:02d} | Loss: {avg_epoch_loss:.4f} | Best Test Error: {test_error:.2f}%")

    return step_losses, step_grad_diffs, epoch_losses, epoch_accuracies


# 滑动平均平滑函数
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)


# 实验 A：收敛轨迹对比
print("\n" + "="*60)
print("EXPERIMENT A: Convergence Analysis")
print("="*60)

set_random_seeds(device=device)
model_standard = VGG_A()
opt_standard = torch.optim.Adam(model_standard.parameters(), lr=5e-4)
_, _, std_losses, std_accs = train(model_standard, opt_standard, nn.CrossEntropyLoss(), train_loader, val_loader, epochs_n=epochs_n, label="Standard VGG")

set_random_seeds(device=device)
model_bn = VGG_BatchNorm()
opt_bn = torch.optim.Adam(model_bn.parameters(), lr=5e-4) 
_, _, bn_losses, bn_accs = train(model_bn, opt_bn, nn.CrossEntropyLoss(), train_loader, val_loader, epochs_n=epochs_n, label="VGG with BatchNorm")

epochs_range = range(1, epochs_n+1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, std_losses, label='Standard VGG-A', marker='o', linestyle='--')
plt.plot(epochs_range, bn_losses, label='VGG-A + BatchNorm', marker='s')
plt.title("Convergence Comparison: Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, alpha=0.5)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, std_accs, label='Standard VGG-A', marker='o', linestyle='--')
plt.plot(epochs_range, bn_accs, label='VGG-A + BatchNorm', marker='s')
plt.title("Convergence Comparison: Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True, alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('performance_convergence_comparison.png', dpi=200)
plt.close()

# 实验 B：多学习率下的优化地形（Loss Landscape）
print("\n" + "="*60)
print("EXPERIMENT B: Landscape & Gradient Analysis")
print("="*60)

learning_rates = [5e-4, 1e-4, 5e-5, 1e-5]  

def collect_landscape_data(model_class, model_name):
    all_loss_steps = []
    all_grad_steps = []
    
    for lr in learning_rates:
        print(f"--- Running: {model_name} [LR={lr}] ---")
        set_random_seeds(device=device)
        model = model_class()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        step_losses, step_grads, _, _ = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n, label=f"{model_name}")
        
        # 移除前几步的极端异常值
        step_losses = np.clip(step_losses, 0, 3.0) 
        
        all_loss_steps.append(step_losses)
        all_grad_steps.append(step_grads)
        
    all_loss_steps = np.array(all_loss_steps)
    all_grad_steps = np.array(all_grad_steps)
    
    min_loss_curve = np.min(all_loss_steps, axis=0)
    max_loss_curve = np.max(all_loss_steps, axis=0)
    
    min_grad_curve = np.min(all_grad_steps, axis=0)
    max_grad_curve = np.max(all_grad_steps, axis=0)

    # 平滑处理
    min_loss_curve = smooth_curve(min_loss_curve, factor=0.8)
    max_loss_curve = smooth_curve(max_loss_curve, factor=0.8)
    min_grad_curve = smooth_curve(min_grad_curve, factor=0.8)
    max_grad_curve = smooth_curve(max_grad_curve, factor=0.8)
    
    return min_loss_curve, max_loss_curve, min_grad_curve, max_grad_curve

min_loss_std, max_loss_std, min_grad_std, max_grad_std = collect_landscape_data(VGG_A, "Standard VGG")
min_loss_bn, max_loss_bn, min_grad_bn, max_grad_bn = collect_landscape_data(VGG_BatchNorm, "VGG with BatchNorm")

# 绘制 Loss Landscape
steps = range(len(min_loss_std))
plt.figure(figsize=(10, 6))
plt.fill_between(steps, min_loss_std, max_loss_std, color='green', alpha=0.3, label='Standard VGG-A')
plt.fill_between(steps, min_loss_bn, max_loss_bn, color='red', alpha=0.6, label='VGG-A + BatchNorm')
plt.title("Loss Landscape Smoothness Analysis", fontsize=14)
plt.xlabel("Training Steps")
plt.ylabel("Loss Variation Area")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('optimization_loss_landscape.png', dpi=200)
plt.close()

# 绘制 Gradient Predictiveness
plt.figure(figsize=(10, 6))
plt.fill_between(steps, min_grad_std, max_grad_std, color='blue', alpha=0.3, label='Standard VGG-A')
plt.fill_between(steps, min_grad_bn, max_grad_bn, color='orange', alpha=0.8, label='VGG-A + BatchNorm')
plt.yscale('log')
plt.title("Gradient Predictiveness Comparison (Log Scale)", fontsize=14)
plt.xlabel("Training Steps")
plt.ylabel("Step-to-step Gradient Difference L2 Norm")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('optimization_gradient_predictiveness.png', dpi=200)
plt.close()

print("\nAll tasks successfully executed.")