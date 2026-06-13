import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 导入网络架构和自定义函数
from model import ModularResNet, count_parameters
from engine import get_dataloaders

# 训练过程可视化函数
def plot_training_process(train_losses, test_errors, config, best_error):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    # 绘制 Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', marker='o', markersize=3, label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
     # 绘制 Test Error
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_errors, 'r-', marker='s', markersize=3, label='Test Error (%)')
    plt.axhline(y=best_error, color='g', linestyle='--', label=f'Best Error: {best_error:.2f}%')
    plt.title('Test Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Error (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    title_str = (f"Model | Opt: {config['optimizer'].upper()} | "
                 f"Base LR: {config['lr']} | Act: {config['activation'].upper()} | "
                 f"Size: {config['model_size'].capitalize()}")
    plt.suptitle(title_str, fontsize=14)
    plt.tight_layout()
    
    save_path = 'training_curve.png'
    plt.savefig(save_path, dpi=200)
    print(f"\nTraining Curve Saved to: {save_path}")
    plt.close()

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 准备数据集
    trainloader, testloader = get_dataloaders(config['batch_size'])
    
    # 定义网络架构
    if config['model_size'] == 'small':
        filters = [32, 64, 128]
    elif config['model_size'] == 'wide':
        filters = [128, 256, 512]
    else:
        filters = [64, 128, 256]
        
    # 初始化模型
    model = ModularResNet(
        filters=filters, 
        activation_str=config['activation'], 
        dropout_rate=config['dropout']
    ).to(device)
    
    print(f"Model Architecture: ModularResNet ({config['model_size']})")
    print(f"Total Trainable Parameters: {count_parameters(model):.2f} M")
    
    # 配置损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'], nesterov=True)
    else:
        raise ValueError("Unsupported optimizer.")

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    train_losses = []
    test_errors = []
    best_test_error = 100.0
    best_model_path = 'best_model.pth'

    # 开始训练循环
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止初期震荡和 Loss 爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        
        # 模型验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        test_accuracy = 100.0 * correct / total
        test_error = 100.0 - test_accuracy
        test_errors.append(test_error)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{config['epochs']}] | LR: {current_lr:.6f} | Loss: {avg_train_loss:.4f} | Test Error: {test_error:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if test_error < best_test_error:
            best_test_error = test_error
            torch.save(model.state_dict(), best_model_path)
            
    print(f"\nTraining Complete! Best Test Error achieved: {best_test_error:.2f}%")
    print(f"Model weights saved to: {best_model_path}")

    plot_training_process(train_losses, test_errors, config, best_test_error)


if __name__ == '__main__':

    # 配置模型参数
    config = {
        'epochs': 50,
        'batch_size': 128,
        
        'model_size': 'standard',    
        'activation': 'relu',        
        'dropout': 0.2,              
        'label_smoothing': 0.1,      # 标签平滑 0.1，有效防止过拟合

        'optimizer': 'sgd',          
        'lr': 0.05,                   # SGD 优化器
        'weight_decay': 5e-4,        # 稍大一些的正则化，配合 SGD
    }
    
    main(config)