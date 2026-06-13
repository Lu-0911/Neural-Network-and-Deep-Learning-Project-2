import torch
import torch.nn as nn
import os
from model import ModularResNet
from engine import get_dataloaders

# CIFAR-10 的 10 个具体类别名称
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def test_model(model_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 检查权重文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' does not exist.")
        return

    # 初始化网络架构 (与训练的架构一致)
    model = ModularResNet(filters=[64, 128, 256], activation_str='relu')
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load weights. Error: {e}")
        return
        
    model = model.to(device)
    
    # 切换到验证模式
    model.eval()

    # 加载测试数据集
    print("Loading CIFAR-10 test dataset...")
    # 只取 testloader
    _, testloader = get_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss()

    # 初始化统计变量
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于统计 10 个类别的分别准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # 开始测试循环
    with torch.no_grad(): # 不计算梯度，节省显存并加速
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 模型推理
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 打印结果
    avg_loss = running_loss / len(testloader)
    final_acc = 100.0 * correct / total
    final_error = 100.0 - final_acc
    
    print(f"Total Test Images : {total}")
    print(f"Overall Test Loss : {avg_loss:.4f}")
    print(f"Overall Test Acc  : {final_acc:.2f}%")
    print(f"Overall Test Error: {final_error:.2f}%")
    print("-" * 45)
    
    print("Per-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f" - {CLASSES[i]:<5s} : {acc:5.2f}%")
        else:
            print(f" - {CLASSES[i]:<5s} : N/A")

if __name__ == '__main__':

    test_model('best_model.pth')