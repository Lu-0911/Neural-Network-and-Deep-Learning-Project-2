import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ModularResNet, count_parameters
from engine import get_dataloaders, train_model

epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对比实验可视化函数
def plot_comparison(histories, title, filename):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], marker='o', markersize=3, label=name)
    plt.title(f'{title} - Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(range(1, len(history['test_acc'])+1), history['test_acc'], marker='s', markersize=3, label=name)
    plt.title(f'{title} - Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def run_all_experiments():
    print(f"========== Using Device: {device} ==========")
    trainloader, testloader = get_dataloaders(batch_size=128)
    
    # ----------------------------------------------------
    # 实验 1: 不同的 Filters (网络宽度对比)
    # ----------------------------------------------------
    print("\n[Experiment 1] Network Width (Filters)")
    h1 = {}
    width_configs = {
        'Small [32, 64, 128]': [32, 64, 128],
        'Standard [64, 128, 256]': [64, 128, 256],
        'Wide [128, 256, 512]': [128, 256, 512]
    }
    for name, filters in width_configs.items():
        print(f"--- Running: {name} ---")
        model = ModularResNet(filters=filters)
        print(f"Params: {count_parameters(model):.2f} M")
        
        # 统一采用 SGD + 学习率退火
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        h1[name] = train_model(model, opt, nn.CrossEntropyLoss(), trainloader, testloader, epochs, sched, device)
    plot_comparison(h1, 'Network Width', 'exp1_filters.png')

    # ----------------------------------------------------
    # 实验 2: 不同的激活函数 (Activations)
    # ----------------------------------------------------
    print("\n[Experiment 2] Activation Functions")
    h2 = {}
    for act in ['relu', 'gelu', 'leaky_relu']:
        print(f"--- Running: {act.upper()} ---")
        model = ModularResNet(filters=[64, 128, 256], activation_str=act)
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        h2[act.upper()] = train_model(model, opt, nn.CrossEntropyLoss(), trainloader, testloader, epochs, sched, device)
    plot_comparison(h2, 'Activations', 'exp2_activations.png')

    # ----------------------------------------------------
    # 实验 3: 不同的优化器 (Optimizers)
    # ----------------------------------------------------
    print("\n[Experiment 3] Optimizers")
    h3 = {}
    opt_configs = {
        'Adam': lambda m: optim.Adam(m.parameters(), lr=0.001),
        'AdamW': lambda m: optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01),
        'SGD + Momentum': lambda m: optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    }
    for name, get_opt in opt_configs.items():
        print(f"--- Running: {name} ---")
        model = ModularResNet(filters=[64, 128, 256])
        opt = get_opt(model)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        h3[name] = train_model(model, opt, nn.CrossEntropyLoss(), trainloader, testloader, epochs, sched, device)
    plot_comparison(h3, 'Optimizers', 'exp3_optimizers.png')
    
    # ----------------------------------------------------
    # 实验 4: 正则化与损失函数 (Regularization & Loss)
    # ----------------------------------------------------
    print("\n[Experiment 4] Regularization Strategies")
    h4 = {}
    reg_configs = {
        'Base (No Extra Reg)': {'wd': 0.0, 'drop': 0.3, 'loss': nn.CrossEntropyLoss()},
        'L2 Weight Decay (5e-4)': {'wd': 5e-4, 'drop': 0.3, 'loss': nn.CrossEntropyLoss()},
        'High Dropout (p=0.6)': {'wd': 0.0, 'drop': 0.6, 'loss': nn.CrossEntropyLoss()},
        'Label Smoothing (0.1)': {'wd': 0.0, 'drop': 0.3, 'loss': nn.CrossEntropyLoss(label_smoothing=0.1)}
    }
    for name, cfg in reg_configs.items():
        print(f"--- Running: {name} ---")
        model = ModularResNet(filters=[64, 128, 256], dropout_rate=cfg['drop'])
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=cfg['wd'])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        h4[name] = train_model(model, opt, cfg['loss'], trainloader, testloader, epochs, sched, device)
    plot_comparison(h4, 'Regularization', 'exp4_regularization.png')

    print("\nAll experiments finished!")

if __name__ == '__main__':
    run_all_experiments()