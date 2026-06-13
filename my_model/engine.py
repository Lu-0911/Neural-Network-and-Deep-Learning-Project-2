import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(batch_size=128):
    # 数据增强，含随机裁剪/水平翻转/旋转
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 创建数据加载器
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train_model(model, optimizer, criterion, trainloader, testloader, epochs=50, scheduler=None, device='cuda', save_name=None):
    model = model.to(device)
    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            # 模型训练
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪，防止初期震荡和 Loss 爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(trainloader)
        history['train_loss'].append(train_loss)
        
        # 模型验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        history['test_acc'].append(acc)
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1:02d}/{epochs}] | LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Test Acc: {acc:.2f}%")
        
        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step()
        
        # 保存最佳模型
        if save_name and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{save_name}.pth')
            
    print(f"  Best Test Acc: {max(history['test_acc']):.2f}%\n")
        
    return history