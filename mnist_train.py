import torch
import torch.nn as nn
from myCNN import Net
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# データ読み込み
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 学習
for epoch in range(6):

    running_loss = 0.0

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = net(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("epoch:", epoch, "loss:", running_loss)

print("学習終了")
torch.save(net.state_dict(), "mnist_model.pth")