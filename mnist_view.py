import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 画像をTensorに変換
transform = transforms.ToTensor()

# MNIST読み込み
trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("データ数:", len(trainset))

# 1枚取り出す
image, label = trainset[0]

print("ラベル:", label)
print("画像サイズ:", image.shape)

# 表示
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"label: {label}")
plt.show()