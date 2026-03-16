import torch
import cv2
import numpy as np
from myCNN import Net

# モデル読み込み
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()


def preprocess(path):

    # 画像読み込み
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 二値化
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 一番大きい輪郭
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + padding * 2
    h = h + padding * 2
    digit = thresh[y:y+h, x:x+w]

    # 正方形キャンバス作成
    size = max(w, h)
    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # 28×28にリサイズ
    img28 = cv2.resize(canvas, (28, 28))

    # 正規化
    img28 = img28 / 255.0

    cv2.imshow("28x28", img28)
    cv2.waitKey(0)

    return img28


for i in range(10):

    path = f"mydigit/{i}.JPEG"

    img = preprocess(path)

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    output = model(tensor)

    pred = torch.argmax(output)

    print(f"{i}.JPEG → 予測 {pred.item()}")
