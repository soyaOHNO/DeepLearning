import torch
import cv2
import numpy as np
from myCNN import CNN

model = CNN()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

# 画像読み込み
img = cv2.imread("mydigit/1.JPEG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化
_, th = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 輪郭検出
contours, _ = cv2.findContours(
    th,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    print("digit not found")
    exit()

# 最大輪郭
digit_cnt = max(contours, key=cv2.contourArea)

x,y,w,h = cv2.boundingRect(digit_cnt)

digit = th[y:y+h, x:x+w]
digit = cv2.GaussianBlur(digit,(3,3),0)

# 正方形パディング
size = max(w,h)
canvas = np.zeros((size,size),np.uint8)

canvas[
    (size-h)//2:(size-h)//2+h,
    (size-w)//2:(size-w)//2+w
] = digit

# MNISTサイズ
digit_resized = cv2.resize(canvas,(20,20))

final = np.zeros((28,28),np.uint8)
final[4:24,4:24] = digit_resized

# 重心中央寄せ
m = cv2.moments(final)

if m["m00"] != 0:

    cx = int(m["m10"]/m["m00"])
    cy = int(m["m01"]/m["m00"])

    shiftx = 14 - cx
    shifty = 14 - cy

    M = np.float32([[1,0,shiftx],[0,1,shifty]])

    final = cv2.warpAffine(final,M,(28,28))

# CNN入力
img = final / 255.0
img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

with torch.no_grad():

    output = model(img)

    pred = torch.argmax(output)

print("prediction:", pred.item())

# 確認表示
cv2.imshow("input", final)
cv2.waitKey(0)