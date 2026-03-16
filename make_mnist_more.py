import cv2
import numpy as np
import os
import random

input_dir = "dataset"
output_dir = "dataset_aug"

for i in range(10):
    os.makedirs(f"{output_dir}/{i}", exist_ok=True)


def augment(img):

    h,w = img.shape

    # 回転
    angle = random.uniform(-15,15)
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img = cv2.warpAffine(img,M,(w,h),borderValue=0)

    # 平行移動
    tx = random.uniform(-3,3)
    ty = random.uniform(-3,3)

    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(w,h),borderValue=0)

    # スケール
    scale = random.uniform(0.9,1.1)
    img = cv2.resize(img,None,fx=scale,fy=scale)

    img = cv2.resize(img,(28,28))

    # ノイズ
    noise = np.random.normal(0,10,(28,28))
    img = img + noise
    img = np.clip(img,0,255)

    return img.astype(np.uint8)


for label in range(10):

    files = os.listdir(f"{input_dir}/{label}")

    count = 0

    for f in files:

        img = cv2.imread(f"{input_dir}/{label}/{f}",0)

        for i in range(100):

            aug = augment(img)

            count += 1

            cv2.imwrite(
                f"{output_dir}/{label}/{label}_{count}.png",
                aug
            )

print("Finished")