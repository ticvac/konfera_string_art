from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pygame
from random import randint

IMG_PATH = "images/"
IMG_NAME = "circle.png"
RESIZE_TO = 64
VECTOR_SIZE = 500
NAILS = 256

cv2_img = cv2.imread(IMG_PATH + IMG_NAME)
cv2_img = cv2.resize(cv2_img, (RESIZE_TO, RESIZE_TO))
cv2.imwrite(IMG_PATH + "resized_" + str(RESIZE_TO) + "_" + IMG_NAME, cv2_img)

img = Image.open(IMG_PATH + "resized_" + str(RESIZE_TO) + "_" + IMG_NAME)
# process
img = ImageOps.grayscale(img)
# img = ImageOps.invert(img) # why ??
img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
img = ImageEnhance.Contrast(img).enhance(1)
np_img = np.array(img)

print(len(np_img))
# print(np_img)

for i in np_img:
    for j in i:
        j_print = " " * (3 - len(str(j))) + str(j)
        print(j_print, end=" ")
    print("\n")
    
nails = []
# make 100 nails in a circle
for i in range(NAILS):
    x = VECTOR_SIZE * pygame.math.Vector2(1, 0).rotate(i * 360 / NAILS)
    nails.append((int(x.x) + VECTOR_SIZE, int(x.y) + VECTOR_SIZE))

plt.gca().set_aspect('equal', adjustable='box')
for n in nails:
    plt.plot(n[0], n[1], "ro", markersize=1)

lines = []
for _ in range(10):
    lines.append((nails[randint(0, NAILS-1)], nails[randint(0, NAILS-1)]))

# plotting lines between nails
for l in lines:
    plt.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], "k-", linewidth=0.1)


plt.show()