from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

IMG_PATH = "images/"
IMG_NAME = "circle.png"
RESIZE_TO = 64

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
    