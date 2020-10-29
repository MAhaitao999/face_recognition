# created at 2017-11-27
# updated at 2018-09-06

# Author:   coneypo
# Dlib:     http://dlib.net/
# Blog:     http://www.cnblogs.com/AdaminXie/
# Github:   https://github.com/coneypo/Dlib_examples

import cv2
import dlib
from skimage import io

# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()

# 图片所在路径
# img = io.imread("lena.jpg")
img = cv2.imread("lena.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

# 生成 Dlib 的图像窗口
win = dlib.image_window()
win.set_image(img)

# 使用detector检测器来检测图像中的人脸
faces = detector(img, 1)
print(type(faces[0]), '\n')

print("人脸数 / faces in all：", len(faces))

for i, d in enumerate(faces):
    print("第", i+1, "个人脸的矩形框坐标：",
          "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(),'\t',  "bottom:", d.bottom())

# 绘制矩阵轮廓
win.add_overlay(faces)

# 保持图像
dlib.hit_enter_to_continue()
