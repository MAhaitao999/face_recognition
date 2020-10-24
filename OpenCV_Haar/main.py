import cv2
# 创建人脸检测级联分类器对象实例
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(face_cascade)
# 或采用lbp特征进行检测
# face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
# 创建人眼检测级联分类器实例
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# 载入图片
img = cv2.imread("lena.jpg")
print(img)
# 图片颜色意义不大, 灰度化处理即可
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 调用级联分类器进行多尺度检测
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# 遍历检测到的结果
for (x, y, w, h) in faces:
    # 绘制矩形框, 颜色值的顺序为BGR, 即矩形框的颜色为蓝色
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # roi即region of interest, 意思是感兴趣的区域
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # 在检测到的人脸区域内检测眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv2.imwrite("detected_face.jpg", img)
cv2.imshow("detected image", img)
cv2.waitKey(0)
