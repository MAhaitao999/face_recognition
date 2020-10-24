import cv2
import os

cameraCapture = cv2.VideoCapture(0)
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    ret, frame = cameraCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roiGray = gray[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roiGray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("face and eye detection", frame)

    inputKey = cv2.waitKey(20)
    if inputKey == ord('q'):
        break

cameraCapture.release() # 释放摄像头
cv2.destroyAllWindows() # 销毁窗口
