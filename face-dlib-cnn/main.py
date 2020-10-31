import dlib
import cv2
import time

img = cv2.imread("lena.jpg")
hog_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
face_rects = hog_face_detector(img, 0)
t1 = time.time()
face_rects = hog_face_detector(img, 0)
t2 = time.time()
print("face detect costs {}ms".format((t2-t1)*1000))
# print(face_rects)

for faceRect in face_rects:
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("detected face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

