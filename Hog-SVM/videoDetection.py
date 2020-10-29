import cv2
import dlib

# 使用Dlib的正面人脸检测器frontal_face_detector
detector = dlib.get_frontal_face_detector()

def discern(img):
    img = cv2.flip(img, 180)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("face detection", img)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = discern(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("detected.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

