import dlib
import cv2
import time

hog_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

def discern(img):
    t1 = time.time()
    img = cv2.flip(img, 180)
    face_rects = hog_face_detector(img, 0)
    for faceRect in face_rects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    t2 = time.time()
    print("detect cost {}ms".format((t2-t1)*1000))
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = discern(frame)
        print(frame.shape)
        cv2.imshow("face detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("detected.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()
