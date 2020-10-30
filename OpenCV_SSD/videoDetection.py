import cv2
import time

# 设置参数, 同时载入模型
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)
threshold = 0.9

# print(net)

# 加载图片
cameraCapture = cv2.VideoCapture(0)
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frameWidth = size[0]
frameHeight =size[1]

while True:
    t1 = time.time()
    ret, frame = cameraCapture.read()
    # 进行必要的预处理工作
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # 设置网络输入
    net.setInput(blob)
    detections = net.forward()
    t2 = time.time()
    print("time costs: {}ms".format((t2-t1)*1000))
    # print(detections.shape)

    for i in range(detections.shape[2]):
        detection_score = detections[0, 0, i, 2]
        # 与阈值做对比, 同一个人脸该过程会进行很多次
        if detection_score > threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # 绘制矩形
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


    # 保持输出
    cv2.imshow("detected result", frame)
    inputKey = cv2.waitKey(20)
    if inputKey == ord('q'):
        cv2.imwrite("found_face.jpg", frame)
        break

cameraCapture.release() # 释放摄像头
cv2.destroyAllWindows() # 销毁窗口
