import cv2
# 设置参数, 同时载入模型
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)
threshold = 0.9

print(net)

# 加载图片
img = cv2.imread("lena.jpg")
print(img.shape)
frameHeight = img.shape[0]
frameWidth = img.shape[1]

# 进行必要的预处理工作
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

# 设置网络输入
net.setInput(blob)
detections = net.forward()
print(detections.shape)

for i in range(detections.shape[2]):
    detection_score = detections[0, 0, i, 2]
    # 与阈值做对比, 同一个人脸该过程会进行很多次
    if detection_score > threshold:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        # 绘制矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


# 保持输出
cv2.imshow("detected result", img)
inputKey = cv2.waitKey(0)
if inputKey == ord('q'):
    cv2.imwrite("found_face.jpg", img)
