#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define WINDOW_NAME1 "原始图"
#define WINDOW_NAME2 "灰度图"
#define WINDOW_NAME3 "检测图"

int main(int argc, char *argv[]) {

    // [1] 摄像头读入视频
    VideoCapture cameraCapture(0);

    // [2] 参数定义
    Mat srcFrame, dstFrame, grayFrame;
    cameraCapture >> srcFrame;
    auto faceCascade = CascadeClassifier("haarcascade_frontalface_default.xml");
    auto eyeCascade = CascadeClassifier("haarcascade_eye.xml");
    int inputKey = 0;

    while (true) {
        cameraCapture >> srcFrame;
        dstFrame = srcFrame.clone();
	cvtColor(srcFrame, grayFrame, CV_BGR2GRAY);

        vector<Rect> faces;
	faceCascade.detectMultiScale(grayFrame, faces, 1.3, 5, 0); // 分类器对象的调用
	printf("检测到人脸的个数: %lu\n", faces.size());

	for (unsigned int i=0; i < faces.size(); i++) {
	    rectangle(dstFrame, Point(faces[i].x, faces[i].y), Point(faces[i].x+faces[i].width, faces[i].y+faces[i].height), Scalar(0, 255, 0), 2);
	    Mat roiGray = grayFrame(Range(faces[i].y, faces[i].y+faces[i].height), Range(faces[i].x, faces[i].x+faces[i].width));
	    Mat roiColor = dstFrame(Range(faces[i].y, faces[i].y+faces[i].height), Range(faces[i].x, faces[i].x+faces[i].width));
	    vector<Rect> eyes;
	    eyeCascade.detectMultiScale(roiGray, eyes);
	    for (unsigned int j=0; j < eyes.size(); j++) {
	        rectangle(roiColor, Point(eyes[j].x, eyes[j].y), Point(eyes[j].x+eyes[j].width, eyes[j].y+eyes[j].height), Scalar(255, 0, 0), 2);
	    }
	}

	imshow(WINDOW_NAME1, srcFrame);
        imshow(WINDOW_NAME2, grayFrame);
	imshow(WINDOW_NAME3, dstFrame);
	inputKey = waitKey(1);
	if (inputKey == 'q') {
	    imwrite("result.jpg", dstFrame);
	    break;
	}
    }

    return 0;
}
