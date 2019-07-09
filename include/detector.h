#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};
class DnnDetector: public Detector
{
private:
	//vector<string> classes;
	string pathtoModel;
	string pathtoConfig;
	int backendId;
	int targetId;
	int width;
	int height;
	double scale;
	Scalar mean;
	bool SwapRB;
	Net net;
public:
	DnnDetector(/*vector<string> classes,*/ string model, string config, int width = 224, 
		int height = 224, double scale = 0.017, Scalar mean = {103.94, 116.78, 123.68}, 
		int backendId = DNN_BACKEND_OPENCV, int targetId = DNN_TARGET_CPU, bool swapRB = false);
	vector<DetectedObject> Detect(Mat image);
};
