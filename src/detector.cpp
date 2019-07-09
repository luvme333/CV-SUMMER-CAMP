#include "detector.h"
DnnDetector::DnnDetector(/*vector<string> _classes,*/ string model, string config, int _width, int _height,
	double _scale, Scalar _mean, int _backendId, int _targetId, bool _SwapRB)
{
	//classes = _classes;
	pathtoModel = model;
	pathtoConfig = config;
	width = _width;
	height = _height;
	scale = _scale;
	mean = _mean;
	SwapRB = _SwapRB;
	backendId = _backendId;
	targetId = _targetId;
	net = readNet(pathtoModel, pathtoConfig);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}
vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	vector<DetectedObject> vdo;
	Mat inputTensor;
	Mat result;

	blobFromImage(image, inputTensor, scale, Size(width, height), mean, SwapRB, false);
	net.setInput(inputTensor);
	result = net.forward();
	result = result.reshape(1, 1);
	result = result.reshape(1, result.cols / 7);

	for (int i = 0; i < result.rows; i++)
	{
		DetectedObject detected;
		if (true)
		{
			detected.uuid = result.at<float>(i, 1);
			detected.Left = result.at<float>(i, 3)*image.cols;
			detected.Bottom = result.at<float>(i, 4)*image.rows;
			detected.Right = result.at<float>(i, 5)*image.cols;
			detected.Top = result.at<float>(i, 6)*image.rows;
			//detected.classname = classes[detected.uuid];
			vdo.push_back(detected);
		}
	}
	return vdo;
}