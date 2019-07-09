#include "classificator.h"
Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	double scale = 5;
	bool crop = false;
	int  ddepth = CV_32F;
	blobFromImage(image, inputTensor, scale, Size(width, height), mean, SwapRB, crop, ddepth);
	net.setInput(inputTensor);
	Mat result = net.forward();
	return result;
}