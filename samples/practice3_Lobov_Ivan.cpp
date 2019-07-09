#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  heigth                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ classes                              |        | path to classes file              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

void drawDetections(vector<DetectedObject> detected_objects, Mat& image)
{
	auto beg = detected_objects.begin();
	for (auto it = beg; it != detected_objects.end(); ++it) {
		//New box with detection
		rectangle(image,
			Point(it->Left, it->Top),
			Point(it->Right, it->Bottom),
			Scalar(0, 255, 0));
		//Draw classname and confidence on the white background near detected box
		string label = format("%.5f", it->confidence);
		label = it->classname + ": " + label;

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int top;
		top = max(it->Top, labelSize.height);
		rectangle(image,
			Point(it->Left, top - labelSize.height),
			Point(it->Left + labelSize.width, it->Top + baseLine),
			Scalar::all(255), FILLED);
		putText(image, label, Point(it->Left, it->Top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
	}
} 

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  vector<string> classes;
  string line;


  //string pathtoClasses = parser.get<String>("classes");
  string pathtoModel = parser.get<String>("model_path");
  string pathtoConfig = parser.get<String>("config_path");
  String pathtoImg(parser.get<String>("image"));
  int width = 300;
  int heigth = 300;
  Scalar mean = { 127.5,127.5,127.5 };
  bool swapRB = false;
  double scale = 0.007843;
  /*ifstream classesFile(pathtoClasses);
  
  while (getline(classesFile, line))
  {
	  classes.push_back(line);
  }*/

  Mat image = imread(pathtoImg);
  Detector* detector = new DnnDetector(/*classes,*/ pathtoModel, pathtoConfig, 
	  width, heigth, scale, mean, swapRB);
  vector<DetectedObject> detected = detector->Detect(image);
  drawDetections(detected, image);
  namedWindow("Detected image", WINDOW_AUTOSIZE);
  imshow("Detected image", image);
  waitKey();

  return 0;
}

