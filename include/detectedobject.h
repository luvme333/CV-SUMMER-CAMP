#pragma once
#include <string>

struct DetectedObject
{
    int Left;
    int Right;
    int Top;
    int Bottom;
    int uuid;
	double confidence;
    std::string classname;
};