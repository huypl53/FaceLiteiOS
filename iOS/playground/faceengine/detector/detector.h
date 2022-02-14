#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include "../common/common.h"

namespace mirror {
class Detector {
public:
    virtual int Init(const char* model_path) = 0;
	virtual int DetectFace(const cv::Mat& img_src,
        std::vector<FaceInfo>* faces, const float scoreThreshold) = 0;
    virtual ~Detector() {}
};

class DetecterFactory {
public:
    virtual Detector* CreateDetecter() = 0;
    virtual ~DetecterFactory() {}
};

class CenterfaceFactory : public DetecterFactory {
public:
    CenterfaceFactory() {}
    Detector* CreateDetecter();
    ~CenterfaceFactory() {}
};

class UltrafaceFactory : public DetecterFactory {
public:
    UltrafaceFactory() {}
    Detector* CreateDetecter();
    ~UltrafaceFactory() {}
};

}


#endif // !_FACE_DETECTER_H_
