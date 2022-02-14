//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#ifndef arcface_hpp
#define arcface_hpp

#pragma once

#include "MNN/Interpreter.hpp"

#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "../recognizer.h"

namespace mirror {
    class ArcFace : public Recognizer {
    public:
        ArcFace(int num_thread_);

        ~ArcFace();

        int Init(const char* model_path);
        int InitMem(void* buffer, size_t size);
        int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat);

    private:
        std::shared_ptr<MNN::Interpreter> arcface_interpreter;
        MNN::Session *arcface_session = nullptr;
        MNN::Tensor *input_tensor = nullptr;
        MNN::BackendConfig backendConfig;

        int num_thread;
        int image_h;
        int image_w;

        int in_w = 112;
        int in_h = 112;

//        const float mean_vals[3] = {127.5, 127.5, 127.5};
//        const float norm_vals[3] = {1.0f / 128, 1.0f / 128, 1.0f / 128};
        const float mean_vals[3] = {0.0, 0.0, 0.0};
        const float norm_vals[3] = {1.0, 1.0, 1.0};

    };
}
#endif
