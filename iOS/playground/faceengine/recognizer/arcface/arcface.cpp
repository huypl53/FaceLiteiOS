//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "arcface.hpp"

using namespace std;

namespace mirror {
    ArcFace::ArcFace(int num_thread_) {
        num_thread = num_thread_;
    }

    int ArcFace::Init(const char *model_path) {
        return 1;
    }

    int ArcFace::InitMem(void* buffer, size_t size) {
        arcface_interpreter = std::shared_ptr<MNN::Interpreter>(
                MNN::Interpreter::createFromBuffer(buffer, size));
        MNN::ScheduleConfig config;
        config.numThread = num_thread;

        backendConfig.precision = (MNN::BackendConfig::PrecisionMode) MNN::BackendConfig::Precision_Low;

        config.backendConfig = &backendConfig;
        config.type = (MNNForwardType) MNN_FORWARD_CPU;
//        config.type = (MNNForwardType) MNN_FORWARD_OPENCL;

        arcface_session = arcface_interpreter->createSession(config);

        input_tensor = arcface_interpreter->getSessionInput(arcface_session, "data");
        arcface_interpreter->resizeTensor(input_tensor, {1, 3, in_w, in_h});
        arcface_interpreter->resizeSession(arcface_session);

        return 0;
    }

    ArcFace::~ArcFace() {
        arcface_interpreter->releaseModel();
        arcface_interpreter->releaseSession(arcface_session);
    }

    int ArcFace::ExtractFeature(const cv::Mat &img_face, std::vector<float> *feat) {
        image_h = img_face.rows;
        image_w = img_face.cols;
        cv::Mat image = img_face.clone();
        cv::resize(img_face, image, cv::Size(in_w, in_h));

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
                                              norm_vals, 3));
        pretreat->convert(image.data, 112, 112, image.step[0], input_tensor);

        // run network
        arcface_interpreter->runSession(arcface_session);

        // get output data
        string scores = "fc1";

        MNN::Tensor *tensor_scores = arcface_interpreter->getSessionOutput(arcface_session,
                                                                           scores.c_str());

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());

        tensor_scores->copyToHostTensor(&tensor_scores_host);

        vector<int> output_tensor_shape = tensor_scores->shape();

        for (int i = 0; i < output_tensor_shape[1]; i++) {
            float t1 = tensor_scores->host<float>()[i];
            feat->push_back(t1);
        }

        return 0;
    }
}
