//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include "detector/ultraface/ultraface.h"
#include "face_engine.h"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv) {
  if (argc <= 2) {
    fprintf(stderr, "Usage: %s <mnn .mnn> [image files...]\n", argv[0]);
    return 1;
  }

  const float scoreThreshold = 0.5;

  // // string mnn_path = argv[1];
  // char *mnn_path = argv[1];
  // // UltraFace ultraface(mnn_path, 320, 240, 4, 0.65); // config model input
  // mirror::UltraFace ultraface; // config model input
  // ultraface.Init(mnn_path);

  // for (int i = 2; i < argc; i++) {
  //   string image_file = argv[i];
  //   cout << "Processing " << image_file << endl;

  //   cv::Mat frame = cv::imread(image_file);
  //   auto start = chrono::steady_clock::now();
  //   vector<mirror::FaceInfo> face_info;
  //   ultraface.DetectFace(frame, &face_info, scoreThreshold);

  //   for (auto face : face_info) {
  //     cv::Point pt1(face.location_.x, face.location_.y);
  //     cv::Point pt2(face.location_.x + face.location_.width,
  //                   face.location_.y + face.location_.height);
  //     cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
  //   }

  //   auto end = chrono::steady_clock::now();
  //   chrono::duration<double> elapsed = end - start;
  //   cout << "all time: " << elapsed.count() << " s" << endl;
  //   cv::imshow("UltraFace", frame);
  //   cv::waitKey();
  //   string result_name = "result" + to_string(i) + ".jpg";
  // }
  //   cv::imwrite(result_name, frame);

  /* Test face_engine*/
  cv::RNG rng(12345);
  cv::Scalar color =
      cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
  char *modelRootPath = argv[1];
  mirror::FaceEngine faceEngine;
  faceEngine.Init(modelRootPath);

  for (int i = 2; i < argc; i++) {
    string image_file = argv[i];
    cout << "Processing " << image_file << endl;

    cv::Mat srcFrame = cv::imread(image_file, cv::IMREAD_UNCHANGED);
    // cv::Mat frame = cv::imread(image_file, cv::IMREAD_UNCHANGED);

    cout << "Image shape: " << srcFrame.size << " x " << srcFrame.channels()
         << endl;
    // cout << "Image shape: " << frame.size << " x " << frame.channels() <<
    // endl;
    cv::Mat frame;
    if (srcFrame.channels() == 4) {
      cv::cvtColor(srcFrame, frame, cv::COLOR_BGRA2BGR);
      cout << "Converted to 3 channels images" << endl;
    } else {
      srcFrame.copyTo(frame);
    }

    auto start = chrono::steady_clock::now();
    vector<mirror::FaceInfo> face_info;
    faceEngine.DetectFace(frame, &face_info);

    for (auto face : face_info) {
      cv::Point pt1(face.location_.x, face.location_.y);
      cv::Point pt2(face.location_.x + face.location_.width,
                    face.location_.y + face.location_.height);
      cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);

      std::vector<cv::Point2f> *keypoints = new std::vector<cv::Point2f>();
      faceEngine.ExtractKeypoints(frame, face.location_, keypoints);
      // cout << "Keypoints num: " << keypoints->size();
      for (auto k : *keypoints) {
        // cout << k << " ";
        cv::circle(frame, k, 1, color);
      }
      // cout << endl;
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "all time: " << elapsed.count() << " s" << endl;
    cv::imshow("UltraFace", frame);
    cv::waitKey();
    // string result_name = "result" + to_string(i) + ".jpg";
  }
  return 0;
}
