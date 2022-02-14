//
//  ViewController.mm
//  MNN
//
//  Created by MNN on 2019/02/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "ViewController.h"

#include "faceengine/common/common.h"
#include "faceengine/detector/ultraface/ultraface.h"
#include "faceengine/face_engine.h"
#include "opencv2/imgproc.hpp"
#import <AVFoundation/AVFoundation.h>
#import <MNN/ErrorCode.hpp>
#import <MNN/HalideRuntime.h>
#import <MNN/ImageProcess.hpp>
#import <MNN/Interpreter.hpp>
#import <MNN/MNNDefine.h>
#import <MNN/Tensor.hpp>
#include <vector>

typedef struct {
  float value;
  int index;
} LabeledElement;

static int CompareElements(const LabeledElement *a, const LabeledElement *b) {
  if (a->value > b->value) {
    return -1;
  } else if (a->value < b->value) {
    return 1;
  } else {
    return 0;
  }
}

#pragma mark -

@interface Model : NSObject {
}
@property(strong, nonatomic) UIImage *defaultImage;
@property(strong, nonatomic) UIImage *resultImage;
@end

@implementation Model
- (void)setType:(MNNForwardType)type threads:(NSUInteger)threads {
}

@end

@interface MNNUltraface : Model {
  mirror::UltraFace faceModel;
}
@end

@implementation MNNUltraface

- (instancetype)init {
  if (self) {
    NSString *model = [[NSBundle mainBundle] pathForResource:@"rfb320"
                                                      ofType:@"bin"];

    //        TODO: dot in path
    //        NSArray *modelSplits = [model componentsSeparatedByString:@"."];
    //        NSLog(@"Model path full: %@", model);

    //        NSString *modelDir = modelSplits[0];
    self.defaultImage = [UIImage imageNamed:@"Hari_Won.png"];
    self.resultImage = [UIImage imageNamed:@"Hari_Won.png"];

    faceModel.Init(model.UTF8String);
  }
  return self;
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
  std::vector<mirror::FaceInfo> faces;
  cv::Mat img_src = mirror::cvMatFromUIImage(image);
  faceModel.DetectFace(img_src, &faces, 0.3);
  int num_faces = static_cast<int>(faces.size());
  NSString *result = @"";

  for (int i = 0; i < num_faces; i++) {
    cv::Rect faceBox = faces[i].location_;
    result = [result
        stringByAppendingFormat:@"xywh: %d %d %d %d, %f\n",
                                faces[i].location_.x, faces[i].location_.y,
                                faces[i].location_.width,
                                faces[i].location_.height, faces[i].score_];
    NSLog(@"%@", result);

    cv::rectangle(img_src, faceBox, cv::Scalar(0, 255, 0));
  }

  self.resultImage = mirror::UIImageFromCVMat(img_src);

  return result;
}
@end

@interface MNNFaceEngine : NSObject {
  mirror::FaceEngine faceModel;
}

@implementation MNNFaceEngine

- (instancetype)init {
  if (self) {
    NSString *model = [[NSBundle mainBundle] pathForResource:@"rfb320"
                                                      ofType:@"bin"];

    //        TODO: dot in path
    NSArray *modelSplits = [model componentsSeparatedByString:@"."];
    //        NSLog(@"Model path full: %@", model);

    NSString *modelDir = modelSplits[0];
    self.defaultImage = [UIImage imageNamed:@"Hari_Won.png"];
    self.resultImage = [UIImage imageNamed:@"Hari_Won.png"];

    faceModel.Init(modelDir.UTF8String);
  }
  return self;
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
  std::vector<mirror::FaceInfo> faces;
  cv::Mat img_src = mirror::cvMatFromUIImage(image);
  faceModel.DetectFace(img_src, &faces);
  int num_faces = static_cast<int>(faces.size());
  NSString *result = @"";

  for (int i = 0; i < num_faces; i++) {
    cv::Rect faceBox = faces[i].location_;
    result = [result
        stringByAppendingFormat:@"xywh: %d %d %d %d, %f\n",
                                faces[i].location_.x, faces[i].location_.y,
                                faces[i].location_.width,
                                faces[i].location_.height, faces[i].score_];
    NSLog(@"%@", result);

    cv::rectangle(img_src, faceBox, cv::Scalar(0, 255, 0));
    std::vector<cv::Point2f> keypoints;
    faceModel.ExtractKeypoints(frame, face.location_, &keypoints);
    auto color = cv::Scalar(0, 255, 0);
    for (auto k : *keypoints) {
      // cout << k << " ";
      cv::circle(img_src, k, 1, color);
    }
  }

  self.resultImage = mirror::UIImageFromCVMat(img_src);

  return result;
}
@end
#pragma mark -

@interface ViewController () <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(assign, nonatomic) MNNForwardType forwardType;
@property(assign, nonatomic) int threadCount;

/* @property(strong, nonatomic) Model *mobileNetV2; */
/* @property(strong, nonatomic) Model *squeezeNetV1_1; */
@property(strong, nonatomic) Model *currentModel;
@property(strong, nonatomic) Model *ultraface;
@property(strong, nonatomic) Model *faceengine;

@property(strong, nonatomic) AVCaptureSession *session;
@property(strong, nonatomic) IBOutlet UIImageView *imageView;
@property(strong, nonatomic) IBOutlet UILabel *resultLabel;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *modelItem;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *forwardItem;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *threadItem;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *runItem;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *benchmarkItem;
@property(strong, nonatomic) IBOutlet UIBarButtonItem *cameraItem;

@end

@implementation ViewController

- (void)awakeFromNib {
  [super awakeFromNib];

  self.forwardType = MNN_FORWARD_CPU;
  self.threadCount = 4;
  self.ultraface = [MNNUltraface new];
  self.faceengine = [MNNFaceEngine new];
  self.currentModel = self.ultraface;

  AVCaptureSession *session = [[AVCaptureSession alloc] init];
  session.sessionPreset = AVCaptureSessionPreset1280x720;
  AVCaptureDevice *device =
      [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  AVCaptureDeviceInput *input =
      [[AVCaptureDeviceInput alloc] initWithDevice:device error:NULL];
  AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
  [output setSampleBufferDelegate:self
                            queue:dispatch_queue_create("video_infer", 0)];
  output.videoSettings =
      @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};

  if ([session canAddInput:input]) {
    [session addInput:input];
  }
  if ([session canAddOutput:output]) {
    [session addOutput:output];
  }
  [session commitConfiguration];

  self.session = session;
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
  [self refresh];
}

- (void)refresh {
  [_currentModel setType:_forwardType threads:_threadCount];
  [self run];
}

- (IBAction)toggleInput {
  if (_session.running) {
    [self usePhotoInput];
    [self run];
  } else {
    [self useCameraInput];
  }
}

- (void)useCameraInput {
  [_session startRunning];
  self.navigationItem.leftBarButtonItem.title = @"Photo";
  self.runItem.enabled = NO;
  self.benchmarkItem.enabled = NO;
}

- (void)usePhotoInput {
  [_session stopRunning];
  _imageView.image = _currentModel.defaultImage;
  self.navigationItem.leftBarButtonItem.title = @"Camera";
  self.runItem.enabled = YES;
  self.benchmarkItem.enabled = YES;
}

- (IBAction)toggleModel {
  __weak typeof(self) weakify = self;
  UIAlertController *alert = [UIAlertController
      alertControllerWithTitle:@"选择模型"
                       message:nil
                preferredStyle:UIAlertControllerStyleActionSheet];
  [alert addAction:[UIAlertAction actionWithTitle:@"取消" sty
                                                e:UIAlertActionStyleCancel han
                                              ler:nil]];
  [alert addAction:[UIAlertAction
                       actionWithTitle:@"Ultraface"
                                 style:UIAlertActionStyleDefault
                               handler:^(UIAlertAction *action) {
                                 __strong typeof(weakify) self = weakify;
                                 self.modelItem.title = action.title;
                                 self.currentModel = self.ultraface;
                                 if (!self.session.running) {
                                   self.imageView.image =
                                       self.currentModel.defaultImage;
                                 }

                                 [self refresh];
                               }]];

  [alert addAction:[UIAlertAction
                       actionWithTitle:@"FaceEngine"
                                 style:UIAlertActionStyleDefault
                               handler:^(UIAlertAction *action) {
                                 __strong typeof(weakify) self = weakify;
                                 self.modelItem.title = action.title;
                                 self.currentModel = self.ultraface;
                                 if (!self.session.running) {
                                   self.imageView.image =
                                       self.currentModel.defaultImage;
                                 }

                                 [self refresh];
                               }]];
  [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleMode {
  __weak typeof(self) weakify = self;
  UIAlertController *alert = [UIAlertController
      alertControllerWithTitle:@"运行模式"
                       message:nil
                preferredStyle:UIAlertControllerStyleActionSheet];
  [alert addAction:[UIAlertAction actionWithTitle:@"取消" sty
                                                e:UIAlertActionStyleCancel han
                                              ler:nil]];
  [alert addAction:[UIAlertAction
                       actionWithTitle:@"CPU"
                                 style:UIAlertActionStyleDefault
                               handler:^(UIAlertAction *action) {
                                 __strong typeof(weakify) self = weakify;
                                 self.forwardItem.title = action.title;
                                 self.forwardType = MNN_FORWARD_CPU;
                                 [self refresh];
                               }]];
  [alert addAction:[UIAlertAction
                       actionWithTitle:@"Metal"
                                 style:UIAlertActionStyleDefault
                               handler:^(UIAlertAction *action) {
                                 __strong typeof(weakify) self = weakify;
                                 self.forwardItem.title = action.title;
                                 self.forwardType = MNN_FORWARD_METAL;
                                 [self refresh];
                               }]];
  [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleThreads {
  __weak typeof(self) weakify = self;
  void (^onToggle)(UIAlertAction *) = ^(UIAlertAction *action) {
    __strong typeof(weakify) self = weakify;
    self.threadItem.title = [NSString stringWithFormat:@"%@", action.title];
    self.threadCount = action.title.intValue;
    [self refresh];
  };
  UIAlertController *alert = [UIAlertController
      alertControllerWithTitle:@"Thread Count"
                       message:nil
                preferredStyle:UIAlertControllerStyleActionSheet];
  [alert addAction:[UIAlertAction actionWithTitle:@"取消" sty
                                                e:UIAlertActionStyleCancel han
                                              ler:nil]];
  [alert addAction:[UIAlertAction actionWithTitle:@"1"
                                            style:UIAlertActionStyleDefault
                                          handler:onToggle]];
  [alert addAction:[UIAlertAction actionWithTitle:@"2"
                                            style:UIAlertActionStyleDefault
                                          handler:onToggle]];
  [alert addAction:[UIAlertAction actionWithTitle:@"4"
                                            style:UIAlertActionStyleDefault
                                          handler:onToggle]];
  [alert addAction:[UIAlertAction actionWithTitle:@"8"
                                            style:UIAlertActionStyleDefault
                                          handler:onToggle]];
  [alert addAction:[UIAlertAction actionWithTitle:@"10"
                                            style:UIAlertActionStyleDefault
                                          handler:onToggle]];
  [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)run {
  if (!_session.running) {
    self.resultLabel.text = [_currentModel inferImage:_imageView.image
                                               cycles:1];
    self.imageView.image = self.currentModel.resultImage;
  }
}

- (IBAction)benchmark {
  if (!_session.running) {
    self.cameraItem.enabled = NO;
    self.runItem.enabled = NO;
    self.benchmarkItem.enabled = NO;
    self.modelItem.enabled = NO;
    self.forwardItem.enabled = NO;
    self.threadItem.enabled = NO;
    UIImage *image = self->_imageView.image;
    dispatch_async(
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
          NSString *str = [self->_currentModel inferImage:image cycles:100];
          dispatch_async(dispatch_get_main_queue(), ^{
            self.resultLabel.text = str;
            self.cameraItem.enabled = YES;
            self.runItem.enabled = YES;
            self.benchmarkItem.enabled = YES;
            self.modelItem.enabled = YES;
            self.forwardItem.enabled = YES;
            self.threadItem.enabled = YES;
          });
        });
  }
}

#pragma mark AVCaptureAudioDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
  CIImage *ci = [[CIImage alloc]
      initWithCVPixelBuffer:CMSampleBufferGetImageBuffer(sampleBuffer)];
  CIContext *context = [[CIContext alloc] init];
  CGImageRef cg = [context createCGImage:ci fromRect:ci.extent];

  UIImageOrientation orientaion;
  switch (connection.videoOrientation) {
  case AVCaptureVideoOrientationPortrait:
    orientaion = UIImageOrientationUp;
    break;
  case AVCaptureVideoOrientationPortraitUpsideDown:
    orientaion = UIImageOrientationDown;
    break;
  case AVCaptureVideoOrientationLandscapeRight:
    orientaion = UIImageOrientationRight;
    break;
  case AVCaptureVideoOrientationLandscapeLeft:
    orientaion = UIImageOrientationLeft;
    break;
  default:
    break;
  }

  UIImage *image = [UIImage imageWithCGImage:cg
                                       scale:1.f
                                 orientation:orientaion];
  CGImageRelease(cg);
  /* NSString *result = [_currentModel inferBuffer:sampleBuffer]; */
  NSString *result = [_currentModel inferImage:image];

  dispatch_async(dispatch_get_main_queue(), ^{
    if (self.session.running) {
      /* self.imageView.image = image; */
      self.imageView.image = self.currentModel.resultImage;
      self.resultLabel.text = result;
    }
  });
}

@end
