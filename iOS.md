# Chuẩn bị

- Tải opencv
  
  ```bash
  cd frameworks
  wget https://sourceforge.net/projects/opencvlibrary/files/3.4.13/opencv-3.4.13-ios-framework.zip
  unzip -q opencv-3.4.13-ios-framework.zip
  ```
- Thêm opencv vào project `Link binary With Libraries`
- Thêm các files trong `playground/faceengine/**` vào `Compile Sources`
- Thêm các ảnh `iOS/playground/assets/*.jpg` và các trọng số `iOS/playground/assets/models/*` vào `Copy Bundle Resources`
- Mở `playground.xcworkspace`
  
  # Thử nghiệm
- Ảnh load lưu bằng `UIImage`, chuyển sang `cv::Mat` bằng `mirror::cvMatFromUIImage` trong `common.h`; chuyển từ `cv::Mat` sang `UIImage` bằng`mirror::UIImageFromCVMat` (`iOS/playground/faceengine/common/common.h`)
- Ảnh đầu vào cho `mirror::FaceEngine` (`iOS/playground/faceengine/face_engine.h`) là `cv::Mat`
  
  > Hạn chế: ảnh lấy bởi camera có 4 kênh màu, ảnh thông thường có 3 kênh màu. Mặc định `mirror:cvMatFromUIImage` sử dụng ảnh từ `UIImage` 4 kênh màu

***

Code demo `iOS/playground/ViewController.mm`