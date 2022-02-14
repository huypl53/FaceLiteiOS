//
//  ConvolutionTiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionTiledExecutor_hpp
#define ConvolutionTiledExecutor_hpp


#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
class ConvolutionTiledImpl : public CPUConvolution {
public:
    ConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b) : CPUConvolution(common, b) {
        // Do nothing
    }
    virtual ~ConvolutionTiledImpl() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) = 0;

protected:
    Tensor mTempBufferTranspose;
    std::pair<int, std::function<void(int)>> mFunction;
};

class ConvolutionTiledExecutor : public Execution {
public:
    ConvolutionTiledExecutor(Backend* b, const float* bias, size_t biasSize);
    ConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, Backend* b);
    virtual ~ConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_EXECUTION;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_EXECUTION;
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function);

protected:
    std::vector<Tensor *> mInputs;
    std::shared_ptr<CPUConvolution::Resource> mResource;
};


#define GENERATE_FUNCTOR  //  empty
#define GENERATE_WEIGHT   //  empty
#define GENERATE_MM       //  empty

#define GENERATE_RESIZE()                                                                                                      \
    CPUConvolution::onResize(inputs, outputs);                                                                                 \
    auto input   = inputs[0];                                                                                                  \
    auto weight  = inputs[1];                                                                                                  \
    Tensor *bias = nullptr;                                                                                                    \
    auto core    = static_cast<CPUBackend *>(backend())->functions();                                                          \
    int bytes    = core->bytes;                                                                                                \
    int unit     = core->pack;                                                                                                 \
    auto packA   = core->MNNPackC4ForMatMul_A;                                                                                 \
    int eP, lP, hP;                                                                                                            \
    getPackParameter(&eP, &lP, &hP, core);                                                                                     \
                                                                                                                               \
    GENERATE_FUNCTOR();                                                                                                        \
                                                                                                                               \
    const float *biasPtr = nullptr;                                                                                            \
    if (inputs.size() > 2) {                                                                                                   \
        bias    = inputs[2];                                                                                                   \
        biasPtr = bias->host<float>();                                                                                         \
    }                                                                                                                          \
    auto output      = outputs[0];                                                                                             \
    auto width       = output->width();                                                                                        \
    auto height      = output->height();                                                                                       \
    int threadNumber = ((CPUBackend *)backend())->threadNumber();                                                              \
                                                                                                                               \
    GENERATE_WEIGHT();                                                                                                         \
                                                                                                                               \
    auto src_width                = input->width();                                                                            \
    auto src_height               = input->height();                                                                           \
    int src_z_step                = input->width() * input->height() * unit;                                                   \
    auto CONVOLUTION_TILED_NUMBER = eP;                                                                                        \
    auto icC4                     = UP_DIV(input->channel(), unit);                                                            \
    auto ic                       = input->channel();                                                                          \
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();                                              \
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();                                                   \
                                                                                                                               \
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();                                                   \
    mTempBufferTranspose.buffer().dimensions    = 2;                                                                           \
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;                                                                \
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * CONVOLUTION_TILED_NUMBER * bytes;                       \
    TensorUtils::setLinearLayout(&mTempBufferTranspose);                                                                       \
                                                                                                                               \
    int tileCount = UP_DIV(width * height, CONVOLUTION_TILED_NUMBER);                                                          \
    int plane     = width * height;                                                                                            \
                                                                                                                               \
    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);                                        \
    if (!success) {                                                                                                            \
        return OUT_OF_MEMORY;                                                                                                  \
    }                                                                                                                          \
    auto outputChannel = output->channel();                                                                                    \
    auto oC4           = UP_DIV(outputChannel, unit);                                                                          \
    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();                                           \
    auto maxLine       = UP_DIV(CONVOLUTION_TILED_NUMBER, width) + 1;                                                          \
    auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));          \
    if (nullptr == tempPtr.first) {                                                                                            \
        return OUT_OF_MEMORY;                                                                                                  \
    }                                                                                                                          \
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);                                                       \
    bufferAlloc->free(tempPtr);                                                                                                \
    std::vector<size_t> parameters(6);                                                                                         \
    parameters[0]          = eP * bytes;                                                                                       \
    parameters[1]          = L;                                                                                                \
    parameters[2]          = outputChannel;                                                                                    \
    parameters[3]          = plane * unit * bytes;                                                                             \
    parameters[4]          = 0;                                                                                                \
    parameters[5]          = 0;                                                                                                \
    auto threadNumberFirst = std::min(threadNumber, tileCount);                                                                \
    auto postParameters    = getPostParameters();                                                                              \
    mFunction.first        = threadNumberFirst;                                                                                \
    auto strideX           = mCommon->strideX();                                                                               \
    auto strideY           = mCommon->strideY();                                                                               \
    auto dilateX           = mCommon->dilateX();                                                                               \
    auto dilateY           = mCommon->dilateY();                                                                               \
    auto padY              = mPadY;                                                                                            \
    auto padX              = mPadX;                                                                                            \
    auto kernel_width      = mCommon->kernelX();                                                                               \
    auto kernel_height     = mCommon->kernelY();                                                                               \
    if (src_width == 1 && width == 1 && height > 1) {                                                                          \
        /* Swap x, y*/                                                                                                         \
        width         = height;                                                                                                \
        height        = 1;                                                                                                     \
        padX          = mPadY;                                                                                                 \
        padY          = mPadX;                                                                                                 \
        strideX       = strideY;                                                                                               \
        strideY       = 1; /* Don't need stride */                                                                             \
        src_width     = src_height;                                                                                            \
        src_height    = 1;                                                                                                     \
        dilateX       = dilateY;                                                                                               \
        dilateY       = 1;                                                                                                     \
        kernel_width  = kernel_height;                                                                                         \
        kernel_height = 1;                                                                                                     \
    }                                                                                                                          \
                                                                                                                               \
    auto outputBatchStride = width * height * oC4 * unit;                                                                      \
    auto inputBatchStride  = src_width * src_height * icC4 * unit;                                                             \
    mFunction.second       = [=](int tId) {                                                                                    \
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;                   \
        auto srcPtr     = (float const **)((uint8_t *)tempPtr.first + tempPtr.second +                                   \
                                       tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));            \
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);                                                    \
                                                                                                                         \
        int32_t info[4];                                                                                                 \
        info[1] = src_width * src_height;                                                                                \
        info[2] = eP;                                                                                                    \
        info[3] = strideX;                                                                                               \
        for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {                                            \
            auto dstOrigin = output->host<uint8_t>() + batchIndex * outputBatchStride * bytes;                           \
            auto srcOrigin = input->host<uint8_t>() + batchIndex * inputBatchStride * bytes;                             \
            for (int x = (int)tId; x < tileCount; x += threadNumberFirst) {                                              \
                int start  = (int)x * CONVOLUTION_TILED_NUMBER;                                                          \
                int remain = plane - start;                                                                              \
                int xC     = remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : remain;                      \
                /* Compute Pack position */                                                                              \
                int oyBegin   = start / width;                                                                           \
                int oxBegin   = start % width;                                                                           \
                int oyEnd     = (start + xC - 1) / width;                                                                \
                remain        = xC;                                                                                      \
                int number    = 0;                                                                                       \
                bool needZero = false;                                                                                   \
                int eStart    = 0;                                                                                       \
                for (int oy = oyBegin; oy <= oyEnd; ++oy) {                                                              \
                    int step    = std::min(width - oxBegin, remain);                                                     \
                    int sySta   = oy * strideY - padY;                                                                   \
                    int kyStart = std::max(0, UP_DIV(-sySta, dilateY));                                                  \
                    int kyEnd   = std::min(kernel_height, UP_DIV(src_height - sySta, dilateY));                          \
                    if (kyEnd - kyStart < kernel_height) {                                                               \
                        needZero = true;                                                                                 \
                    }                                                                                                    \
                    for (int ky = kyStart; ky < kyEnd; ++ky) {                                                           \
                        auto lKYOffset = ky * kernel_width * ic;                                                         \
                        auto srcKy     = srcOrigin + (sySta + ky * dilateY) * src_width * bytes * unit;                  \
                        for (int kx = 0; kx < kernel_width; ++kx) {                                                      \
                            /* Compute x range:*/                                                                        \
                            /* 0 <= (oxBegin + x) * strideX - padX + dilateX * kx < src_width*/                          \
                            /* 0 <= x <= step*/                                                                          \
                            int end = std::min(                                                                          \
                                step, (src_width - oxBegin * strideX - dilateX * kx + padX + strideX - 1) / strideX);    \
                            int sta = std::max(0, UP_DIV((padX - oxBegin * strideX - dilateX * kx), strideX));           \
                            if (end - sta < step) {                                                                      \
                                needZero = true;                                                                         \
                            }                                                                                            \
                            if (end > sta) {                                                                             \
                                auto lOffset = lKYOffset + (kx * ic);                                                    \
                                auto srcKx   = srcKy + ((oxBegin + sta) * strideX + dilateX * kx - padX) * bytes * unit; \
                                srcPtr[number]     = (const float *)srcKx;                                               \
                                el[4 * number + 0] = end - sta;                                                          \
                                el[4 * number + 1] = ic;                                                                 \
                                el[4 * number + 2] = eStart + sta;                                                       \
                                el[4 * number + 3] = lOffset;                                                            \
                                /* MNN_PRINT("e:%d, l:%d, eoffset:%d, loffset:%d\n",*/                                   \
                                /*     el[4 * number + 0],*/                                                             \
                                /*     el[4 * number + 1],*/                                                             \
                                /*     el[4 * number + 2],*/                                                             \
                                /*     el[4 * number + 3]*/                                                              \
                                /*     );*/                                                                              \
                                number++;                                                                                \
                            }                                                                                            \
                        }                                                                                                \
                    }                                                                                                    \
                    oxBegin = 0;                                                                                         \
                    remain -= step;                                                                                      \
                    eStart += step;                                                                                      \
                }                                                                                                        \
                info[0] = number;                                                                                        \
                if (needZero || lP != 1) {                                                                               \
                    ::memset(gemmBuffer, 0, mTempBufferTranspose.stride(0));                                             \
                }                                                                                                        \
                if (number > 0) {                                                                                        \
                    packA((float *)gemmBuffer, srcPtr, info, el);                                                        \
                }                                                                                                        \
                GENERATE_MM();                                                                                           \
            }                                                                                                            \
        }                                                                                                                \
    };                                                                                                                   \
    return NO_ERROR;

#undef GENERATE_FUNCTOR
#undef GENERATE_WEIGHT
#undef GENERATE_MM


} // namespace MNN

#endif /* ConvolutionTiledExecutor_hpp */
