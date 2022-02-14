//
//  CPUTensorConvert.cpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUTensorConvert.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"

namespace MNN {

static void _NC4HW42NHWCUint8(const uint8_t* source, uint8_t* dest, int b, int c, int area) {
    int sourceBatchsize = ALIGN_UP4(c) * area;
    int destBatchSize   = c * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNPackTransposeUint8(dstBatch, srcBatch, area, c);
    }
}

static void _NC4HW42NHWCInt16(const int16_t* source, int16_t* dest, int b, int c, int area) {
    int sourceBatchsize = ALIGN_UP4(c) * area;
    int destBatchSize   = c * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNPackTransposeInt16(dstBatch, srcBatch, area, c);
    }
}

static void _NHWC2NC4HW4Uint8(const uint8_t* source, uint8_t* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNUnpackTransposeUint8(dstBatch, srcBatch, area, c);
    }
}
static void _NHWC2NC4HW4Int16(const int16_t* source, int16_t* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNUnpackTransposeInt16(dstBatch, srcBatch, area, c);
    }
}

static void NC4HW42NHWC(const float* source, float* dest, int b, int c, int area) {
    int sourceBatchsize = ALIGN_UP4(c) * area;
    int destBatchSize   = c * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNPackTranspose(dstBatch, srcBatch, area, c);
    }
}

static void NHWC2NC4HW4(const float* source, float* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNUnpackTranspose(dstBatch, srcBatch, area, c);
    }
}

template<typename T>
void NCHW2NHWC(const T* source, T* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i;
            auto dstArea = dstBatch + i * c;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci] = srcArea[ci * area];
            }
        }
    }
}

template<typename T>
void NHWC2NCHW(const T* source, T* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i * c;
            auto dstArea = dstBatch + i;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci * area] = srcArea[ci];
            }
        }
    }
}

ErrorCode CPUTensorConverter::convert(const void* inputRaw, void* outputRaw, MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest, int batch, int area, int channel, int bitLength, const CoreFunctions* core) {
    auto channelC4 = UP_DIV(channel, core->pack);
    auto batchStrideC4 = channelC4 * area * core->pack;
    auto batchStride = area * channel;

    // the case when source and dest data layout are the same
    // This case occurs in BackendTest of BF16 data.
    if(source == dest) {
        ::memcpy(outputRaw, inputRaw, batch * area * channel * bitLength);
        return NO_ERROR;
    }
    if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
        if (bitLength == 1) {
            for (int i = 0; i < batch; ++i) {
                MNNUnpackC4Uint8((uint8_t*)outputRaw + batchStride * i,
                                 (const uint8_t*)inputRaw + batchStrideC4 * i, area, channel);
            }
            return NO_ERROR;
        }
        if (bitLength == 2) {
            for (int i = 0; i < batch; ++i) {
                MNNUnpackC4Int16((int16_t*)outputRaw + batchStride * i,
                                 (const int16_t*)inputRaw + batchStrideC4 * i, area, channel);
            }
            return NO_ERROR;
        }
        for (int i = 0; i < batch; ++i) {
            core->MNNUnpackCUnit((float*)outputRaw + batchStride * i, (const float*)inputRaw + batchStrideC4 * i, area, channel);
        }
        return NO_ERROR;
    }

    if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        if (bitLength == 1) {
            for (int i = 0; i < batch; ++i) {
                MNNPackC4Uint8((uint8_t*)outputRaw + batchStrideC4 * i, (const uint8_t*)inputRaw + batchStride * i, area, channel);
            }
            return NO_ERROR;
        }
        if (bitLength == 2) {
            for (int i = 0; i < batch; ++i) {
                MNNPackC4Int16((int16_t*)outputRaw + batchStrideC4 * i, (const int16_t*)inputRaw + batchStride * i, area, channel);
            }
            return NO_ERROR;
        }
        for (int i = 0; i < batch; ++i) {
            core->MNNPackCUnit((float*)outputRaw + batchStrideC4 * i, (const float*)inputRaw + batchStride * i, area, channel);
        }
        return NO_ERROR;
    }

    if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        if (bitLength == 1) {
            _NHWC2NC4HW4Uint8((uint8_t*)inputRaw, (uint8_t*)outputRaw, batch, channel, area);
        } else if (bitLength == 2){
            _NHWC2NC4HW4Int16((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
        } else {
            for (int i = 0; i < batch; ++i) {
                core->MNNPackCUnitTranspose((float*)outputRaw + batchStrideC4 * i, (const float*)inputRaw + batchStride * i, area, channel);
            }
        }
    } else if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NHWC == dest) {
        if (bitLength == 1) {
            _NC4HW42NHWCUint8((uint8_t*)inputRaw, (uint8_t*)outputRaw, batch, channel, area);
        } else if (bitLength == 2){
            _NC4HW42NHWCInt16((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
        } else {
            for (int i = 0; i < batch; ++i) {
                core->MNNUnpackCUnitTranspose((float*)outputRaw + batchStride * i, (const float*)inputRaw + batchStrideC4 * i, area, channel);
            }
        }
    } else if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
        switch (bitLength) {
            case 1:
                NHWC2NCHW((int8_t*)inputRaw, (int8_t*)outputRaw, batch, channel, area);
                break;
            case 2:
                NHWC2NCHW((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
                break;
            case 4:
                NHWC2NCHW((float*)inputRaw, (float*)outputRaw, batch, channel, area);
                break;
            default:
                break;
        }
    } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
        switch (bitLength) {
            case 1:
                NCHW2NHWC((int8_t*)inputRaw, (int8_t*)outputRaw, batch, channel, area);
                break;
            case 2:
                NCHW2NHWC((int16_t*)inputRaw, (int16_t*)outputRaw, batch, channel, area);
                break;
            case 4:
                NCHW2NHWC((float*)inputRaw, (float*)outputRaw, batch, channel, area);
                break;
            default:
                break;
        }
    } else {
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

std::tuple<int, int, int> CPUTensorConverter::splitDimensions(const halide_buffer_t& ib, MNN_DATA_FORMAT source) {
    int area = 1, batch = ib.dim[0].extent, channel;
    if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
        channel = ib.dim[1].extent;
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= ib.dim[axis].extent;
        }
    } else {
        channel = ib.dim[ib.dimensions - 1].extent;
        for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
            area *= ib.dim[axis].extent;
        }
    }
    return std::make_tuple(batch, area, channel);
}
ErrorCode CPUTensorConverter::convert(const Tensor* input, const Tensor* output, const CoreFunctions* core) {
    auto ib     = input->buffer();
    auto ob     = output->buffer();
    auto source = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
    if (ib.dimensions <= 1 || source == dest) {
        ::memcpy(ob.host, ib.host, input->size());
        return NO_ERROR;
    }
    if (nullptr == core) {
        core = MNNGetCoreFunctions();
    }
    if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
        MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
        return INVALID_VALUE;
    }
    auto tup = splitDimensions(ib, source);
    int area = std::get<1>(tup), batch = std::get<0>(tup), channel = std::get<2>(tup);
    const int bitLength = ib.type.bytes();
    auto code = convert(ib.host, ob.host, source, dest, batch, area, channel, bitLength, core);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUTensorConver\n");
        return code;
    }
    return NO_ERROR;
}

} // namespace MNN
