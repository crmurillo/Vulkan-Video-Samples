/*
* Copyright 2023 NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <sstream>
#include <fstream>
#include "VkDecoderUtils/VideoStreamDemuxer.h"


VkResult VideoStreamDemuxer::Create(const char *pFilePath,
                                    VkVideoCodecOperationFlagBitsKHR codecType,
                                    bool requiresStreamDemuxing,
                                    int32_t defaultWidth,
                                    int32_t defaultHeight,
                                    int32_t defaultBitDepth,
                                    VkSharedBaseObj<VideoStreamDemuxer>& videoStreamDemuxer)
{
#ifdef FFMPEG_DEMUXER_SUPPORT
    if (requiresStreamDemuxing || (codecType == VK_VIDEO_CODEC_OPERATION_NONE_KHR)) {
        return FFmpegDemuxerCreate(pFilePath,
                                   codecType,
                                   requiresStreamDemuxing,
                                   defaultWidth,
                                   defaultHeight,
                                   defaultBitDepth,
                                   videoStreamDemuxer);
    }  else
#endif // FFMPEG_DEMUXER_SUPPORT
    {
        if (codecType == VK_VIDEO_CODEC_OPERATION_NONE_KHR) {
            fprintf(stderr, "Error: No video codec specified for %s.\n", pFilePath);
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        if (defaultWidth <= 0 || defaultHeight <= 0) {
            fprintf(stderr, "Error: Invalid video dimensions %dx%d for %s.\n",
                    defaultWidth, defaultHeight, pFilePath);
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        if (defaultBitDepth != 8 && defaultBitDepth != 10 && defaultBitDepth != 12) {
            fprintf(stderr, "Error: Unsupported bit depth %d for %s. Must be 8, 10, or 12.\n",
                    defaultBitDepth, pFilePath);
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        return ElementaryStreamCreate(pFilePath,
                                      codecType,
                                      defaultWidth,
                                      defaultHeight,
                                      defaultBitDepth,
                                      videoStreamDemuxer);
    }
}



