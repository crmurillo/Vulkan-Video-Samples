#
# Find the native FFmpeg includes and libraries.
#
# This module defines:
#   FFMPEG_FOUND                   - True if FFmpeg was found
#   FFMPEG_INCLUDE_DIRS            - FFmpeg include directories
#   FFMPEG_LIBRARIES               - All FFmpeg libraries
#   FFMPEG_LIBAVCODEC_LIBRARIES    - avcodec library
#   FFMPEG_LIBAVFORMAT_LIBRARIES   - avformat library
#   FFMPEG_LIBAVUTIL_LIBRARIES     - avutil library
#   FFMPEG_LIBSWSCALE_LIBRARIES    - swscale library (optional)
#
# Accepts:
#   FFMPEG_ROOT or ENV{FFMPEG_DIR} or ENV{FFMPEG_ROOT} - Custom search path
#

set(FFMPEG_ROOT "$ENV{FFMPEG_DIR}" CACHE PATH "Location of FFmpeg")

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_FFMPEG QUIET
        libavcodec
        libavformat
        libavutil
        libswscale
    )
endif()

# Windows FFmpeg paths
if(WIN32)
    set(FFMPEG_WIN32_PREBUILT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vk_video_decoder/bin/libs/ffmpeg")
    if((CMAKE_GENERATOR_PLATFORM MATCHES "^(aarch64|arm64|ARM64)"))
        set(FFMPEG_WIN32_PREBUILT_DIR "${FFMPEG_WIN32_PREBUILT_ROOT}/winarm64")
    elseif((CMAKE_GENERATOR_PLATFORM MATCHES "^(arm|ARM)"))
        set(FFMPEG_WIN32_PREBUILT_DIR "${FFMPEG_WIN32_PREBUILT_ROOT}/winarm")
    else()
        set(FFMPEG_WIN32_PREBUILT_DIR "${FFMPEG_WIN32_PREBUILT_ROOT}/win64")
    endif()
endif()

# Common include search paths
set(FFMPEG_INCLUDE_SEARCH_PATHS
    $ENV{FFMPEG_DIR}/include
    $ENV{FFMPEG_ROOT}/include
    ${FFMPEG_ROOT}/include
    ${FFMPEG_WIN32_PREBUILT_DIR}/include
    ${PC_FFMPEG_INCLUDE_DIRS}
    /Library/Frameworks
    ~/Library/Frameworks
    /opt/csw/include       # Blastwave
    /opt/include
    /opt/local/include     # DarwinPorts
    /sw/include            # Fink
    /usr/freeware/include
    /usr/include
    /usr/local/include
)

# Find include directories per component
find_path(FFMPEG_LIBAVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS}
    PATH_SUFFIXES ffmpeg
)

find_path(FFMPEG_LIBAVFORMAT_INCLUDE_DIR
    NAMES libavformat/avformat.h
    PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS}
    PATH_SUFFIXES ffmpeg
)

find_path(FFMPEG_LIBAVUTIL_INCLUDE_DIR
    NAMES libavutil/avutil.h
    PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS}
    PATH_SUFFIXES ffmpeg
)

# Common library search paths
set(FFMPEG_LIB_SEARCH_PATHS
    $ENV{FFMPEG_DIR}/lib
    $ENV{FFMPEG_ROOT}/lib
    ${FFMPEG_ROOT}/lib
    ${FFMPEG_WIN32_PREBUILT_DIR}/lib
    ${PC_FFMPEG_LIBRARY_DIRS}
    /Library/Frameworks
    ~/Library/Frameworks
    /opt/csw/lib       # Blastwave
    /opt/lib
    /opt/local/lib     # DarwinPorts
    /sw/lib            # Fink
    /usr/freeware/lib64
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
)

# Find required libraries
find_library(FFMPEG_LIBAVCODEC_LIBRARIES
    NAMES avcodec
    PATHS ${FFMPEG_LIB_SEARCH_PATHS}
)

find_library(FFMPEG_LIBAVFORMAT_LIBRARIES
    NAMES avformat
    PATHS ${FFMPEG_LIB_SEARCH_PATHS}
)

find_library(FFMPEG_LIBAVUTIL_LIBRARIES
    NAMES avutil
    PATHS ${FFMPEG_LIB_SEARCH_PATHS}
)

# Find optional libraries
find_library(FFMPEG_LIBSWSCALE_LIBRARIES
    NAMES swscale
    PATHS ${FFMPEG_LIB_SEARCH_PATHS}
)

# All required result variables
set(FFMPEG_REQUIRED_VARS
    FFMPEG_LIBAVCODEC_INCLUDE_DIR
    FFMPEG_LIBAVFORMAT_INCLUDE_DIR
    FFMPEG_LIBAVUTIL_INCLUDE_DIR
    FFMPEG_LIBAVCODEC_LIBRARIES
    FFMPEG_LIBAVFORMAT_LIBRARIES
    FFMPEG_LIBAVUTIL_LIBRARIES
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS ${FFMPEG_REQUIRED_VARS}
)

mark_as_advanced(
    ${FFMPEG_REQUIRED_VARS}
    FFMPEG_LIBSWSCALE_LIBRARIES
)

if(FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIRS
        ${FFMPEG_LIBAVCODEC_INCLUDE_DIR}
        ${FFMPEG_LIBAVFORMAT_INCLUDE_DIR}
        ${FFMPEG_LIBAVUTIL_INCLUDE_DIR}
    )
    list(REMOVE_DUPLICATES FFMPEG_INCLUDE_DIRS)
    set(FFMPEG_LIBRARIES
        ${FFMPEG_LIBAVCODEC_LIBRARIES}
        ${FFMPEG_LIBAVFORMAT_LIBRARIES}
        ${FFMPEG_LIBAVUTIL_LIBRARIES}
    )
    if(FFMPEG_LIBSWSCALE_LIBRARIES)
        list(APPEND FFMPEG_LIBRARIES ${FFMPEG_LIBSWSCALE_LIBRARIES})
    endif()
endif() # FFMPEG_FOUND
