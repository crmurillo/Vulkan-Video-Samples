#
# Find the native FFmpeg includes and libraries.
#
# This module defines:
#   FFMPEG_FOUND                   - True if FFmpeg was found
#   FFMPEG_INCLUDE_DIR             - FFmpeg include directory
#   FFMPEG_INCLUDE_DIRS            - Same as above (alias)
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

# Find include directory
find_path(FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h libavformat/avformat.h libavutil/avutil.h
    PATHS
        ${FFMPEG_ROOT}/include
        $ENV{FFMPEG_ROOT}/include
        ${PC_FFMPEG_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
    PATH_SUFFIXES ffmpeg
)

# Find required libraries
find_library(FFMPEG_LIBAVCODEC_LIBRARIES
    NAMES avcodec
    PATHS
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
        ${PC_FFMPEG_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

find_library(FFMPEG_LIBAVFORMAT_LIBRARIES
    NAMES avformat
    PATHS
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
        ${PC_FFMPEG_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

find_library(FFMPEG_LIBAVUTIL_LIBRARIES
    NAMES avutil
    PATHS
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
        ${PC_FFMPEG_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

# Find optional libraries
find_library(FFMPEG_LIBSWSCALE_LIBRARIES
    NAMES swscale
    PATHS
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
        ${PC_FFMPEG_LIBRARY_DIRS}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS
        FFMPEG_INCLUDE_DIR
        FFMPEG_LIBAVCODEC_LIBRARIES
        FFMPEG_LIBAVFORMAT_LIBRARIES
        FFMPEG_LIBAVUTIL_LIBRARIES
)

mark_as_advanced(
    FFMPEG_INCLUDE_DIR
    FFMPEG_LIBAVCODEC_LIBRARIES
    FFMPEG_LIBAVFORMAT_LIBRARIES
    FFMPEG_LIBAVUTIL_LIBRARIES
    FFMPEG_LIBSWSCALE_LIBRARIES
)

if(FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
    set(FFMPEG_LIBRARIES
        ${FFMPEG_LIBAVCODEC_LIBRARIES}
        ${FFMPEG_LIBAVFORMAT_LIBRARIES}
        ${FFMPEG_LIBAVUTIL_LIBRARIES}
    )
    if(FFMPEG_LIBSWSCALE_LIBRARIES)
        list(APPEND FFMPEG_LIBRARIES ${FFMPEG_LIBSWSCALE_LIBRARIES})
    endif()
endif() # FFMPEG_FOUND
