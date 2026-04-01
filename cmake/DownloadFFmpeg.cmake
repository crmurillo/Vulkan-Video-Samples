# DownloadFFmpeg.cmake
#
# Automatically downloads pre-built FFmpeg shared libraries for Windows
# from https://github.com/BtbN/FFmpeg-Builds/releases
#
# This module is only active on WIN32 builds when the FFmpeg libraries
# are not already present in the expected location.

if((CMAKE_GENERATOR_PLATFORM MATCHES "^(aarch64|arm64|ARM64)"))
    set(FFMPEG_ARCHIVE_NAME "ffmpeg-master-latest-winarm64-lgpl-shared")
    set(FFMPEG_PLATFORM_DIR "winarm64")
else()
    set(FFMPEG_ARCHIVE_NAME "ffmpeg-master-latest-win64-lgpl-shared")
    set(FFMPEG_PLATFORM_DIR "win64")
endif()

set(FFMPEG_DEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vk_video_decoder/bin/libs/ffmpeg/${FFMPEG_PLATFORM_DIR}")
set(FFMPEG_DOWNLOAD_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FFMPEG_ARCHIVE_NAME}.zip")
set(FFMPEG_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ffmpeg-download")
set(FFMPEG_ARCHIVE_PATH "${FFMPEG_DOWNLOAD_DIR}/${FFMPEG_ARCHIVE_NAME}.zip")

# Check if FFmpeg libraries are already present
file(GLOB FFMPEG_EXISTING_LIBS "${FFMPEG_DEST_DIR}/lib/avcodec*.lib")
if(FFMPEG_EXISTING_LIBS)
    message(STATUS "FFmpeg libraries already present in ${FFMPEG_DEST_DIR}")
    return()
endif()

message(STATUS "FFmpeg libraries not found in ${FFMPEG_DEST_DIR}")

file(MAKE_DIRECTORY "${FFMPEG_DOWNLOAD_DIR}")

# Download checksums and extract the expected hash for our archive
set(FFMPEG_CHECKSUMS_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256")
set(FFMPEG_CHECKSUMS_PATH "${FFMPEG_DOWNLOAD_DIR}/checksums.sha256")

file(DOWNLOAD "${FFMPEG_CHECKSUMS_URL}" "${FFMPEG_CHECKSUMS_PATH}" STATUS FFMPEG_CHECKSUM_STATUS TLS_VERIFY ON)
list(GET FFMPEG_CHECKSUM_STATUS 0 FFMPEG_CHECKSUM_RESULT)
if(NOT FFMPEG_CHECKSUM_RESULT EQUAL 0)
    message(WARNING "Failed to download FFmpeg checksums. Proceeding without hash verification.")
    set(FFMPEG_HASH_ARG "")
else()
    file(STRINGS "${FFMPEG_CHECKSUMS_PATH}" FFMPEG_CHECKSUM_LINES)
    set(FFMPEG_EXPECTED_HASH "")
    foreach(LINE ${FFMPEG_CHECKSUM_LINES})
        if(LINE MATCHES "^([0-9a-f]+)  ${FFMPEG_ARCHIVE_NAME}\\.zip$")
            set(FFMPEG_EXPECTED_HASH "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(FFMPEG_EXPECTED_HASH)
        set(FFMPEG_HASH_ARG "EXPECTED_HASH" "SHA256=${FFMPEG_EXPECTED_HASH}")
    else()
        message(WARNING "Could not find checksum for ${FFMPEG_ARCHIVE_NAME}.zip. Proceeding without hash verification.")
        set(FFMPEG_HASH_ARG "")
    endif()
endif()

message(STATUS "Downloading FFmpeg from ${FFMPEG_DOWNLOAD_URL} ...")

file(DOWNLOAD
    "${FFMPEG_DOWNLOAD_URL}"
    "${FFMPEG_ARCHIVE_PATH}"
    STATUS FFMPEG_DOWNLOAD_STATUS
    TLS_VERIFY ON
    ${FFMPEG_HASH_ARG}
)

list(GET FFMPEG_DOWNLOAD_STATUS 0 FFMPEG_DOWNLOAD_RESULT)
list(GET FFMPEG_DOWNLOAD_STATUS 1 FFMPEG_DOWNLOAD_ERROR)

if(NOT FFMPEG_DOWNLOAD_RESULT EQUAL 0)
    message(WARNING "Failed to download FFmpeg: ${FFMPEG_DOWNLOAD_ERROR}")
    message(WARNING "Please download FFmpeg manually from ${FFMPEG_DOWNLOAD_URL}")
    return()
endif()

message(STATUS "Extracting FFmpeg archive ...")

# Extract the archive to a temporary location
set(FFMPEG_EXTRACT_DIR "${FFMPEG_DOWNLOAD_DIR}/extract")
file(MAKE_DIRECTORY "${FFMPEG_EXTRACT_DIR}")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf "${FFMPEG_ARCHIVE_PATH}"
    WORKING_DIRECTORY "${FFMPEG_EXTRACT_DIR}"
    RESULT_VARIABLE FFMPEG_EXTRACT_RESULT
)

if(NOT FFMPEG_EXTRACT_RESULT EQUAL 0)
    message(WARNING "Failed to extract FFmpeg archive")
    return()
endif()

# The archive extracts to a directory named like the archive (without .zip)
set(FFMPEG_EXTRACTED_DIR "${FFMPEG_EXTRACT_DIR}/${FFMPEG_ARCHIVE_NAME}")

if(NOT EXISTS "${FFMPEG_EXTRACTED_DIR}")
    file(GLOB FFMPEG_EXTRACTED_DIRS "${FFMPEG_EXTRACT_DIR}/ffmpeg-*")
    list(LENGTH FFMPEG_EXTRACTED_DIRS FFMPEG_NUM_DIRS)
    if(FFMPEG_NUM_DIRS EQUAL 1)
        list(GET FFMPEG_EXTRACTED_DIRS 0 FFMPEG_EXTRACTED_DIR)
    else()
        message(WARNING "Could not find extracted FFmpeg directory in ${FFMPEG_EXTRACT_DIR}")
        return()
    endif()
endif()

# Copy bin/, lib/, include/ into the destination
foreach(SUBDIR bin lib include)
    if(EXISTS "${FFMPEG_EXTRACTED_DIR}/${SUBDIR}")
        file(GLOB FFMPEG_SUBDIR_FILES "${FFMPEG_EXTRACTED_DIR}/${SUBDIR}/*")
        foreach(FILE_PATH ${FFMPEG_SUBDIR_FILES})
            get_filename_component(FNAME "${FILE_PATH}" NAME)
            if(NOT FNAME STREQUAL ".gitkeep")
                file(COPY "${FILE_PATH}" DESTINATION "${FFMPEG_DEST_DIR}/${SUBDIR}")
            endif()
        endforeach()
    endif()
endforeach()

file(GLOB FFMPEG_VERIFY_LIBS "${FFMPEG_DEST_DIR}/lib/avcodec*.lib")
if(FFMPEG_VERIFY_LIBS)
    message(STATUS "FFmpeg libraries successfully downloaded and installed to ${FFMPEG_DEST_DIR}")
else()
    message(WARNING "FFmpeg download completed but libraries not found in ${FFMPEG_DEST_DIR}/lib/")
    message(WARNING "You may need to download and extract FFmpeg manually")
endif()

file(REMOVE_RECURSE "${FFMPEG_DOWNLOAD_DIR}")
