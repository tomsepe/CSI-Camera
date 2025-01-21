# CSI-Camera
Simple example of using a MIPI-CSI(2) Camera (like the Raspberry Pi Version 2 camera) with the NVIDIA Jetson Developer Kits with CSI camera ports. This includes the recent Jetson Nano and Jetson Xavier NX. This is support code for the article on JetsonHacks: https://wp.me/p7ZgI9-19v

## Quick Start
1. Connect your CSI camera
2. Test basic functionality:
```bash
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```
3. Run simple camera example:
```bash
python3 simple_camera.py
```
4. Try face detection:
```bash
python3 face_detect.py
```

For GPU-accelerated versions, see TensorRT Setup below.

## Camera Setup
For the Nanos and Xavier NX, the camera should be installed in the MIPI-CSI Camera Connector on the carrier board. The pins on the camera ribbon should face the Jetson module, the tape stripe faces outward.

Some Jetson developer kits have two CSI camera slots. You can use the sensor_mode attribute with the GStreamer nvarguscamerasrc element to specify which camera. Valid values are 0 or 1 (the default is 0 if not specified).

## Requirements

- NVIDIA Jetson Developer Kit (tested on Nano, Xavier NX)
- CSI Camera (Raspberry Pi v2 or similar)
- OpenCV 4.1.1+ (pre-installed with JetPack)
- Python 3.6+

## Basic Camera Test

```bash
# Simple Test
#  Ctrl^C to exit
# sensor_id selects the camera: 0 or 1 on Jetson Nano B01
gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink

# More specific - width, height and framerate are from supported video modes
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=1280,height=720' ! nvvidconv ! nvegltransform ! nveglglessink -e

# Fullscreen without window decorations
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=1024, height=600, framerate=30/1, format=NV12' ! \
    nvvidconv flip-method=2 ! \
    'video/x-raw, width=1024, height=600' ! \
    nvvidconv ! \
    nvegltransform ! \
    nveglglessink fullscreen=true window-width=1024 window-height=600 sync=false -e
```

The `fullscreen=true` parameter in nveglglessink forces true fullscreen mode. Add `sync=false` for potentially better performance.

## Programs

### simple_camera.py
Basic CSI camera implementation using GStreamer pipeline. This serves as the foundation for the other programs, providing the camera interface code:
```bash
python3 simple_camera.py
```

### face_detect.py
Face detection implementation using OpenCV's Haar Cascade classifier. Features:
- Real-time face detection
- Rectangle drawing around detected faces
- FPS counter
- 720p resolution support

```bash
python3 face_detect.py
```

### dual_camera.py
For Jetson boards with two CSI-MIPI camera ports. Features:
- Reads from both CSI cameras simultaneously
- Displays both feeds in one window (1920x540)
- Multi-threaded for better performance

Note: Requires numpy:
```bash
pip3 install numpy
python3 dual_camera.py
```

### Advanced Face Detection Programs
The repository now includes enhanced versions of face detection:

#### face_tracker_zoom.py
Enhanced version with automatic zooming:
- Automatic zoom on detected faces
- Smooth transitions for zoom and pan
- Center tracking of faces
- Position smoothing to reduce jitter

#### face_tracker_gpu.py
GPU-accelerated version:
- CUDA-accelerated image processing
- GPU memory optimization
- Asynchronous operations
- Improved performance over CPU version

#### face_tracker_gpu_tensor.py
TensorRT-optimized version:
- TensorRT-accelerated face detection
- Highest performance of all versions
- Real-time inference optimization

## Building C++ Examples

### simple_camera.cpp
Can be built using either method:

1. Using g++ directly:
```bash
g++ -std=c++11 simple_camera.cpp -o simple_camera \
    `pkg-config --cflags --libs opencv4` \
    -I/usr/include/opencv4
```

2. Using CMake (recommended):
```cmake
cmake_minimum_required(VERSION 3.10)
project(simple_camera)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(simple_camera simple_camera.cpp)
target_link_libraries(simple_camera ${OpenCV_LIBS})
```

Then build:
```bash
mkdir build
cd build
cmake ..
make
```

## Performance Notes

Approximate FPS on Jetson Nano (may vary based on conditions):
- face_detect.py: 15-20 FPS
- face_tracker_zoom.py: 12-15 FPS
- face_tracker_gpu.py: 20-25 FPS
- face_tracker_gpu_tensor.py: 30-35 FPS

## Camera Image Formats
You can use v4l2-ctl to determine the camera capabilities:
```bash
sudo apt-get install v4l-utils
v4l2-ctl --list-formats-ext
```

## GStreamer Parameters
The nvvidconv flip-method parameter can rotate/flip the image:
```
flip-method         : video flip methods
                        flags: readable, writable, controllable
                        Enum "GstNvVideoFlipMethod" Default: 0, "none"
                           (0): none             - Identity (no rotation)
                           (1): counterclockwise - Rotate counter-clockwise 90 degrees
                           (2): rotate-180       - Rotate 180 degrees
                           (3): clockwise        - Rotate clockwise 90 degrees
                           (4): horizontal-flip  - Flip horizontally
                           (5): upper-right-diagonal - Flip across upper right/lower left diagonal
                           (6): vertical-flip    - Flip vertically
                           (7): upper-left-diagonal - Flip across upper left/low
```

## Troubleshooting

If you encounter issues:
1. Verify camera connection
2. Check OpenCV CUDA support: `python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"`
3. Monitor system temperature: `tegrastats`
4. Ensure adequate power supply (5V/4A recommended)

For TensorRT issues:
1. Verify TensorRT installation
2. Monitor GPU memory: `watch -n 0.5 nvidia-smi`

## Release Notes

v3.3 Release January, 2025
* Added advanced face detection implementations
* TensorRT optimization support
* Improved documentation and examples
* Performance optimizations

v3.2 Release January, 2022
* Add Exception handling to Python code
* Faster GStreamer pipelines, better performance
* Better naming of variables, simplification
* Remove Instrumented examples
* L4T 32.6.1 (JetPack 4.5)
* OpenCV 4.4.1
* Python3
* Tested on Jetson Nano B01, Jetson Xavier NX
* Tested with Raspberry Pi V2 cameras

v3.11 Release April, 2020
* Release both cameras in dual camera example (bug-fix)

v3.1 Release March, 2020
* L4T 32.3.1 (JetPack 4.3)
* OpenCV 4.1.1
* Tested on Jetson Nano B01
* Tested with Raspberry Pi v2 cameras

v3.0 December 2019
* L4T 32.3.1
* OpenCV 4.1.1.
* Tested with Raspberry Pi v2 camera

v2.0 Release September, 2019
* L4T 32.2.1 (JetPack 4.2.2)
* OpenCV 3.3.1
* Tested on Jetson Nano

Initial Release (v1.0) March, 2019
* L4T 32.1.0 (JetPack 4.2)
* Tested on Jetson Nano


