# CSI-Camera
Simple example of using a MIPI-CSI(2) Camera (like the Raspberry Pi Version 2 camera) with the NVIDIA Jetson Developer Kits with CSI camera ports. This includes the recent Jetson Nano and Jetson Xavier NX. This is support code for the article on JetsonHacks: https://wp.me/p7ZgI9-19v

This branch is specifically for Ubuntu 20.04 with pre-installed:
- OpenCV 4.8.0 with CUDA support
- TensorFlow
- PyTorch 1.13.0
- TensorRT 8.0.1.6

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
5. For GPU-accelerated detection:
```bash
python3 face_tracker_gpu.py
```
6. For Edge TPU acceleration (requires Coral USB):
```bash
python3 face_tracker_tpu.py
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
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1024, height=600, framerate=30/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=1024,height=600' ! nvvidconv ! nvegltransform ! nveglglessink -e

# Fullscreen without window decorations
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=1024, height=600, framerate=30/1, format=NV12' ! \
    nvvidconv flip-method=2 ! \
    'video/x-raw, width=1024, height=600' ! \
    nvvidconv ! \
    nvegltransform ! \
    nveglglessink window-x=0 window-y=0 window-width=1024 window-height=600 -e
```

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
2. Monitor GPU memory: `

## GPU Acceleration Setup

For GPU-accelerated face detection (required for face_tracker_gpu.py):

1. First verify if OpenCV has CUDA support:
```bash
python3 -c "import cv2; print('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"
```

If the output shows CUDA is not enabled, you'll need to build OpenCV with CUDA support:

1. Install dependencies:
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    gfortran \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavresample-dev \
    libcanberra-gtk3-module \
    libdc1394-22-dev \
    libeigen3-dev \
    libglew-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtiff-dev \
    libv4l-dev \
    libxine2-dev \
    libxvidcore-dev \
    pkg-config \
    python3-dev \
    python3-numpy
```

2. Clone and build OpenCV with CUDA support:
```bash
# Clone OpenCV and OpenCV contrib
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.4.0
cd ../opencv_contrib
git checkout 4.4.0
cd ../opencv

# Create build directory
mkdir build
cd build

# Configure build with CUDA support
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=5.3 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_NEON=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF ..

# Build and install (this will take a while)
make -j4
sudo make install
sudo ldconfig

# Verify installation
python3 -c "import cv2; print('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"
```

After successful installation, you should be able to run the GPU-accelerated face detection:
```bash
python3 face_tracker_gpu.py
```

## Edge TPU Setup

To use the Coral USB Edge TPU accelerator:

1. Install Edge TPU runtime and libraries:
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y python3-pycoral
```

2. Download required models:
```bash
chmod +x setup_models.sh
./setup_models.sh
```

3. Run the Edge TPU version:
```bash
python3 face_tracker_tpu.py
```

Note: Make sure your Coral USB Accelerator is connected before running the program.