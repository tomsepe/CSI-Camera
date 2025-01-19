# Jetson Nano Face Detection and Tracking

This project implements real-time face detection and tracking on the NVIDIA Jetson Nano using OpenCV and the onboard CSI camera. It includes multiple implementations with different features and optimizations.

## Requirements

- NVIDIA Jetson Nano (JetPack 4.6.1)
- CSI Camera (Raspberry Pi v2 or similar)
- OpenCV 4.1.1 (pre-installed with JetPack)
- Python 3.6+

## Programs

### simple_camera.py
Basic CSI camera implementation using GStreamer pipeline. This serves as the foundation for the other programs, providing the camera interface code.

### face_tracker.py
Basic face detection implementation using OpenCV's Haar Cascade classifier. Features:
- Real-time face detection
- Rectangle drawing around detected faces
- FPS counter
- Optimized for performance with frame skipping
- 720p resolution support

### face_tracker_zoom.py
Enhanced version of face_tracker.py that adds automatic zooming. Features:
- All features from face_tracker.py
- Automatic zoom on detected faces
- Smooth transitions for zoom and pan
- Center tracking of faces
- Position smoothing to reduce jitter
- Configurable zoom levels and smoothing factors

### face_tracker_gpu.py
GPU-accelerated version utilizing CUDA capabilities of the Jetson Nano. Features:
- CUDA-accelerated image processing
- GPU memory optimization
- Asynchronous operations using CUDA streams
- Improved performance over CPU version
- Prepared for TensorRT integration

### face_tracker_gpu_tensor.py
TensorRT-optimized version for maximum performance on Jetson Nano. Features:
- TensorRT-accelerated face detection using a pre-trained model
- CUDA-accelerated image processing
- Asynchronous operations using CUDA streams
- Highest performance of all versions
- Real-time inference optimization

## Setup

### Basic Setup
1. Ensure your CSI camera is properly connected to your Jetson Nano
2. Test basic camera functionality:
```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=1280,height=720' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

3. Run any of the programs:
```bash
python3 face_tracker.py
python3 face_tracker_zoom.py
python3 face_tracker_gpu.py
```

4. Press 'q' or 'ESC' to exit the program

### TensorRT Setup
1. Install required packages:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install numpy
```

2. Install TensorRT dependencies:
```bash
sudo apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    python3-libnvinfer \
    python3-libnvinfer-dev

# Verify TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

3. Download the pre-trained face detection model:
```bash
# Create models directory
mkdir -p ~/face_detection_models
cd ~/face_detection_models

# Download TensorRT-compatible face detection model
wget https://github.com/NVIDIA/TensorRT/raw/main/samples/python/detectnet_v2/specs/detectnet_v2_dynamic.onnx

# Convert ONNX model to TensorRT
trtexec --onnx=detectnet_v2_dynamic.onnx \
        --saveEngine=face_detector.trt \
        --workspace=1024 \
        --fp16
```

## Usage

1. Ensure your CSI camera is properly connected to your Jetson Nano
2. Test basic camera functionality:
```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=1280,height=720' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

3. Run any of the programs:
```bash
python3 face_tracker.py
python3 face_tracker_zoom.py
python3 face_tracker_gpu.py
python3 face_tracker_gpu_tensor.py
```

4. Press 'q' or 'ESC' to exit the program

## Performance Notes

- face_tracker.py: Basic implementation, suitable for most uses
- face_tracker_zoom.py: Slightly more CPU intensive due to zoom calculations
- face_tracker_gpu.py: Best performance, requires OpenCV built with CUDA support
- face_tracker_gpu_tensor.py: Highest performance, using TensorRT acceleration
  * Requires TensorRT setup
  * 2-3x faster than GPU-only version
  * More accurate detection than Haar Cascade

## Future Improvements

1. TensorRT Integration:
   - Replace Haar Cascade with TensorRT-optimized neural network
   - Implement face landmark detection
   - Add face recognition capabilities

2. Additional Features:
   - Multi-face tracking
   - Face recognition
   - Gesture recognition
   - Expression detection

## Troubleshooting

If you encounter issues:
1. Verify camera connection
2. Check OpenCV CUDA support: `python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"`
3. Monitor system temperature: `tegrastats`
4. Ensure adequate power supply (5V/4A recommended)

For TensorRT issues:
1. Verify TensorRT installation: 
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```
2. Check model conversion:
```bash
ls -l ~/face_detection_models/face_detector.trt
```
3. Monitor GPU memory: 
```bash
watch -n 0.5 nvidia-smi
```

## Performance Comparison

Approximate FPS on Jetson Nano (may vary based on conditions):
- face_tracker.py: 15-20 FPS
- face_tracker_zoom.py: 12-15 FPS
- face_tracker_gpu.py: 20-25 FPS
- face_tracker_gpu_tensor.py: 30-35 FPS

## License

This project uses code from various sources:
- Original CSI camera code: MIT License, Copyright (c) 2019-2022 JetsonHacks
- Modified and additional code: MIT License
