#!/bin/bash

# Create models directory
mkdir -p ~/models
cd ~/models

# Download face detection model
wget https://github.com/google-coral/test_data/raw/master/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

echo "Models downloaded successfully" 