import cv2
import numpy as np
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from simple_camera import gstreamer_pipeline

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load the TensorRT engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for input and output
        self.input_shape = (1, 3, 300, 300)  # Batch, Channels, Height, Width
        self.output_shape = (1, 100, 7)  # Batch, Max detections, Detection info
        
        # Create GPU buffers and a stream
        self.stream = cuda.Stream()
        self.d_input = cuda.mem_alloc(self.input_shape[0] * self.input_shape[1] * 
                                    self.input_shape[2] * self.input_shape[3] * 
                                    np.dtype('float32').itemsize)
        self.d_output = cuda.mem_alloc(self.output_shape[0] * self.output_shape[1] * 
                                     self.output_shape[2] * np.dtype('float32').itemsize)
        
        # Create host buffers
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        
        # Store bindings
        self.bindings = [int(self.d_input), int(self.d_output)]

    def preprocess(self, img):
        # Resize to network input size
        resized = cv2.resize(img, (300, 300))
        # Convert to float and normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0
        # Transpose to CHW format
        transposed = normalized.transpose((2, 0, 1))
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        return batched

    def infer(self, img):
        # Preprocess the image
        preprocessed = self.preprocess(img)
        np.copyto(self.h_input, preprocessed)
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()
        
        return self.h_output

def face_tracker():
    # Initialize TensorRT face detector
    model_path = os.path.expanduser('~/face_detection_models/face_detector.trt')
    if not os.path.exists(model_path):
        print(f"Error: TensorRT model not found at {model_path}")
        print("Please follow the TensorRT setup instructions in the README")
        return
        
    detector = TensorRTInference(model_path)
    window_title = "Face Tracker (TensorRT)"
    
    # Initialize camera
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=1280,
            capture_height=720,
            display_width=1280,
            display_height=720,
            framerate=30,
            flip_method=0
        ),
        cv2.CAP_GSTREAMER
    )
    
    if video_capture.isOpened():
        try:
            # Initialize CUDA stream for OpenCV operations
            cuda_stream = cv2.cuda.Stream()
            gpu_frame = cv2.cuda.GpuMat()
            
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
            # FPS calculation variables
            frame_count = 0
            fps = 0
            fps_update_interval = 1.0
            last_fps_update = time.time()
            
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break
                
                # Upload frame to GPU for preprocessing
                gpu_frame.upload(frame, cuda_stream)
                
                # Run TensorRT inference
                detections = detector.infer(frame)
                
                # Process detections (confidence threshold = 0.5)
                for i in range(detections.shape[1]):
                    confidence = detections[0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, i, 3:7] * np.array([frame.shape[1], 
                                                              frame.shape[0], 
                                                              frame.shape[1], 
                                                              frame.shape[0]])
                        (x, y, x2, y2) = box.astype(int)
                        
                        # Draw detection box
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                        text = f"Face: {confidence:.2f}"
                        cv2.putText(frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if time.time() - last_fps_update >= fps_update_interval:
                    fps = frame_count / (time.time() - last_fps_update)
                    frame_count = 0
                    last_fps_update = time.time()
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
                    
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    face_tracker()
