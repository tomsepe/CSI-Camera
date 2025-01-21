import cv2
import numpy as np
import os
from simple_camera import gstreamer_pipeline
import time
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
import platform

def face_tracker_tpu():
    # Check if Edge TPU is available
    try:
        # Initialize Edge TPU with face detection model
        # Using Google's pre-trained face detection model
        model_paths = [
            'models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite',
            os.path.expanduser('~/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
        ]
        
        model_file = None
        for path in model_paths:
            if os.path.exists(path):
                model_file = path
                break
                
        if model_file is None:
            print("Error: Could not find Edge TPU model file")
            print("Please download the model using setup_models.sh")
            return
            
        interpreter = make_interpreter(model_file)
        interpreter.allocate_tensors()
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get model input size
        input_shape = input_details[0]['shape']
        input_size = (input_shape[2], input_shape[1])  # (width, height)
        
    except Exception as e:
        print(f"Error initializing Edge TPU: {str(e)}")
        print("Please ensure the Coral USB Accelerator is connected and drivers are installed")
        return
    
    window_title = "Face Tracker (Edge TPU)"
    
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
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
            # Variables for FPS calculation
            frame_count = 0
            fps = 0
            fps_update_interval = 1.0
            last_fps_update = time.time()
            
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break
                
                # Prepare image for TPU
                # Resize to model input size
                input_tensor = cv2.resize(frame, input_size)
                # Convert to RGB (TPU models expect RGB)
                input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
                # Add batch dimension
                input_tensor = np.expand_dims(input_tensor, 0)
                
                # Run inference on Edge TPU
                common.set_input(interpreter, input_tensor)
                interpreter.invoke()
                
                # Get detection results
                faces = detect.get_objects(interpreter, score_threshold=0.4)
                
                # Scale detection boxes to original frame size
                scale_x = frame.shape[1] / input_size[0]
                scale_y = frame.shape[0] / input_size[1]
                
                # Draw results
                for face in faces:
                    bbox = face.bbox
                    x1 = int(bbox.xmin * scale_x)
                    y1 = int(bbox.ymin * scale_y)
                    x2 = int(bbox.xmax * scale_x)
                    y2 = int(bbox.ymax * scale_y)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence score
                    confidence = f"{face.score:.2f}"
                    cv2.putText(frame, f"Face: {confidence}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add face center coordinates
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.putText(frame, f"({center_x}, {center_y})", (x1, y2+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and display FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    fps = frame_count / (current_time - last_fps_update)
                    frame_count = 0
                    last_fps_update = current_time
                
                # Display FPS on frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Check if window is still open
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                
                # Stop the program on the ESC key or 'q'
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
                    
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    face_tracker_tpu() 