import cv2
import numpy as np
import os
from simple_camera import gstreamer_pipeline
import time

def face_tracker():
    # Initialize face detection classifier
    cascade_paths = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    ]
    
    cascade_file = None
    for path in cascade_paths:
        if os.path.exists(path):
            cascade_file = path
            break
    
    if cascade_file is None:
        print("Error: Could not find haar cascade file")
        return
        
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    window_title = "Face Tracker"
    
    # Initialize camera with lower resolution
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=1280,
            capture_height=720,
            display_width=1280,
            display_height=720,
            framerate=60,
            flip_method=2
        ),
        cv2.CAP_GSTREAMER
    )
    
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
            # Variables for FPS calculation
            frame_count = 0
            fps = 0
            fps_update_interval = 1.0  # Update FPS every second
            last_fps_update = time.time()
            
            # Variables for face detection optimization
            process_this_frame = True
            last_faces = []
            
            while True:
                start_time = time.time()
                
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break
                
                # Debug information
                if frame_count == 0:  # Print only once
                    print(f"Frame shape: {frame.shape}")
                    print(f"Frame dtype: {frame.dtype}")
                    print(f"Number of channels: {1 if len(frame.shape) == 2 else frame.shape[2]}")
                
                # Only process every other frame
                if process_this_frame:
                    # Resize frame for faster face detection
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    
                    # Check number of channels and convert to grayscale only if needed
                    if len(small_frame.shape) == 3:  # Color image (3 channels)
                        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    else:  # Already grayscale
                        gray = small_frame
                    
                    # Detect faces with adjusted parameters
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=4,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Convert face coordinates back to original size
                    last_faces = [(int(x * 2), int(y * 2), int(w * 2), int(h * 2)) for (x, y, w, h) in faces]
                
                # Draw rectangle around faces using the last known positions
                for (x, y, w, h) in last_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"Face: ({x+w//2}, {y+h//2})"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if time.time() - last_fps_update >= fps_update_interval:
                    fps = frame_count / (time.time() - last_fps_update)
                    frame_count = 0
                    last_fps_update = time.time()
                
                # Display FPS on frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Toggle frame processing
                process_this_frame = not process_this_frame
                
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
    face_tracker()
