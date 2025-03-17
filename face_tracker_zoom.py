import cv2
import numpy as np
import os
from simple_camera import gstreamer_pipeline
import time

def face_tracker_zoom():
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
    
    window_title = "Face Tracker with Zoom"
    
    # Initialize camera
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=1280,
            capture_height=720,
            display_width=1280,
            display_height=720,
            framerate=30,
            flip_method=2,
            sensor_mode=4
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
            
            # Variables for face tracking and zooming
            process_this_frame = True
            current_zoom = 1.0
            target_zoom = 1.0
            zoom_speed = 0.03  # Even slower transitions
            last_face_time = 0
            face_timeout = 1.0  # Shorter timeout
            min_face_size = 60
            max_zoom = 1.5
            
            # Add smoothing for face position
            smooth_center_x = None
            smooth_center_y = None
            position_smooth = 0.7  # Higher = more position smoothing
            
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break
                
                # Debug information
                if frame_count == 0:  # Print only once
                    print(f"Frame shape: {frame.shape}")
                    print(f"Frame dtype: {frame.dtype}")
                    print(f"Number of channels: {1 if len(frame.shape) == 2 else frame.shape[2]}")
                
                original_frame = frame.copy()
                frame_height, frame_width = frame.shape[:2]
                
                # Only process every other frame for better performance
                if process_this_frame:
                    # Resize frame for faster face detection
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    
                    # Check number of channels and convert to grayscale only if needed
                    if len(small_frame.shape) == 3:  # Color image (3 channels)
                        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    else:  # Already grayscale
                        gray = small_frame
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=4,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Find largest face
                    largest_face = None
                    max_area = 0
                    
                    for (x, y, w, h) in faces:
                        area = w * h
                        if area > max_area:
                            max_area = area
                            largest_face = (int(x * 2), int(y * 2), int(w * 2), int(h * 2))
                    
                    current_time = time.time()
                    
                    # Handle zooming
                    if largest_face and (largest_face[2] > min_face_size):
                        x, y, w, h = largest_face
                        
                        # Calculate face center
                        face_center_x = x + w // 2
                        face_center_y = y + h // 2
                        
                        # Initialize or update smooth center
                        if smooth_center_x is None:
                            smooth_center_x = face_center_x
                            smooth_center_y = face_center_y
                        else:
                            smooth_center_x = int(smooth_center_x * position_smooth + face_center_x * (1 - position_smooth))
                            smooth_center_y = int(smooth_center_y * position_smooth + face_center_y * (1 - position_smooth))
                        
                        # Calculate zoom based on face size
                        face_zoom = min(frame_width / (w * 2), frame_height / (h * 2))
                        target_zoom = min(max(face_zoom, 1.2), max_zoom)  # Minimum zoom of 1.2x
                        last_face_time = current_time
                    elif current_time - last_face_time > face_timeout:
                        target_zoom = 1.0
                        smooth_center_x = None
                        smooth_center_y = None
                    
                    # Smooth zoom transition
                    current_zoom = current_zoom * position_smooth + target_zoom * (1 - position_smooth)
                    
                    # Apply zoom only if we have a valid center and zoom
                    if current_zoom > 1.1 and smooth_center_x is not None:
                        # Calculate zoom region using smoothed center
                        new_w = int(frame_width / current_zoom)
                        new_h = int(frame_height / current_zoom)
                        
                        # Calculate boundaries with smoothed center
                        x1 = max(0, smooth_center_x - new_w // 2)
                        y1 = max(0, smooth_center_y - new_h // 2)
                        
                        # Adjust if too close to edges
                        if x1 + new_w > frame_width:
                            x1 = frame_width - new_w
                        if y1 + new_h > frame_height:
                            y1 = frame_height - new_h
                        
                        x2 = x1 + new_w
                        y2 = y1 + new_h
                        
                        # Crop and resize
                        frame = original_frame[y1:y2, x1:x2]
                        frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Draw rectangles around all detected faces
                if largest_face:
                    x, y, w, h = largest_face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Zoom: {current_zoom:.1f}x", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and display FPS
                frame_count += 1
                if time.time() - last_fps_update >= fps_update_interval:
                    fps = frame_count / (time.time() - last_fps_update)
                    frame_count = 0
                    last_fps_update = time.time()
                
                # Display FPS and zoom level
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
    face_tracker_zoom() 