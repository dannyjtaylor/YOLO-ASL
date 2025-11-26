from ultralytics import YOLO
import cv2
import time

# Initialize FPS tracking
fps_start_time = time.time()
fps_counter = 0
fps = 0

# Initialize YOLO model (nano model - lightweight)
model = YOLO("yolov8n.pt")

# Open camera (0 is default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully")
print("Press 'q' to quit")

# Main loop for camera
while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Run YOLO inference on frame to detect objects
    results = model(frame, conf=0.25)

    # Get detections
    detections = results[0]
    
    # Annotate frame with detection results
    annotated_frame = detections.plot()

    # Calculate FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    # Display FPS
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Break loop when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close window
cap.release()
cv2.destroyAllWindows()
print("Camera closed successfully")