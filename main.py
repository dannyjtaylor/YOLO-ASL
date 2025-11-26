from ultralytics import YOLO
import cv2
import time
import os
from pathlib import Path


#ASL dataset
import kagglehub

#download latest version
path = kagglehub.dataset_download("ayuraj/asl-dataset")

print("Path to dataset files:", path)

# # explore structure of ASL dataset
# asl_path = Path(path)
# print("ASL dataset:")
# for item in asl_path.iterdir():
#     print(item.name)
#     if item.is_dir():
#         for subitem in item.iterdir():
#             print(f"  {subitem.name}")





fps_start_time = time.time()
fps_counter = 0
fps = 0

#initalize YOLO model
# model = YOLO("runs/detect/asl_yolo3/weights/best.pt")
model = YOLO("yolo8n.pt")
#open camera (0 is default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully")
print("Press 'q' to quit")

# main loop for camera

while True:
    #read frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    #run yolo inference on frame to detect objects
    results = model(frame, conf=0.1)  # Lower confidence threshold to see more detections

    # Get detections
    detections = results[0]
    
    # Debug: Print detection info every 30 frames
    if fps_counter % 30 == 0:
        print(f"Detections: {len(detections.boxes)}")
        for box in detections.boxes:
            class_id = int(box.cls[0])
            class_name = detections.names[class_id]
            confidence = float(box.conf[0])
            print(f"  {class_name}: {confidence:.3f}")
    
    # Annotate frame with detection results
    annotated_frame = detections.plot()
    
    # Draw all detections with lower confidence threshold
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        confidence = float(box.conf[0])

        # Show detections with lower confidence threshold
        if confidence > 0.15:  # Lowered from 0.5 to 0.15
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw box with letter name
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #calculate FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    #display FPS
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #display frame
    cv2.imshow('ASL Letter Detection', annotated_frame)

    #break loop when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resoures to close window
cap.release()
cv2.destroyAllWindows()
print("Camera closed successfully")