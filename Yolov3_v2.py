import cv2
import numpy as np
import requests
from io import BytesIO

# URL kamera web server pada ESP32-CAM
url = "http://192.168.242.27/capture"

# Load model YOLOv3
net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')
classes = open('coco.names').read().strip().split('\n')

while True:
    # Ambil gambar dari kamera web server
    response = requests.get(url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(img_array, -1)

    # Preprocess image
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass and get predictions
    detections = net.forward(output_layer_names)

    # Loop over detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                # Draw bounding box
                box = obj[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype(int)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cv2.destroyAllWindows()