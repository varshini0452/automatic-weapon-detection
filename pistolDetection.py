import cv2
import numpy as np
import winsound  

# Load video file
cap = cv2.VideoCapture("gun_clip.mp4")

# Width/Height for blob input
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Class names (only detecting Pistol)
classNames = ["Pistol"]

# Paths to model config and weights
modelConfiguration = r"C:\yolo\yolov3-custom.cfg"
modelWeights = r"C:\yolo\yolov3-custom_2000.weights"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    boundingBoxes = []
    classIndexes = []
    confidenceValues = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classIndex = np.argmax(scores)
            confidence = scores[classIndex]

            # Only detect Pistol (index 0)
            if confidence > confThreshold and classIndex == 0:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                x, y = int((detection[0]*wT) - w/2), int((detection[1]*hT) - h/2)
                boundingBoxes.append([x, y, w, h])
                classIndexes.append(classIndex)
                confidenceValues.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boundingBoxes[i]
        label = f'{classNames[classIndexes[i]].upper()} {int(confidenceValues[i]*100)}%'

        # 🚨 ALERT: Pistol detected
        print("🚨 ALERT! Pistol Detected!")
        winsound.Beep(1000, 200)  # Frequency 1000Hz, Duration 200ms

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - 1, y - 25), (x + w + 1, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠ Could not read frame or end of video.")
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)

    outputLayerNames = net.getUnconnectedOutLayersNames()
    outputs = net.forward(outputLayerNames)

    findObjects(outputs, frame)

    # Add instruction at bottom-left
    cv2.putText(frame, "Press 'q' or 'Esc' to exit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    # Resize for presentation
    frame_resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Pistol Detection', frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("✅ Video closed by user.")
        break

cap.release()
cv2.destroyAllWindows()