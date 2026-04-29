# Automatic Weapon Detection using YOLOv3

## Overview
This project detects pistols in real-time using the YOLOv3. It processes video frames using OpenCV and highlights detected weapons with bounding boxes.

## Features
- Real-time pistol detection  
- YOLOv3-based model  
- Video frame processing  
- Bounding box with confidence score  

## Tech Stack
- Python  
- OpenCV  
- YOLOv3  
- NumPy  

## How to Run
1. Install dependencies:
   pip install opencv-python numpy  

2. Run the code:
   python src/detect.py  

## Output
The system detects pistols in video and displays results with bounding boxes in real-time and also gives a sound alert.

## Limitations
- Detects only pistols  
- Accuracy depends on video quality  

## Future Scope
- Detect multiple weapons  
- Integrate with CCTV systems

