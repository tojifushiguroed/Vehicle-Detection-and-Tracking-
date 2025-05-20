# Vehicle-Detection-and-Tracking-
Vehicle Detection and Counting in Urban Traffic Scenes

https://drive.google.com/file/d/16A5rJtqlK2XrMOv2z0JrsmP9nbX3pXux/view faster_rcnn_module
https://universe.roboflow.com/rjacaac1/ua-detrac-dataset-10k dataset link

#### Owners:
Mert Taşlıyurt
Eren Efe Taşlıyurt
Elif Deniz Gölboyu 


### 1. Introduction
This project addresses the problem of real-time vehicle detection and counting in urban traffic scenes using deep learning techniques. The system utilizes a YOLOv8 model for object detection and processes video inputs to count vehicles per class (car, bus, truck, etc.). The system analyzes pre-recorded video footage and is adaptable to real-time surveillance or traffic management scenarios.

### 2. System Architecture
The vehicle detection and counting system comprises the following components:
Video Source: Urban traffic video, sourced from a dataset or pre-recorded footage.
YOLOv8 Model: Performs object detection and classification.
Tracking Module: Maintains object identities across video frames (ByteTrack).
Counting Mechanism: Detects objects crossing a defined virtual line and updates class-wise counts.
Visualization Output: Annotates video frames with bounding boxes, class names, object IDs, and overlays class-wise counts.

#### 2.1 Architecture Diagram

<img width="625" alt="image" src="https://github.com/user-attachments/assets/4c536523-fa67-4d73-857f-8f973bb37961" />

### 3. Data Flow
Video Loading: A traffic video is loaded using OpenCV.
Frame-by-Frame Detection: Each frame is processed by the YOLOv8 model to detect vehicles.
Object Tracking: Detected vehicles are tracked across frames using the ByteTrack algorithm, providing temporal consistency.
Line Crossing Check: A virtual line is defined within the video frame. The system determines when tracked objects cross this line.
Frame Annotation: The original video frames are annotated with bounding boxes around detected vehicles, class names, unique object IDs assigned by the tracker, and a display of the accumulated vehicle count for each class.
Output: The annotated video is saved to a file (output.avi).

### 4. Setup and Installation
The system requires Python 3.8 or later. The following steps detail the installation procedure:
Install Dependencies:
pip install -r requirements.txt
Core dependencies include: ultralytics, opencv-python, matplotlib, supervision, and byte_track.
Dataset Preparation:
The system is designed to work with the UA-DETRAC dataset. Video files from the dataset should be placed in the directory referenced in main.py. Ensure the correct path to the dataset is configured.

### 5. Code Explanation
The core logic of the vehicle detection and counting system is implemented in main.py. Key functionalities include:
YOLOv8 Model Initialization: Loads a pre-trained YOLOv8 model for vehicle detection.
Counting Line Definition: Defines the coordinates of a virtual line within the video frame.
Detection and Tracking: Performs vehicle detection using YOLOv8 and tracks detected vehicles using the ByteTrack algorithm.
Line Crossing Logic: Determines when a tracked vehicle crosses the defined virtual line and updates the corresponding class counter.
Visualization and Output: Annotates video frames with detection and tracking information and saves the processed video to output.avi.

### 6. Key Features
Real-time vehicle detection using YOLOv8.
Multi-object tracking with ByteTrack.
Vehicle counting based on line crossing.
Class-wise vehicle count maintenance (e.g., cars, buses, trucks).
Annotated output video generation.

### 7. Operating Instructions
To run the vehicle detection and counting system:
Clone the Repository
Install Dependencies: pip install -r requirements.txt
Prepare Dataset: Download the UA-DETRAC dataset and organize the video files as required by the main.py script.
Run the System: python main.py
View Output: The processed video, with vehicle detections, tracks, and counts, is saved as output.avi. Class-wise vehicle counts are also displayed in the terminal.

### 8. Training and Evaluation Results
The YOLOv8 model was trained for 80 epochs. The table below shows a sample of the training and validation losses for the first 10 epochs.
<img width="622" alt="image" src="https://github.com/user-attachments/assets/69d85bf8-df15-4516-b284-f04b668b47f8" />

The YOLOv8 model achieved the following performance metrics:
Precision: Up to ~0.91
Recall: Up to ~0.89
mAP@0.5: ~0.94
mAP@0.5:0.95: ~0.50

The model was trained for a total of 80 epochs. We experimented with different numbers of epochs (starting with 50) to optimize the balance between accuracy and preventing overfitting.

### 9. Future Improvements
Real-time Processing: Implement the system to process video streams in real-time instead of only pre-recorded footage. This would involve optimizing the code for speed and deploying it on a system with a capable GPU.
Camera Calibration: Calibrate the camera to obtain accurate measurements of vehicle speeds and distances. This would allow for more advanced traffic analysis, such as speed enforcement and congestion monitoring.
Handling Occlusion: Improve the tracking algorithm to handle cases where vehicles are partially or fully occluded by other objects. This could involve using more sophisticated tracking algorithms or incorporating information from multiple cameras.
Integration with Traffic Management Systems: Integrate the system with existing traffic management systems to provide real-time data on traffic flow and allow for dynamic adjustments to traffic signals.
Expanded Vehicle Classification: Expand the system to classify a wider range of vehicle types, such as motorcycles, bicycles, and emergency vehicles. This would provide a more comprehensive understanding of traffic patterns.
Adverse Weather Conditions: Evaluate the system's performance under various weather conditions, such as rain, snow, and fog, and implement techniques to improve its robustness to these conditions.

### 10. References
Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
Zhang et al. (2021), ByteTrack
OpenCV: https://docs.opencv.org/
UA-DETRAC: https://detrac-db.rit.albany.edu/
Roboflow: https://roboflow.com
