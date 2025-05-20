from types import SimpleNamespace
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from yolox.tracker.byte_tracker import BYTETracker

num_classes_original = 91
model = fasterrcnn_resnet50_fpn(pretrained=False)  
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_original)
checkpoint = torch.load(r"C:\Users\DELL\source\repos\VisionProject\VisionProject\fasterrcnn_vehicle_detector.pth")
model.load_state_dict(checkpoint)
num_classes_custom=5
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_custom)
model.to('cuda')
model.eval()

# Your class names
MY_CLASSES = ['_background_', 'car', 'bus', 'truck','van']  # update accordingly
args = SimpleNamespace()
args.track_thresh = 0.5
args.track_buffer = 30
args.match_thresh = 0.8
args.aspect_ratio_thresh = 1.6
args.mot20 = False
tracker = BYTETracker(args)

cap = cv2.VideoCapture(r"C:\Users\DELL\Downloads\example_video_n2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(640,640))
    img_tensor = torch.from_numpy(frame / 255.).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    with torch.no_grad():
        detections = model(img_tensor)[0]

    scores = detections['scores'].cpu().numpy()
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()

    conf_thresh = 0.5
    mask = (scores >= conf_thresh) & (labels > 0) & (labels < len(MY_CLASSES))
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    dets = np.array([*zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores, labels)])
    dets_tensor = torch.tensor(dets, dtype=torch.float32)
    img_size = frame.shape[:2]  
    img_info = (frame.shape[1], frame.shape[0], 1.0)
    tracks = tracker.update(dets_tensor,img_info,img_size)

    for track in tracks:
        tlwh = track.tlwh  
        track_id = track.track_id
        cls_id=labels[0]
        x1 = int(tlwh[0])
        y1 = int(tlwh[1])
        x2 = int(tlwh[0] + tlwh[2])
        y2 = int(tlwh[1] + tlwh[3])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'{MY_CLASSES[cls_id]} ID:{track_id}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
