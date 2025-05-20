import cv2
from ultralytics import YOLO
model = YOLO(r"C:\Users\DELL\source\repos\VisionProject\VisionProject\runs\detect\train5\weights\best.pt")
source=input('Location of the video or index of the camera to use:')
cap=None
if source.isnumeric():
    cap = cv2.VideoCapture(int(source))
else:
    cap =cv2.VideoCapture(source)

display_width = 800

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, tracker="bytetrack.yaml")
    result = results[0]

    img = result.orig_img.copy()

    height, width = img.shape[:2]
    scale_ratio = display_width / width
    resized_img = cv2.resize(img, (display_width, int(height * scale_ratio)))

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (box * scale_ratio).astype(int)
            class_id = classes[i]
            class_name = model.names[class_id]  
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_img, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('BYTETrack', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
