import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

cap = cv2.VideoCapture(0)
current_time = 0
past_time = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)

    # Convert the image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box and confidence
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    current_time = time.time()
    frames_per_second = 1 / (current_time - past_time)
    past_time = current_time

    cv2.putText(img, f'FPS: {str(int(frames_per_second))}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    # Display the result
    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
