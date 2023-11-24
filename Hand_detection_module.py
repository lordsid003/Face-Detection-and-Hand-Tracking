import cv2
import mediapipe as mp
import time

class Detector():
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1,
               min_detection_confidence=0.5,min_tracking_confidence=0.5):
        
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.model_complexity = model_complexity
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                            self.detection_confidence, self.tracking_confidence)
        
        self.mp_draw = mp.solutions.drawing_utils
        self.tips_id = [4, 8, 12, 16, 20]

    def detect_func(self, img, draw=True):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        self.results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def position_detection(self, img, hand_num=0, draw=True):   

        self.landmarks_container = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id, land_mark in enumerate(hand.landmark):
                    # print(id, land_mark)
                    height, width, channels = img.shape
                    x_coordinate, y_coordinate = int(land_mark.x * width), int(land_mark.y * height)
                    # print(id, x_coordinate, y_coordinate)
                    self.landmarks_container.append([id, x_coordinate, y_coordinate])
                    if draw:
                        cv2.circle(img, (x_coordinate, y_coordinate), 10, (255, 0, 255), cv2.FILLED)

        return self.landmarks_container
    
    def track_finger(self):
        self.fingers = []

        # Tracking thumb
        if self.landmarks_container[self.tips_id[0]][1] < self.landmarks_container[self.tips_id[0] - 1][1]:
            self.fingers.append(1)
        else:
            self.fingers.append(0)

        # Tracking fingers
        for id in range(1, 5):
            if self.landmarks_container[self.tips_id[id]][2] < self.landmarks_container[self.tips_id[id] - 2][2]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

        return self.fingers

def main():
    current_time = 0
    past_time = 0
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while cap.isOpened():
        _, img = cap.read()
        img = detector.detect_func(img)
        landmarks_list = detector.position_detection(img, draw=False)
        if len(landmarks_list) != 0:
            print(landmarks_list)
            cv2.putText(img, "Hand detected!", (40, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        current_time = time.time()
        frames_per_second = 1 / (current_time - past_time)
        past_time = current_time

        cv2.putText(img, f'FPS: {str(int(frames_per_second))}', (10, 30), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.imshow("Virtual Signature Portal", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
