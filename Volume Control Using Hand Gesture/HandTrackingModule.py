import cv2
import mediapipe as mp
import time

#  __init__ method, the class is initialized with parameters such as mode,
#  maxHands, detectionCon, and trackCon.
#  These parameters configure the hand detection and tracking model.
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.maxHands,
                                         min_detection_confidence=self.detectionCon,
                                         min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return frame
    # This method takes a frame as input, converts it to RGB format, and processes it using the MediaPipe hands model to detect hands.
    # If draw is set to True, it draws landmarks and connections on the frame.
    def findPosition(self, frame, handNo=0, draw= True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                lmList.append([id, cx, cy])
                # if id == 4:
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


        return lmList


# It initializes variables for tracking frame rate (pTime, cTime)
# and captures video from the default camera using cv2.VideoCapture(0).


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    # Inside the main loop, it continuously reads frames from the camera, passes each frame to the findHands method of the handDetector class to detect and draw hands,
    # and then calls findPosition to extract landmark positions.
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition( frame )
        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        if not ret:
            print("Error: Could not capture frame.")
            break

        cv2.imshow("Frame", frame)
        #It exits the loop if 'q' is pressed, releases the camera,
        # and closes all OpenCV windows.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
