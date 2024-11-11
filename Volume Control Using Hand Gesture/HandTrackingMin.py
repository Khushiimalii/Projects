import cv2
import mediapipe as mp
import time
# Initialize VideoCapture with camera index
# It will capture video from the default camera (index 0).
cap = cv2.VideoCapture(0)

# Here, we are setting up the mediapipe hands module for hand tracking.
# We create an instance of Hands() class which will be used to detect hands in the frames we capture.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# These variables are used for calculating frames per second (fps) later in the code.
pTime = 0
cTime = 0
# Check if the camera opened successfully
# This checks if the camera was opened successfully. If not, it prints an error message and exits the program.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames from the camera
# This starts an infinite loop where we continuously capture frames from the camera and perform hand tracking
# on each frame.
# The captured frame is in BGR (Blue-Green-Red) color format,
# but mediapipe requires RGB format, so we convert the frame to RGB format.
while True:
    # Capture a frame from the camera
    # ret is a boolean value indicating whether the frame was captured successfully,
    # and frame is the captured frame.
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

# If hand landmarks are detected in the frame, this condition is true.
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # This loop iterates over each landmark point (lm) detected in the hand.
            # id represents the index of the landmark.
            for id, lm in enumerate(handLms.landmark):
               # print(id,lm)
               # This line extracts the height (h), width (w), and number of channels (c) of the frame.
                h, w, c=frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
               # Here, we calculate the pixel coordinates (cx, cy) of each landmark point relative to the frame size.
                if id == 4:
                    cv2.circle(frame, (cx,cy), 15, (255,0,255), cv2.FILLED)
            # If the landmark point is the tip of the thumb (index 4),
            # it draws a filled circle at that point on the frame.

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


# Check if the frame was captured successfully
# This checks if the frame was captured successfully. If not, it prints an error message and breaks the loop.
    if not ret:
        print("Error: Could not capture frame.")
        break

    # These lines calculate the frames per second (fps) based on the time taken to process each frame.
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)),  (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    # This line adds the calculated fps value as text to the frame.
    # Display the captured frame
    cv2.imshow("Frame", frame)
    # This line displays the frame in a window with the title "Frame".
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # This waits for a key press (with a delay of 1 millisecond) and breaks the loop if the pressed key is 'q'.
# Release the VideoCapture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
