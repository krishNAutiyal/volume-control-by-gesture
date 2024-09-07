import mediapipe as mp
import cv2
import time
import os
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize mediapipe hands solution
mpHands = mp.solutions.hands
hands = mpHands.Hands()     #used to keep the actual tracking of hands
mpDraw = mp.solutions.drawing_utils      #used to indicate landmarks on screen

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert the image to RGB  ...opencv captures image in BGR format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    results = hands.process(imgRGB)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # Debugging statement to print coordinates
                print(f"Landmark ID: {id}, X: {cx}, Y: {cy}")
                cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            if lmList:
                # Get the tip of the thumb and index finger
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                # Calculate the distance between the thumb and index finger
                length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

                # Map the hand gesture to volume range directly
                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                # Draw volume bar
                volBar = np.interp(vol, [minVol, maxVol], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                volPer = np.interp(vol, [minVol, maxVol], [0, 100])
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image
    cv2.imshow("Tracking Window", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
