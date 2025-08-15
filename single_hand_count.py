import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np

# Set page title
st.title("✋ Hand Gesture Recognition with Finger Counting")
st.markdown("Detect and count fingers using OpenCV + MediaPipe + Streamlit")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

run = st.checkbox('Start Camera')

# Streamlit video placeholder
frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not working.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            total = 0

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((id, cx, cy))

                    fingers = []

                    # Thumb using wrist distance logic
                    wrist_x = lmList[0][1]
                    thumb_tip_x = lmList[4][1]
                    thumb_ip_x = lmList[3][1]

                    if abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x) + 20:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other 4 fingers
                    for i in range(1, 5):
                        if lmList[tips_ids[i]][2] < lmList[tips_ids[i] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total = fingers.count(1)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Show result on frame
            cv2.putText(frame, f'Fingers: {total}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            # Show frame in Streamlit
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.info("Check the box to start camera.")
