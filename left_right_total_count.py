import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np

st.set_page_config(layout="centered")
st.title("🖐️ Streamlit Finger Counter - Left & Right Hands")

# --- State control ---
if "run" not in st.session_state:
    st.session_state.run = False

start_btn, stop_btn = st.columns(2)
with start_btn:
    if st.button("▶️ Start Camera"):
        st.session_state.run = True

with stop_btn:
    if st.button("⛔ Stop Camera"):
        st.session_state.run = False

frame_placeholder = st.empty()

# Setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tips_ids = [4, 8, 12, 16, 20]

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Unable to access webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_count, right_count = 0, 0

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                    lm = hand_landmarks.landmark
                    points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]

                    # -------- Stable Thumb Logic -------- #
                    thumb_tip = points[4]
                    thumb_ip = points[3]

                    fingers = []

                    # Use direction logic
                    if label == 'Right':
                        if thumb_tip[0] > thumb_ip[0]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        if thumb_tip[0] < thumb_ip[0]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Other fingers
                    for i in range(1, 5):
                        if points[tips_ids[i]][1] < points[tips_ids[i] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    count = fingers.count(1)

                    if label == 'Left':
                        left_count = count
                    else:
                        right_count = count

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show counts
            total = left_count + right_count
            cv2.putText(frame, f'Left Hand: {left_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, f'Right Hand: {right_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f'Total Fingers: {total}', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        frame_placeholder.empty()
else:
    st.info("📷 Click 'Start Camera' to begin finger detection.")


