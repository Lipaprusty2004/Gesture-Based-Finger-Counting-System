import streamlit as st
import cv2
import mediapipe as mp

# Streamlit UI
st.title("🖐️ Hand Gesture Recognition (Up to 10 Fingers)")
st.markdown("Detect both hands and count total fingers using OpenCV + MediaPipe")

run = st.checkbox('Start Camera')
frame_window = st.image([])

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
tips_ids = [4, 8, 12, 16, 20]

if run:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not available.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            total_fingers = 0

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((id, cx, cy))

                    fingers = []

                    # Thumb (better logic using distance from wrist)
                    wrist_x = lmList[0][1]
                    thumb_tip_x = lmList[4][1]
                    thumb_ip_x = lmList[3][1]

                    if abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x) + 20:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Fingers (index to pinky)
                    for i in range(1, 5):
                        if lmList[tips_ids[i]][2] < lmList[tips_ids[i] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Add to total
                    total_fingers += fingers.count(1)

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Display total fingers
            cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Click the box to start webcam.")
