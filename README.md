📌 Real-Time Finger Counting using OpenCV

This project demonstrates real-time finger counting using a webcam feed and OpenCV for image processing. It utilizes hand landmark detection to accurately identify and count raised fingers.

🔹 Features

Single-Hand Finger Counting – Detects and counts fingers from one hand.

Dual-Hand Finger Counting – Detects and counts fingers from both hands simultaneously.

Sequential Left & Right Hand Counting – Counts fingers on the left hand first, then the right hand, and finally displays the total finger count.

🛠 Tech Stack

Python

OpenCV

MediaPipe (for hand landmark detection)

NumPy

🚀 How It Works

Captures live video from the webcam.

Detects hand landmarks using MediaPipe.

Uses finger tip positions to determine which fingers are raised.

Displays the count in real-time on the video feed.

📂 Files in the Repository

single_hand_count.py → Counts fingers of one hand.

dual_hand_count.py → Counts fingers of both hands.

left_right_total_count.py → Counts left hand first, then right hand, and shows total count.
