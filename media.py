import cv2
import mediapipe as mp
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load audio files
audio_files = ['song1.mp3', 'song2.mp3', 'song3.mp3', 'song4.mp3']  # Replace with your actual audio files
current_audio_index = -1  # To keep track of currently playing audio

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def play_audio(index):
    global current_audio_index
    if index != current_audio_index:
        pygame.mixer.music.load(audio_files[index])
        pygame.mixer.music.play()
        current_audio_index = index
        print(f"Playing audio: {audio_files[index]}")

def stop_audio():
    global current_audio_index
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        current_audio_index = -1
        print("Stopping audio...")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect number of fingers up
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
            finger_status = []

            for tip in finger_tips:
                # Check if the finger is up (y value of tip is less than corresponding base joint)
                finger_status.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

            fingers_up = sum(finger_status)

            # Control audio based on number of fingers up
            if fingers_up == 1:
                play_audio(0)
            elif fingers_up == 2:
                play_audio(1)
            elif fingers_up == 3:
                play_audio(2)
            elif fingers_up == 4:
                play_audio(3)
            elif fingers_up == 5:
                stop_audio()
            else:
                stop_audio()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()