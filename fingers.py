import cv2
import mediapipe as mp

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_fingers(hand_landmarks):
    # Landmark indices for tips of the fingers (thumb, index, middle, ring, little fingers)
    finger_tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: Compare tip with IP joint
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # right hand
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers: Compare tip with PIP joint
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

# Access webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR frame to RGB for processing with Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of raised fingers
            fingers_count = count_fingers(hand_landmarks)

            # Display the count on the frame
            cv2.putText(frame, f'Fingers: {fingers_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Finger Counter', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
