import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained age detection model
age_net = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt',  # Path to the prototxt file
    'age_net.caffemodel'    # Path to the caffemodel file
)

# Define age ranges (these ranges are based on the training data)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def detect_age(frame):
    """Detect age in the provided frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess the face region for the age detection model
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Perform age prediction
        age_net.setInput(blob)
        age_predictions = age_net.forward()
        age = AGE_LIST[age_predictions[0].argmax()]

        # Display the detected age range
        cv2.putText(frame, age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

def main():
    """Main function to capture video and perform age detection."""
    cap = cv2.VideoCapture(0)  # Use webcam (camera index 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Detect age in the current frame
        processed_frame = detect_age(frame)

        # Display the frame with detected age
        cv2.imshow("Age Detection", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main();