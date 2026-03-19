import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Loading the trained model
model = load_model("dance_model.h5")

# Initializing media pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

labels = ["Bharatanatyam", "HipHop"]

# Load the input 
sequence = []
cap = cv2.VideoCapture("D:\AnacondaPython\Deep_Learning\Dance_pose_movement_Prediction\HB_Dance1.mp4")
#frame = cv2.imread("Hdance.jpeg")
                 
# Read frames per video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
# Converting the frames BGR to RGB  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)     # Detecting the poses

# Check if pose is detected and draw pose skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Extract the landmark coordinates
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

# Frame pose are added and keeping the first 30 frames 
        sequence.append(landmarks)
        sequence = sequence[-30:]

# Collecting the frames and preparing the input model
        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data)[0]  # Predicting the model
            label = labels[np.argmax(prediction)]      # Predicting the max

# Displaying prediction on video
            cv2.putText(frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Dance Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()