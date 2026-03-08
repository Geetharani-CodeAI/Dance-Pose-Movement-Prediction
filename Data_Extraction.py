# importing libraries
import cv2              
import os               
import numpy as np      
import mediapipe as mp  

# using pretrained model mediapipe pose to identify 33 body keypoints
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Defining input and output paths
DATASET_PATH = "dataset"
OUTPUT_PATH = "pose_data"

# create output path if it does not exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Reading each video from the dataset path
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    output_class_path = os.path.join(OUTPUT_PATH, class_name)

# Creating output class path 
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)

    for video_name in os.listdir(class_path):
        cap = cv2.VideoCapture(os.path.join(class_path, video_name))

        sequence = []

# Reading each video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

# Each frame is converting into BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

# Each body poses to be analysed and appended in sequence
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)

# After reading the video release the memory
        cap.release()

# Converting into numpy array and save the pose data
        sequence = np.array(sequence)
        np.save(os.path.join(output_class_path, video_name.split('.')[0]), sequence)

print("Data extraction completed!")