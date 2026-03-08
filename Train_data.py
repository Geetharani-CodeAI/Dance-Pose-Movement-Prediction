import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# pose data contains extracted pose movements
DATA_PATH = "pose_data"

# Initialize data containers
X = []
y = []
labels = {}
label_count = 0

# Reading each dance classes
for class_name in os.listdir(DATA_PATH):
    labels[class_name] = label_count
    class_path = os.path.join(DATA_PATH, class_name)

# Loading the pose data path 
    for file in os.listdir(class_path):
        data = np.load(os.path.join(class_path, file))

# Fixing the sequence length
        if len(data) > 30:
            data = data[:30]   # Fixed length 30 frames
            X.append(data)
            y.append(label_count)
           
    label_count += 1

# Convert data to numpy arrays and labels to one hot encoding
X = np.array(X)
y = to_categorical(y)
# print(X)

# Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 99)))  # First LSTM Layer
model.add(LSTM(64))                                         # Second LSTM Layer
model.add(Dense(64, activation='relu'))                     # Dense Hidden Layer
model.add(Dense(y.shape[1], activation='softmax'))          # Output Layer

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Creating the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Saving the trained model
model.save("dance_model.h5")

print("Model trained and saved!")