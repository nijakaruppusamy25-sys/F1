import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Read labels file
labels = pd.read_csv("labels.csv")

# Arrays to store data
X = []
y = []

# -----------------------------------
# Function to extract rPPG signal
# -----------------------------------
def extract_signal(video_path):

    cap = cv2.VideoCapture(video_path)

    signal = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]

                h, w, _ = frame.shape

                # Forehead + cheeks landmarks
                indices = [
                    10, 67, 69, 104, 108,
                    205, 206, 207,
                    425, 426, 427
                ]

                points = []

                for idx in indices:

                    lm = face_landmarks.landmark[idx]

                    x = int(lm.x * w)
                    y1 = int(lm.y * h)

                    points.append((x, y1))

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                # ROI extraction
                roi = frame[y1:y2, x1:x2]

                if roi.size != 0:

                    # Green channel mean
                    green_intensity = np.mean(roi[:, :, 1])

                    signal.append(green_intensity)

    cap.release()

    # Keep fixed length
    signal = signal[:150]

    return signal


# -----------------------------------
# Process all videos
# -----------------------------------
for index, row in labels.iterrows():

    video_name = row["video"]
    bpm = row["bpm"]

    video_path = os.path.join("dataset", video_name)

    print(f"Processing: {video_name}")

    signal = extract_signal(video_path)

    # Only keep videos with enough frames
    if len(signal) == 150:

        X.append(signal)
        y.append(bpm)

    else:
        print(f"Skipped {video_name} (not enough frames)")


# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save processed data
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("\nDataset Prepared Successfully ✅")
print("X shape:", X.shape)
print("y shape:", y.shape)