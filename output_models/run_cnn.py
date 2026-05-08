import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# -----------------------------------
# BANDPASS FILTER
# -----------------------------------

def bandpass_filter(signal, fs):

    low = 0.7
    high = 4.0

    nyquist = 0.5 * fs

    low = low / nyquist
    high = high / nyquist

    b, a = butter(
        3,
        [low, high],
        btype='band'
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# -----------------------------------
# MEDIAPIPE
# -----------------------------------

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

signal_buffer = []

filtered_signal = []

buffer_size = 150

bpm = 0


# -----------------------------------
# FACE MESH LOOP
# -----------------------------------

with mp_face_mesh.FaceMesh(

    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5

) as face_mesh:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        # -----------------------------------
        # DEFAULT FRAME SIZE
        # -----------------------------------

        frame_h, frame_w, _ = frame.shape

        # -----------------------------------
        # FACE DETECTION
        # -----------------------------------

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            h, w, _ = frame.shape

            # -----------------------------------
            # FOREHEAD + CHEEKS LANDMARKS
            # -----------------------------------

            indices = [

                10, 67, 69, 104, 108,

                205, 206, 207,

                425, 426, 427
            ]

            points = []

            for idx in indices:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)

                y = int(lm.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            xs = [p[0] for p in points]

            ys = [p[1] for p in points]

            x1, x2 = min(xs), max(xs)

            y1, y2 = min(ys), max(ys)

            # -----------------------------------
            # ROI
            # -----------------------------------

            roi = frame[
                y1:y2,
                x1:x2
            ]

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2
            )

            if roi.size != 0:

                # -----------------------------------
                # GREEN SIGNAL EXTRACTION
                # -----------------------------------

                green_intensity = np.mean(
                    roi[:, :, 1]
                )

                signal_buffer.append(
                    green_intensity
                )

                if len(signal_buffer) > buffer_size:

                    signal_buffer.pop(0)

                # -----------------------------------
                # BPM CALCULATION
                # -----------------------------------

                if len(signal_buffer) == buffer_size:

                    fs = 30

                    signal = np.array(
                        signal_buffer
                    )

                    # Normalize

                    signal = (
                        signal - np.mean(signal)
                    ) / np.std(signal)

                    # Filter

                    filtered_signal = (
                        bandpass_filter(
                            signal,
                            fs
                        )
                    )

                    # -----------------------------------
                    # FFT
                    # -----------------------------------

                    N = len(filtered_signal)

                    yf = fft(filtered_signal)

                    xf = fftfreq(
                        N,
                        1 / fs
                    )

                    positive_freqs = xf[:N // 2]

                    magnitudes = np.abs(
                        yf[:N // 2]
                    )

                    mask = (

                        (positive_freqs >= 0.7)

                        &

                        (positive_freqs <= 4.0)
                    )

                    valid_freqs = (
                        positive_freqs[mask]
                    )

                    valid_magnitudes = (
                        magnitudes[mask]
                    )

                    peak_freq = valid_freqs[
                        np.argmax(valid_magnitudes)
                    ]

                    bpm = int(
                        peak_freq * 60
                    )

        # -----------------------------------
        # GRAPH AREA
        # -----------------------------------

        graph_height = 150

        graph_width = 400

        graph_x = 50

        graph_y = frame_h - 200

        cv2.rectangle(

            frame,

            (graph_x, graph_y),

            (
                graph_x + graph_width,
                graph_y + graph_height
            ),

            (0, 0, 0),

            -1
        )

        # -----------------------------------
        # DRAW SIGNAL GRAPH
        # -----------------------------------

        if len(filtered_signal) > 0:

            signal_draw = np.array(
                filtered_signal
            )

            signal_draw = (
                signal_draw - np.min(signal_draw)
            )

            signal_draw = signal_draw / (
                np.max(signal_draw) + 1e-6
            )

            signal_draw = (
                signal_draw * graph_height
            )

            for i in range(
                1,
                len(signal_draw)
            ):

                x1_line = graph_x + i - 1

                y1_line = int(
                    graph_y
                    + graph_height
                    - signal_draw[i - 1]
                )

                x2_line = graph_x + i

                y2_line = int(
                    graph_y
                    + graph_height
                    - signal_draw[i]
                )

                cv2.line(

                    frame,

                    (x1_line, y1_line),

                    (x2_line, y2_line),

                    (0, 0, 255),

                    2
                )

        # -----------------------------------
        # DISPLAY BPM
        # -----------------------------------

        cv2.putText(

            frame,

            f"BPM: {bpm}",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            2
        )

        # -----------------------------------
        # WINDOW
        # -----------------------------------

        cv2.imshow(
            "rPPG Heart Rate Monitor",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


cap.release()

cv2.destroyAllWindows()