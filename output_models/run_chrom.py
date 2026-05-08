"""
run_chrom.py  —  CHROM rPPG with stability improvements
=========================================================
Changes over v1:
  1. BPM smoothing      — rolling mean over last 8 readings
  2. Plausibility gate  — reject values outside 40-180 BPM
  3. Spike rejection    — ignore jumps > 20 BPM from previous
  4. Hanning window     — applied before FFT to reduce leakage
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from collections import deque


# -----------------------------------
# CONSTANTS
# -----------------------------------

BUFFER_SIZE  = 150          # ~5 s at 30 fps
TARGET_FPS   = 30
BPM_MIN      = 40
BPM_MAX      = 180
BPM_MAX_JUMP = 20           # max allowed BPM change between readings
SMOOTH_LEN   = 8            # rolling average window for BPM

ROI_INDICES = [
    10, 67, 69, 104, 108,   # forehead
    205, 206, 207,           # left cheek
    425, 426, 427,           # right cheek
]


# -----------------------------------
# BANDPASS FILTER
# -----------------------------------

def bandpass_filter(signal, fs):

    low  = 0.7
    high = 4.0

    nyquist = 0.5 * fs

    low  = low  / nyquist
    high = high / nyquist

    b, a = butter(3, [low, high], btype='band')

    return filtfilt(b, a, signal)


# -----------------------------------
# CHROM ALGORITHM
# -----------------------------------

def chrom_transform(r_buf, g_buf, b_buf):
    """
    CHROM (de Haan & Jeanne, 2013).

    Xs = 3R - 2G
    Ys = 1.5R + G - 1.5B
    alpha = std(Xs) / std(Ys)
    S = Xs - alpha * Ys
    """
    R_n = r_buf / (np.mean(r_buf) + 1e-8)
    G_n = g_buf / (np.mean(g_buf) + 1e-8)
    B_n = b_buf / (np.mean(b_buf) + 1e-8)

    Xs = 3.0 * R_n - 2.0 * G_n
    Ys = 1.5 * R_n + G_n - 1.5 * B_n

    alpha = (np.std(Xs) + 1e-8) / (np.std(Ys) + 1e-8)
    S = Xs - alpha * Ys

    return S


# -----------------------------------
# BPM VIA FFT  (improvement: Hanning window)
# -----------------------------------

def estimate_bpm(signal, fs):

    N = len(signal)
    s = signal - np.mean(signal)

    # FIX 3: Hanning window reduces spectral leakage
    # so the FFT peak is sharper and easier to pick
    window = np.hanning(N)
    s = s * window

    yf = fft(s)
    xf = fftfreq(N, 1 / fs)

    positive_freqs = xf[:N // 2]
    magnitudes     = np.abs(yf[:N // 2])

    mask = (positive_freqs >= 0.7) & (positive_freqs <= 4.0)

    valid_freqs = positive_freqs[mask]
    valid_mags  = magnitudes[mask]

    if len(valid_freqs) == 0:
        return 0

    peak_freq = valid_freqs[np.argmax(valid_mags)]

    return int(peak_freq * 60)


# -----------------------------------
# MEDIAPIPE
# -----------------------------------

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# -----------------------------------
# BUFFERS
# -----------------------------------

r_buffer = []
g_buffer = []
b_buffer = []

filtered_signal = []

buffer_size = BUFFER_SIZE

# FIX 1: BPM history for rolling average
bpm_history = deque(maxlen=SMOOTH_LEN)

bpm         = 0       # smoothed display value
raw_bpm     = 0       # latest raw estimate
last_bpm    = 0       # previous accepted BPM for spike check


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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        # -----------------------------------
        # DEFAULT FRAME SIZE
        # -----------------------------------

        frame_h, frame_w, _ = frame.shape

        # -----------------------------------
        # FACE DETECTION
        # -----------------------------------

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            # -----------------------------------
            # FOREHEAD + CHEEKS LANDMARKS
            # -----------------------------------

            points = []

            for idx in ROI_INDICES:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # -----------------------------------
            # ROI
            # -----------------------------------

            roi = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if roi.size != 0:

                # -----------------------------------
                # RGB SIGNAL EXTRACTION
                # -----------------------------------

                r_mean = np.mean(roi[:, :, 2])   # BGR -> R is index 2
                g_mean = np.mean(roi[:, :, 1])
                b_mean = np.mean(roi[:, :, 0])

                r_buffer.append(r_mean)
                g_buffer.append(g_mean)
                b_buffer.append(b_mean)

                if len(r_buffer) > buffer_size:
                    r_buffer.pop(0)
                    g_buffer.pop(0)
                    b_buffer.pop(0)

                # -----------------------------------
                # BPM CALCULATION
                # -----------------------------------

                if len(r_buffer) == buffer_size:

                    R = np.array(r_buffer, dtype=np.float64)
                    G = np.array(g_buffer, dtype=np.float64)
                    B = np.array(b_buffer, dtype=np.float64)

                    # CHROM pulse signal
                    S = chrom_transform(R, G, B)

                    # Bandpass filter
                    filtered_signal = bandpass_filter(S, TARGET_FPS)

                    # FFT with Hanning window
                    raw_bpm = estimate_bpm(filtered_signal, TARGET_FPS)

                    # -----------------------------------
                    # FIX 2: Plausibility gate
                    # Reject values outside 40-180 BPM
                    # -----------------------------------

                    if BPM_MIN <= raw_bpm <= BPM_MAX:

                        # -----------------------------------
                        # FIX 2b: Spike rejection
                        # Ignore jumps > 20 BPM from last value
                        # -----------------------------------

                        if last_bpm == 0 or abs(raw_bpm - last_bpm) <= BPM_MAX_JUMP:

                            bpm_history.append(raw_bpm)
                            last_bpm = raw_bpm

                    # -----------------------------------
                    # FIX 1: Smoothed BPM (rolling mean)
                    # -----------------------------------

                    if len(bpm_history) > 0:
                        bpm = int(np.mean(bpm_history))

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

            signal_draw = np.array(filtered_signal)

            signal_draw = (
                signal_draw - np.min(signal_draw)
            )

            signal_draw = signal_draw / (
                np.max(signal_draw) + 1e-6
            )

            signal_draw = (
                signal_draw * graph_height
            )

            for i in range(1, len(signal_draw)):

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
            "CHROM rPPG Heart Rate Monitor",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()

cv2.destroyAllWindows()