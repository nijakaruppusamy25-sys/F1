"""
run_deepphys.py
===============
Real-Time Heart Rate Estimation using the trained DeepPhys model.

PIPELINE:
    Webcam Frame
        ↓
    MediaPipe Face Mesh
        ↓
    ROI Extraction (Forehead + Cheeks)  —  same as run_final.py / run_chrom.py
        ↓
    DeepPhys preprocessing
        appearance = F_t / 255
        motion     = (F_t - F_{t-1}) / (F_{t-1} + 1)
        ↓
    DeepPhys CNN  →  BVP amplitude per frame
        ↓
    BVP buffer (150 frames)
        ↓
    Bandpass Filter (0.7 – 4.0 Hz)
        ↓
    FFT peak detection
        ↓
    BPM (with smoothing + plausibility gate)
        ↓
    OpenCV display:  BPM text  +  waveform graph  +  ROI rectangle

USAGE:
    1. Train first:   python train_deepphys.py
    2. Run:           python run_deepphys.py

CONTROLS:
    Press  q  →  Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from collections import deque

from deepphys_model import DeepPhys, preprocess_frame_pair


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE     = 36          # must match training
MODEL_PATH   = "models/deepphys_model.pth"

BUFFER_SIZE  = 150         # BVP history frames (~5 s at 30 fps)
TARGET_FPS   = 30

BPM_MIN      = 40
BPM_MAX      = 180
BPM_MAX_JUMP = 20          # reject spikes > 20 BPM from previous reading
SMOOTH_LEN   = 8           # rolling mean window for display BPM

# Forehead + cheeks landmarks  (same as run_final.py / run_chrom.py)
ROI_INDICES = [
    10, 67, 69, 104, 108,   # forehead
    205, 206, 207,           # left cheek
    425, 426, 427,           # right cheek
]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass 0.7–4.0 Hz."""
    nyquist = 0.5 * fs
    low  = 0.7 / nyquist
    high = 4.0 / nyquist
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def estimate_bpm(signal: np.ndarray, fs: float) -> int:
    """FFT peak detection with Hanning window."""
    N = len(signal)
    s = (signal - signal.mean()) * np.hanning(N)   # detrend + window

    yf = fft(s)
    xf = fftfreq(N, 1.0 / fs)

    pos_mask = (xf >= 0.7) & (xf <= 4.0)
    valid_freqs = xf[pos_mask]
    valid_mags  = np.abs(yf[pos_mask])

    if len(valid_freqs) == 0:
        return 0

    return int(valid_freqs[np.argmax(valid_mags)] * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ROI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi(frame: np.ndarray, face_landmarks, draw: bool = True):
    """
    Return the forehead + cheek bounding-box crop.
    Draws green landmark dots and blue ROI rectangle on frame if draw=True.
    Returns None if ROI is empty.
    """
    h, w, _ = frame.shape
    points = []

    for idx in ROI_INDICES:
        lm = face_landmarks.landmark[idx]
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

        if draw:
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0, min(xs))
    x2 = min(w, max(xs))
    y1 = max(0, min(ys))
    y2 = min(h, max(ys))

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    if draw:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return roi, (x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device) -> DeepPhys:
    """Load trained DeepPhys weights.  Falls back to random weights if not found."""
    model = DeepPhys(img_size=IMG_SIZE).to(device)
    model.eval()

    import os
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded model weights from {path}")
    else:
        print(f"[WARNING] Model file not found at '{path}'.")
        print("          Run train_deepphys.py first for meaningful BPM estimates.")
        print("          Continuing with random (untrained) weights for demo.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Device (CPU preferred for macOS stability) ────────────────────────────
    device = torch.device("cpu")
    print(f"[INFO] Running inference on: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH, device)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # ── Buffers ───────────────────────────────────────────────────────────────
    bvp_buffer   = deque(maxlen=BUFFER_SIZE)   # DeepPhys BVP output per frame
    bpm_history  = deque(maxlen=SMOOTH_LEN)    # smoothed BPM history

    prev_roi     = None      # previous ROI crop for motion computation
    filtered_signal = []     # latest filtered BVP signal (for waveform graph)
    bpm          = 0         # displayed BPM
    last_bpm     = 0         # for spike rejection

    print("[INFO] DeepPhys rPPG started. Press 'q' to quit.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:

            success, frame = cap.read()

            if not success:
                print("[WARNING] Frame grab failed.")
                break

            frame = cv2.flip(frame, 1)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # -----------------------------------
            # FRAME SIZE
            # -----------------------------------

            frame_h, frame_w, _ = frame.shape

            # -----------------------------------
            # FACE DETECTION
            # -----------------------------------

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]

                roi, bbox = extract_roi(frame, face_landmarks, draw=True)

                if roi is not None and prev_roi is not None:

                    # -----------------------------------
                    # DEEPPHYS PREPROCESSING
                    # -----------------------------------

                    appearance, motion = preprocess_frame_pair(
                        roi,
                        prev_roi,
                        img_size=IMG_SIZE,
                    )

                    # Add batch dimension → (1, 3, H, W)
                    app_t = torch.from_numpy(appearance).unsqueeze(0).to(device)
                    mot_t = torch.from_numpy(motion).unsqueeze(0).to(device)

                    # -----------------------------------
                    # DEEPPHYS INFERENCE
                    # -----------------------------------

                    with torch.no_grad():
                        bvp_pred = model(app_t, mot_t)   # (1, 1)

                    bvp_val = bvp_pred.item()
                    bvp_buffer.append(bvp_val)

                    # -----------------------------------
                    # BPM CALCULATION
                    # -----------------------------------

                    if len(bvp_buffer) == BUFFER_SIZE:

                        signal = np.array(bvp_buffer, dtype=np.float64)

                        # Bandpass filter
                        filtered_signal = bandpass_filter(signal, TARGET_FPS)

                        # FFT with Hanning window
                        raw_bpm = estimate_bpm(filtered_signal, TARGET_FPS)

                        # Plausibility gate
                        if BPM_MIN <= raw_bpm <= BPM_MAX:

                            # Spike rejection
                            if last_bpm == 0 or abs(raw_bpm - last_bpm) <= BPM_MAX_JUMP:
                                bpm_history.append(raw_bpm)
                                last_bpm = raw_bpm

                        # Rolling mean → smooth display
                        if len(bpm_history) > 0:
                            bpm = int(np.mean(bpm_history))

                # Update previous ROI
                prev_roi = roi.copy() if roi is not None else prev_roi

            else:
                # No face — reset prev_roi so motion is clean on re-detection
                prev_roi = None

            # -----------------------------------
            # GRAPH AREA  (identical to run_final.py)
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

            # Label so user knows which algorithm is running
            cv2.putText(
                frame,
                "DeepPhys",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 0),
                1
            )

            # Buffer fill progress during calibration
            fill_pct = int(len(bvp_buffer) / BUFFER_SIZE * 100)
            if fill_pct < 100:
                cv2.putText(
                    frame,
                    f"Calibrating... {fill_pct}%",
                    (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 0),
                    1
                )

            # -----------------------------------
            # WINDOW
            # -----------------------------------

            cv2.imshow(
                "DeepPhys rPPG Heart Rate Monitor",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit signal received.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
