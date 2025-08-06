import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
from filterpy.kalman import KalmanFilter
from datetime import datetime

# --- Config ---
video_path = "videos/juggling_clip.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"output/juggling_output_{timestamp}.mp4"
hit_threshold = 5
max_missing = 10

# --- Device ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --- Load YOLO model ---
model = YOLO("yolov8x.pt")
model.to(device)

# --- Video setup ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

slow_factor = 0.5  # Adjust to 0.25, 0.5, 1.0, etc.
slow_fps = max(1, int(fps * slow_factor))

out = cv2.VideoWriter(output_path, fourcc, slow_fps, (width, height))

# --- Kalman filter setup ---
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
kf.P *= 1000.  # Initial uncertainty
kf.R *= 10     # Measurement noise
kf.Q *= 0.01   # Process noise

# --- Ball detection utils ---
def is_similar_size(prev_box, new_box, tolerance=0.25):
    prev_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
    new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
    return abs(prev_area - new_area) / prev_area < tolerance

def make_square_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1)
    half = size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, frame_width - 1)
    y2 = min(cy + half, frame_height - 1)
    return [x1, y1, x2, y2]

def deblur_frame(frame):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def detect_ball(frame, prev_box, missing_count, frame_idx):
    frame = deblur_frame(frame)
    results = model.predict(frame, verbose=False, device=device, conf=0.2)[0]
    candidates = [box for box in results.boxes if int(box.cls[0]) == 32]

    if not candidates:
        print(f"[BALL] not found at frame {frame_idx}")
        if missing_count < max_missing:
            kf.predict()
            pred_x, pred_y = kf.x[0], kf.x[1]
            box_size = (prev_box[2] - prev_box[0]) if prev_box else 30
            x1 = int(pred_x - box_size / 2)
            y1 = int(pred_y - box_size / 2)
            x2 = int(pred_x + box_size / 2)
            y2 = int(pred_y + box_size / 2)
            predicted_box = make_square_box([x1, y1, x2, y2], width, height)
            return predicted_box, missing_count + 1, True
        return None, missing_count + 1, False

    best = None
    best_area = 0
    for box in candidates:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        candidate_box = [x1, y1, x2, y2]
        candidate_box = make_square_box(candidate_box, width, height)
        if prev_box and not is_similar_size(prev_box, candidate_box):
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = candidate_box

    if best:
        cx = (best[0] + best[2]) // 2
        cy = (best[1] + best[3]) // 2
        kf.predict()
        kf.update(np.array([cx, cy]))
        return best, 0, False
    elif prev_box and missing_count < max_missing:
        kf.predict()
        pred_x, pred_y = kf.x[0].item(), kf.x[1].item()
        box_size = (prev_box[2] - prev_box[0])
        x1 = int(pred_x - box_size / 2)
        y1 = int(pred_y - box_size / 2)
        x2 = int(pred_x + box_size / 2)
        y2 = int(pred_y + box_size / 2)
        predicted_box = make_square_box([x1, y1, x2, y2], width, height)
        return predicted_box, missing_count + 1, True
    else:
        return None, missing_count + 1, False

# --- Hit detection ---
def detect_bounce(y_positions, max_track_len, frame_idx, last_hit_frame, hit_count):
    if len(y_positions) == max_track_len:
        y_prev = y_positions
        if y_prev[2] < y_prev[1] and y_prev[2] < y_prev[3]:
            if frame_idx - last_hit_frame > 10:
                hit_count += 1
                last_hit_frame = frame_idx
                print(f"[BOUNCE] #{hit_count} at frame {frame_idx}")
    return hit_count, last_hit_frame

# --- Tracking ---
y_positions = []
max_track_len = 10
hit_count = 0
last_hit_frame = -30
frame_idx = 0
missing_count = 0
prev_box = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    box, missing_count, is_predicted = detect_ball(frame, prev_box, missing_count, frame_idx)
    if box:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        prev_box = box

        # Draw ball box and center
        color = (0, 165, 255) if is_predicted else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Track Y for hit detection
        y_positions.append(cy)
        if len(y_positions) > max_track_len:
            y_positions.pop(0)

        # Detect hit
        hit_count, last_hit_frame = detect_bounce(y_positions, max_track_len, frame_idx, last_hit_frame, hit_count)

    # Display debug info
    cv2.putText(frame, f"Bounces: {hit_count}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 0, 0), 3)

    out.write(frame)

cap.release()
out.release()
print(f"[✅] Saved video with hit counter to: {output_path}")
