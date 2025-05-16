import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("lane_detection_unet2.h5")

# Set input size for the model
input_size = (256, 256)  # Reduced size for faster processing

# Open the live camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Force camera to 30 FPS

# Global variables
latest_frame = None
processed_frame = None
lock = threading.Lock()
frame_skip = 2  # Process every 2nd frame for faster FPS

def process_frame():
    global latest_frame, processed_frame
    while True:
        if latest_frame is None:
            continue  # Skip processing if no frame is available

        with lock:
            frame = latest_frame.copy()

        # Resize only for model input
        frame_resized = cv2.resize(frame, input_size)

        # Normalize and reshape for model input
        frame_input = frame_resized / 255.0
        frame_input = np.expand_dims(frame_input, axis=0).astype(np.float16)  # Use float16 for speed

        # Predict lane mask (Only process every 2nd frame)
        pred_mask = model.predict(frame_input, verbose=0)[0]

        # Convert prediction to binary lane mask
        pred_mask = (pred_mask * 255).astype(np.uint8)
        _, lane_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)

        # Resize mask back to original frame size
        lane_mask = cv2.resize(lane_mask, (640, 480))

        # Overlay mask onto the original frame
        overlay = frame.copy()
        overlay[lane_mask > 0] = [0, 255, 0]  # Green color for lane

        with lock:
            processed_frame = overlay  # Update processed frame

# Start processing in a separate thread
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the camera feed is unavailable

    frame_count += 1
    if frame_count % frame_skip == 0:  # Skip every 2nd frame for speed
        with lock:
            latest_frame = frame.copy()  # Update the latest frame

    with lock:
        if processed_frame is not None:
            cv2.imshow("Real-Time Lane Detection", processed_frame)
        else:
            cv2.imshow("Real-Time Lane Detection", frame)  # Show raw video initially

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
