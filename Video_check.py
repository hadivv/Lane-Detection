import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("lane_detection_unet2.h5")

# Set input size (update if different)
input_size = (256, 256)  # Adjust based on your model

# Open the video file
video_path = "Road video.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))  # Width of the frame
frame_height = int(cap.get(4))  # Height of the frame
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define codec and create a VideoWriter object
output_path = "lane_detection_output2.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    # Resize frame to match model input size
    frame_resized = cv2.resize(frame, input_size)

    # Normalize and reshape for model input
    frame_input = frame_resized / 255.0  # Normalize
    frame_input = np.expand_dims(frame_input, axis=0)  # Add batch dimension

    # Predict lane mask
    pred_mask = model.predict(frame_input)[0]

    # Normalize prediction to [0, 255]
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # Apply threshold to highlight lanes
    _, lane_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)

    # Resize mask back to original frame size
    lane_mask = cv2.resize(lane_mask, (frame_width, frame_height))

    # Overlay lane markings on the original frame
    overlay = frame.copy()
    overlay[lane_mask > 0] = [0, 255, 0]  # Green for detected lanes

    # Write the processed frame to output video
    out.write(overlay)

    # Show the frame (optional, remove if not needed)
    cv2.imshow("Lane Detection", overlay)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit early
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_path}")
