import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
import matplotlib.pyplot as plt

# ----------------- PATH CONFIGURATION -----------------
dataset_path = r"C:\Users\hadii\PycharmProjects\LKA\TUSimple\train_set"
json_files = [
    os.path.join(dataset_path, "label_data_0313.json"),
    os.path.join(dataset_path, "label_data_0531.json"),
    os.path.join(dataset_path, "label_data_0601.json"),
]  # Include all JSON files

# ----------------- DATA LOADING FUNCTION -----------------
def load_data(json_paths, img_size=(256, 256)):
    """Load images and generate corresponding segmentation masks from all JSON files."""
    images, masks = [], []

    for json_path in json_paths:
        with open(json_path, "r") as file:
            labels = [json.loads(line) for line in file]

        for data in labels:
            raw_file = data["raw_file"]
            lanes = data["lanes"]
            h_samples = data["h_samples"]

            image_path = os.path.join(dataset_path, raw_file)
            if not os.path.exists(image_path):
                continue

            # Load and preprocess image
            image = cv2.imread(image_path)
            image = cv2.resize(image, img_size)
            image = image / 255.0  # Normalize
            images.append(image)

            # Create mask
            mask = np.zeros(img_size, dtype=np.uint8)
            for lane in lanes:
                for x, y in zip(lane, h_samples):
                    if x != -2:
                        x_scaled = int(x * img_size[0] / 1280)  # Rescale to 256x256
                        y_scaled = int(y * img_size[1] / 720)
                        mask[y_scaled, x_scaled] = 1

            masks.append(mask)

    return np.array(images), np.expand_dims(np.array(masks), axis=-1)

# Load dataset
X, Y = load_data(json_files)
print(f"Dataset Loaded: {X.shape}, {Y.shape}")

# ----------------- U-NET MODEL -----------------
def build_unet(input_shape=(256, 256, 3)):
    """Builds a simple U-Net model for lane segmentation."""
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

# Initialize U-Net model
model = build_unet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ----------------- TRAINING -----------------
# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Train the model (Increased to 50 epochs)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=8)

# ----------------- PREDICTION -----------------
def predict_and_display(index=0):
    """Predict lane mask and overlay on original image."""
    image = X_test[index]
    mask_pred = model.predict(np.expand_dims(image, axis=0))[0]
    mask_pred = (mask_pred > 0.5).astype(np.uint8)  # Threshold

    # Overlay mask
    overlay = np.zeros_like(image)
    overlay[mask_pred.squeeze() == 1] = [0, 255, 0]  # Green for lanes

    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_pred.squeeze(), cmap="gray")
    plt.title("Predicted Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Overlay Result")

    plt.show()

# Test the prediction
predict_and_display(index=0)

# Save the model
model.save("lane_detection_unet2.h5")
print("Model saved successfully!")
