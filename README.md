This repository contains a TensorFlow-based implementation of the U-Net architecture for semantic segmentation. U-Net is a powerful convolutional neural network designed primarily for biomedical image segmentation but is also effective in a variety of other pixel-wise prediction tasks.
This repository contains a deep learning-based lane detection system using a UNet architecture.

Files Overview
Training.py
This script is used to train the lane detection model using a dataset from Kaggle.
📦 Kaggle dataset link: [https://www.kaggle.com/datasets/manideep1108/tusimple]

lane_detection_unet2.h5
This is the trained model file generated from Training.py.
➕ If you make any changes to the model architecture or dataset, you must retrain it using Training.py.

Video_check.py
Use this script to test the lane detection model on a pre-recorded video (test_video.mp4).

live check.py
This script runs real-time lane detection using a live camera feed.

test_video.mp4
Sample video file provided for testing the model.

requirements.txt
Contains a list of required Python packages.
🔧 Install them using:
pip install -r requirements.txt
