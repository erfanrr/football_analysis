# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Quick Start

1. **Prerequisites**: Ensure you have the required models and input video:
   - Place your input video in `input_videos/` directory
   - Ensure `models/best.pt` exists (or download from [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing))

2. **Run the pipeline**:
   ```bash
   python main.py
   ```

3. **Output**: Check `output_videos/output_video.avi` for the annotated video

## Documentation

For detailed information about the system:

- **[Execution Pipeline](docs/execution-pipeline.md)**: Complete overview of how the system works, from video input to annotated output
- **[Ball Tracking & Camera Compensation](docs/ball-tracking-and-camera-compensation.md)**: Advanced techniques for improving ball tracking, camera motion compensation, and scaling to full matches

## Architecture Overview

The system processes football video through several stages:
1. **Detection & Tracking**: YOLO + ByteTrack for object detection and tracking
2. **Motion Compensation**: Lucas-Kanade optical flow for camera movement estimation
3. **View Transformation**: Homography-based mapping to field coordinates
4. **Analysis**: Team assignment, ball possession, speed/distance calculation
5. **Visualization**: Annotated video with metrics and overlays