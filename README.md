
###Face Detection and Recognition System

A real-time face detection and recognition system using OpenCV, designed to recognize and log familiar faces from a single camera feed. This system collects, trains, and recognizes faces, storing each detection with a timestamp for efficient monitoring.

## Features
- **Real-Time Face Detection**: Uses OpenCV's Haar cascades for accurate face detection.
- **Image Collection**: Collects and saves images of faces for training the recognition model.
- **Face Recognition**: Recognizes faces using the Local Binary Patterns Histograms (LBPH) algorithm.
- **Logging**: Records each detection with a timestamp and camera ID, saving to a CSV file for easy analysis.

## Project Structure
- **images/**: Stores collected face images for each person in separate folders.
- **trained_model.yml**: The trained model file used for recognizing faces.
- **detection_log.csv**: Logs each detected face with a timestamp, camera ID, and recognized name.

## Requirements
- Python 3.x
- OpenCV (with `opencv-contrib-python` for LBPH face recognizer)
- Pandas for logging detections to CSV

Install the dependencies:

```bash
pip install opencv-contrib-python pandas
```

## How to Use
1. **Collect Images**:
   - Run the program and enter the name of the person when prompted.
   - Face images will be captured and stored in `images/<person_name>/faces`.

2. **Train the Model**:
   - After collecting images, the system will automatically train a model based on the collected data.
   - The trained model is saved as `trained_model.yml`.

3. **Run Recognition**:
   - Once trained, you can start the recognition phase.
   - The system will identify familiar faces from the camera feed and log each detection.

## Code Overview
- **collect_images()**: Captures 50 face images for each person.
- **train_recognizer()**: Trains an LBPH face recognizer on collected images.
- **recognize_and_log()**: Continuously monitors the camera feed, recognizes known faces, and logs detections.

## Example
To start collecting images and training, run:

```bash
python app.py
```

Follow the prompts to enter names for image collection and start recognition.

