import cv2
import os
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime


stop_flag = False
stop_event = threading.Event()

def collect_images(person_name, camera_id=0):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(camera_id)
    face_count = 0

    os.makedirs(f'images/{person_name}/faces', exist_ok=True)

    while face_count < 50:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read from the camera.")
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face_count < 50:
                cv2.imwrite(f'images/{person_name}/faces/face_{face_count}.jpg', face)
                face_count += 1

        time.sleep(0.2)
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    face_ids = []

    person_dirs = os.listdir('images')
    fixed_image_size = (150, 150)
    for idx, person in enumerate(person_dirs):
        face_folder = f'images/{person}/faces'
        for face_image in os.listdir(face_folder):
            img = cv2.imread(f'{face_folder}/{face_image}', cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, fixed_image_size)
            face_samples.append(img)
            face_ids.append(idx)

    recognizer.train(face_samples, np.array(face_ids))
    recognizer.save('trained_model.yml')
    print("Training complete!")

def log_detection(camera_id, person_name):
    df = pd.DataFrame([[datetime.now(), camera_id, person_name]],
                      columns=['Timestamp', 'CameraID', 'PersonName'])
    df.to_csv('detection_log.csv', mode='a', header=False, index=False)

def recognize_and_log(camera_id, recognizer):
    global stop_event

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(camera_id)
    person_dirs = os.listdir('images')

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read from camera {camera_id}")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            if confidence < 50:
                person_name = person_dirs[label]
                print("Person name:", person_name, "Confidence:", confidence)
            else:
                person_name = "unknown"
                print("Unknown person detected")
            log_detection(camera_id, person_name)

        cv2.imshow(f'Working Phase {camera_id}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

    cap.release()
    cv2.destroyAllWindows()


ys = int(input("Press 0 for data collection\nPress 1 for recognizing\n"))
if ys == 0:
    person_name = input("Enter the name of the person to add: ")
    collect_images(person_name)
    print("Images collected, beginning training")
    train_recognizer()
elif ys == 1:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yml')
    recognize_and_log(0, recognizer)
