import cv2
import os
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime
cameras=[]
stop_flag = False
def collect_images(person_name, camera_id=0):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    cap = cv2.VideoCapture(camera_id)
    face_count, body_count = 0, 0
    os.makedirs(f'images/{person_name}/faces', exist_ok=True)
    #os.makedirs(f'images/{person_name}/bodies', exist_ok=True)

    while face_count < 50 : #or body_count < 50:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read from the camera.")
            break 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face_count < 50:
                cv2.imwrite(f'images/{person_name}/faces/face_{face_count}.jpg', face)
                face_count += 1

        # # Detect bodies
        # bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
        # for (x, y, w, h) in bodies:
        #     body = frame[y:y+h, x:x+w]
        #     if body_count < 50:
        #         cv2.imwrite(f'images/{person_name}/bodies/body_{body_count}.jpg', body)
        #         body_count += 1
        time.sleep(0.2)
        # Display the frame for live feedback
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    face_ids = []
    
    # Loop through collected images
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




def run_multi_camera_system():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Load your recognizer
    recognizer.read('trained_model.yml')  # Load trained model

    cameras=[0,"http://192.168.1.35:4747/video"]  # Assuming 3 cameras; adjust to your system

    threads = []
    
    for camera_id in cameras:
        t = threading.Thread(target=recognize_and_log, args=(camera_id, recognizer))
        threads.append(t)
        t.start()
    while True:
        # Listen for 'q' keypress in the main thread
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit signal detected. Stopping all cameras...")
            stop_flag = True
            break
    # Join all threads
    for t in threads:
        t.join()

def recognize_and_log(camera_id,recognizer):
    global stop_flag
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read('trained_model.yml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(camera_id)
    # cap2=cv2.VideoCapture("http://192.168.78.:4747/video")
    person_dirs = os.listdir('images')
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    while True:
        if stop_flag:
            print(f"Exiting camera {camera_id}...")
            break
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Unable to read from camera {camera_id}")
            break    
        # cam,camera_id=camera_id,cam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            if confidence < 50:  # Confidence threshold
                person_name = person_dirs[label]
                print("peersonname:",person_name,"confidence:",confidence)
                
            else:
                person_name = "unknown"
                print("unknown")
                # time.sleep(2)
            log_detection(camera_id, person_name)

            # Display the frame for live feedback
            cv2.imshow('Working Phase{camera_id}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()



ys=int(input("press 0 for data collection\n press 1 for  recognizing"))
if ys==0:
    person_name = input("Enter the name of the person to add: ")
    collect_images(person_name)
    print("images collected , begining training")
    train_recognizer()
elif ys==1:
    run_multi_camera_system()