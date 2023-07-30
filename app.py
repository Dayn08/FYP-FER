from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file
import cv2
import numpy as np
import datetime
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os, time
import threading
from queue import Queue

app = Flask(__name__)

model = tf.keras.models.load_model('Trained_model_FER2013.h5')

path = "haarcascade_frontalface_default.xml"

cap = None  # Variable to hold the VideoCapture object
emotion_recognition_active = False  # Flag to indicate if facial emotion recognition is active

# Create a Queue to store emotion data
emotion_queue = Queue()

def start_emotion_recognition():
    global cap, emotion_recognition_active

    # Open the video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set the flag to indicate emotion recognition is active
    emotion_recognition_active = True


def stop_emotion_recognition():
    global cap, emotion_recognition_active

    # Release the video capture and reset the flag
    cap.release()
    emotion_recognition_active = False


def gen_frames():
    global cap

    last_update_time = datetime.datetime.now().strftime('%H:%M:%S')

    while True:
        if emotion_recognition_active:
            ret, frame = cap.read()
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                final_image = cv2.resize(roi_color, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image/255.0
                Predictions = model.predict(final_image)
                if (np.argmax(Predictions) == 0):
                    status = "Angry"
                elif (np.argmax(Predictions) == 1):
                    status = "Disgust"
                elif (np.argmax(Predictions) == 2):
                    status = "Fear"
                elif (np.argmax(Predictions)==3):
                    status = "Happy"
                elif (np.argmax(Predictions)==4):
                    status = "Neutral"
                elif (np.argmax(Predictions)==5):
                    status = "Sad"
                elif (np.argmax(Predictions)==6):
                    status = "Surprise"
                cv2.putText(frame, status, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                date =  datetime.datetime.now().strftime("%Y-%m-%d")
                if timestamp != last_update_time:
                    last_update_time = timestamp
                    data = f"data: {date} {timestamp} {status}\n\n"
                    emotion_queue.put(data)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_emotion_data():
    while True:
        if emotion_queue.empty():
            continue

        # Get emotion data from the queue
        emotion_data = emotion_queue.get()
        yield emotion_data

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/db', methods=['GET', 'POST'])
def db():
    return render_template('db.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_emotion_recognition', methods=['POST'])
def start_emotion_recognition_route():
    start_emotion_recognition()
    return jsonify({'message': 'Emotion recognition started.'})


@app.route('/stop_emotion_recognition', methods=['POST'])
def stop_emotion_recognition_route():
    stop_emotion_recognition()
    return jsonify({'message': 'Emotion recognition stopped.'})

@app.route('/emotion_data')
def emotion_data():
    def generate():
        while True:
            # Read data from the queue (this will block if the queue is empty)
            data = emotion_queue.get()

            # Yield the data to the client
            yield data

            # Signal that the data has been consumed from the queue
            emotion_queue.task_done()

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
