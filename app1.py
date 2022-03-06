from flask import Flask, render_template, request, Response
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import winsound
import time

switch = 1

app = Flask(__name__)
camera = cv2.VideoCapture(0)




def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")


def gen_frames(): 
    flag = 0 
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
          
            frame = imutils.resize(frame, width=400)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                label = "Mask" if mask > withoutMask else "No Mask"
                if label == "Mask":
                    color = (0, 255, 0)
                    winsound.PlaySound(None, winsound.SND_ASYNC)
                    flag = 0
                else:
                    color = (0, 0, 255)
                    if flag==0:
                        print('Wear your Mask!')
                        winsound.PlaySound("alarm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
                        flag=1
                label2 = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label2, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')





@app.route('/')
def index():
    return render_template('index.html')


    
@app.route('/after', methods = ['GET', 'POSt'])
def after():
    return render_template('after.html')

@app.route('/stop', methods = ['GET', 'POST'])
def stop():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('stop') == 'Camera Stop':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
            return redirect({{url_for("index")}})
                
    return render_template('index.html')




@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/feature')
def feature():
    return render_template('about.html')

@app.route('/about')
def about ():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug = True)