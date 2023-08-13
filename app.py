from flask import Flask, render_template, Response
import cv2
from torchvision import models, transforms
from ultralytics import YOLO

app = Flask(__name__)

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
print('Model loaded....')
###

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture('static/video.mp4')  # Change 'static/video.mp4' to your video path
        #self.model = cv2.dnn.readNet('path/to/your/model/config', 'path/to/your/model/weights')  # Load your object detection model

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(640, 640), mean=(104, 177, 123))
            self.model.setInput(blob)
            detections = self.model.forward()

            for detection in detections[0, 0]:
                confidence = detection[2]
                if confidence > 0.5:  # Adjust the confidence threshold as needed
                    x = int(detection[3] * frame.shape[1])
                    y = int(detection[4] * frame.shape[0])
                    x_end = int(detection[5] * frame.shape[1])
                    y_end = int(detection[6] * frame.shape[0])
                    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        else:
            return b''

camera = VideoCamera()
####
@app.route('/')
def index():
    return render_template('home.html')

def generate():
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000",debug=True)
