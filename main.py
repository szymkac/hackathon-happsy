from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def test_image(im):
    target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX


    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)
    x = faces[0, 0]
    y = faces[0, 1]
    w = faces[0, 2]
    h = faces[0, 3]

    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
    face_crop = im[y:y + h, x:x + w]
    face_crop = cv2.resize(face_crop, (48, 48))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop = face_crop.astype('float32') / 255
    face_crop = np.asarray(face_crop)
    face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
    result = target[np.argmax(model.predict(face_crop))]
    #print(result)
    #cv2.putText(im, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

    #cv2.imshow('result', im)
    #cv2.imwrite('result_emotion_detection_app.jpg', im)
    #cv2.waitKey(0)

    return result

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
model = load_model('keras_model/model_5-49-0.62.hdf5')

@app.route('/')
def sessions():
    return render_template('index.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    #result = test_image(cv2.imread('monia.text'))
    #json['emotion'] = result
    print('send response: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)

if __name__ == '__main__':
    socketio.run(app, debug=True)