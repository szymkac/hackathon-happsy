from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import base64
<<<<<<< HEAD
=======
import glob, random
>>>>>>> bbff5f004a4f8ffc16a6925515317a5fe3918ef4

def test_image(bytes):
    inp = np.asarray(bytearray(bytes), dtype=np.uint8)
    im = cv2.imdecode(np.fromstring(inp, dtype=np.uint8), cv2.IMREAD_COLOR)
    target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
<<<<<<< HEAD
    font = cv2.FONT_HERSHEY_SIMPLEX
=======
>>>>>>> bbff5f004a4f8ffc16a6925515317a5fe3918ef4
    faces = faceCascade.detectMultiScale(im, scaleFactor=1.1)
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
    print('Face {:}'.format(face_crop))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))
    model = load_model('keras_model/model_5-49-0.62.hdf5')
    result = target[np.argmax(model.predict(face_crop))]
<<<<<<< HEAD
    return result
=======

    return result

def test_image(bytes):
    inp = np.asarray(bytearray(bytes), dtype=np.uint8)
    im = cv2.imdecode(np.fromstring(inp, dtype=np.uint8), cv2.IMREAD_COLOR)

    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
    #emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions
    emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    # Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    # Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    # Cut and save face
    result=''
    for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
        gray = gray[y:y + h, x:x + w]  # Cut the frame to size
        try:
            out = cv2.resize(gray, (200, 200))  # Resize face so all images have same size
            print("Wynik: {:}".format(fishface.predict(out)))
            result = emotions[np.argmax(fishface.predict(out))]
        except Exception as e:
            print('Blad: {:}'.format(e))
            pass  # If error, pass file

    return result

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    fishface.save('model.xml')
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
>>>>>>> bbff5f004a4f8ffc16a6925515317a5fe3918ef4

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

<<<<<<< HEAD
=======
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}

#Now run it
metascore = []
correct = run_recognizer()
print("got", correct, "percent correct!")
metascore.append(correct)


>>>>>>> bbff5f004a4f8ffc16a6925515317a5fe3918ef4
@app.route('/')
def sessions():
    return render_template('index.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    try:
        #img_data = 'data:image/jpeg;base64,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'
        img_data=json['photo']
        string_data = img_data[23:]
        img = base64.b64decode(string_data)
<<<<<<< HEAD
        #filename = 'some_image.jpeg'  # I assume you have a way of picking unique filenames
        #with open(filename, 'wb') as f:
        #    f.write(img)
        result = test_image(img)
        json['emotion'] = result
    except Exception as e:
        #print('Blad: {:}'.format(e))
=======
        filename = 'some_image.jpeg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(img)
        result = test_image(img)
        json['emotion'] = result
    except Exception as e:
        print('Blad: {:}'.format(e))
>>>>>>> bbff5f004a4f8ffc16a6925515317a5fe3918ef4
        json['emotion'] = -1
        pass
    print('send response: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)

if __name__ == '__main__':
    socketio.run(app, debug=True)