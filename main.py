# Import section
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import glob
import random

# Predictable emotions
emotions = ["neutral", "anger", "disgust", "happy", "sadness", "surprise"]

# Get images list for emotion
def getImageList(emotion):
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    # Create training and test set
    trainingFiles = files[:int(len(files)*0.7)]
    testFiles = files[-int(len(files)*0.3):]
    return trainingFiles, testFiles

def createTrainingAndTestingSets():
    trainingImages = []
    trainingLabels = []
    testImages = []
    testLabels = []
    for emotion in emotions:
        trainingFiles, testFiles = getImageList(emotion)
        for filePath in trainingFiles:
            trainingImages.append(cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2GRAY))
            trainingLabels.append(emotions.index(emotion))
        for filePath in testFiles:
            testImages.append(cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2GRAY))
            testLabels.append(emotions.index(emotion))
    return trainingImages, trainingLabels, testImages, testLabels

# Train classifier and calculate accuracy
def trainClassifier():
    trainingImages, trainingLabels, testImages, testLabels = createTrainingAndTestingSets()
    fishface.train(trainingImages, np.asarray(trainingLabels))

    trues = 0
    falses = 0
    for i, image in enumerate(testImages):
        predictedEmotion = fishface.predict(image)
        if predictedEmotion == testLabels[i]:
            trues += 1
        else:
            falses += 1
    return (trues/(trues + falses))*100

# Detect face on image and predict its' emotion
def predictEmotion(image_bytes):
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_color = cv2.imdecode(np.fromstring(image_array, dtype=np.uint8), cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Read 4 face detection classifiers
    faceDetectionDefaultMethod = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDetectionAlt2Method = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDetectionAltMethod = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDetectionTreeMethod = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    # Detect face
    resultDefaultMethod = faceDetectionDefaultMethod.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    resultAlt2Method = faceDetectionAlt2Method.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    resultAltMethod = faceDetectionAltMethod.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    resultTreeMethod = faceDetectionTreeMethod.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    # Select detected face
    if len(resultDefaultMethod) == 1:
        faceRegion = resultDefaultMethod
    elif len(resultAlt2Method) == 1:
        faceRegion = resultAlt2Method
    elif len(resultAltMethod) == 1:
        faceRegion = resultAltMethod
    elif len(resultTreeMethod) == 1:
        faceRegion = resultTreeMethod
    else:
        faceRegion = ""

    # Select best face detection
    x, y, w, h = faceRegion[0]

    try:
        # Preprocessing
        image_gray = image_gray[y:y + h, x:x + w]
        image_test = cv2.resize(image_gray, (200, 200))
        # Prediction
        result = emotions[fishface.predict(image_test)[0]]
    except Exception as e:
        #print('Prediction error: {:}'.format(e))
        result = ''
        pass

    return result


# Run Flask serwer
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

# Create and train classifier
fishface = cv2.face.FisherFaceRecognizer_create()
result = trainClassifier()
print("Classifier accuracy: {:}%".format(result))

@app.route('/')
def sessions():
    return render_template('index.html')

# Receive message
def messageReceived(methods=['GET', 'POST']):
    print('Received message.')

# Send message
@socketio.on('sendMessageEvent')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    try:
        image_base=json['photo']
        image_bytes = base64.b64decode(image_base[23:])
        json['emotion'] = predictEmotion(image_bytes)
    except Exception as e:
        print('Error occured: {:}'.format(e))
        json['emotion'] = ''
        pass
    socketio.emit('my response', json, callback=messageReceived)

if __name__ == '__main__':
    socketio.run(app, debug=True)