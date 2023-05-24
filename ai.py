import tensorflow.keras
import numpy as np
import cv2
import h5py


f = h5py.File('keras_model.h5')

model = tensorflow.keras.models.load_model(f)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def preprocessing(frame):
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))  
    return frame_reshaped


def predict(frame):
    prediction = model.predict(frame)
    return prediction

ret, frame = capture.read()

preprocessed = preprocessing(frame)
prediction = predict(preprocessed)


if (prediction[0,0] > prediction[0,1] and prediction[0,0] > prediction[0,2]):
    player = 'human'
    cv2.putText(frame, 'human', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

elif (prediction[0,1] > prediction[0,0] and prediction[0,1] > prediction[0,2]):
    cv2.putText(frame, 'object', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    player = 'object'

else:
    cv2.putText(frame, 'road', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    player = 'road'


