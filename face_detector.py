import cv2
import numpy as np
from keras.models import load_model
from keras import utils


emotions_dict = {0: "angry", 1: "happy", 2: "sad"}
model = load_model('./my_model.h5', compile=False)


def get_prediction(img):
    """
    Функция классификации эмоцций

    :param img: изображение от пользователя
    :return: номер класса
    """
    img_array = utils.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array / 255.  # Preprocess the image
    prediction = model.predict(preprocessed_img)
    pred = np.argmax(prediction, axis=1)
    lbl = pred[0]
    return lbl


webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        face_frame = frame[y:y + h, x:x + w]
        img = cv2.resize(face_frame, (224, 224))

        emotion_prediction = emotions_dict[get_prediction(img)]
        cv2.putText(frame, emotion_prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()








