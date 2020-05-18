from EmotionRecognition.FaceDetector import *
from EmotionRecognition.FaceEmotionRecognizer import *
from EmotionRecognition.FacePreprocessor import *
import numpy as np
import cv2


cap = cv2.VideoCapture(0)
detector = FaceDetector()
preprocessor = FacePreprocessor(use_alignment=False)
emotion_recognizer = FaceEmotionRecognizer(path_to_network_model="../Models/model_new.h5")
while 1:
    ret, image = cap.read()
    emotion_list = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Scorn', 'Surprise']
    image = image.copy()
    face_frames_coordinates = detector.face_detect(image) # Координаты лиц
    value = emotion_recognizer.emotion_recognize(preprocessor.preprocess(image, face_frames_coordinates))
    draw_list = list(zip(face_frames_coordinates, value))
    if len(face_frames_coordinates) > 0:
        for (xmin, ymin, xmax, ymax), emotion in draw_list:
            print(xmin, ymin, xmax, ymax, emotion)
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (210,50,22), 2)
            cv2.putText(image, emotion_list[np.where(emotion == np.max(emotion))[0][0]],
                        (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()