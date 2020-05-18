import numpy as np
from tensorflow.keras.models import load_model


class FaceEmotionRecognizer(object):
    def __init__(self, path_to_network_model="../Models/model_new.h5"):
        self.model=load_model(path_to_network_model)

    def emotion_recognize(self, face_images):
        face_image_list = np.expand_dims(face_images, axis=-1)
        return self.model.predict(face_image_list)