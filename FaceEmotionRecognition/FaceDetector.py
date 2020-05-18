import numpy as np
import cv2


class FaceDetector(object):
    def __init__(self, path_to_prototxt='../Models/deploy.prototxt',
                 path_to_caffemodel = '../Models/res10_300x300_ssd_iter_140000.caffemodel'):
        self.detection_network = \
            cv2.dnn.readNetFromCaffe(path_to_prototxt,
                                     path_to_caffemodel)

    def face_detect(self, image, desired_confidence=0.9):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.detection_network.setInput(blob)
        detections = self.detection_network.forward()
        face_frames_coordinates=[]
        (h, w) = image.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > desired_confidence:  
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (xmin, ymin, xmax, ymax) = box.astype("int")
                face_frames_coordinates.append((xmin, ymin, xmax, ymax))
        return face_frames_coordinates