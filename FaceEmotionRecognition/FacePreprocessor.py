import numpy as np
import cv2


class FacePreprocessor(object):
    def __init__(self, path_to_cascade_classifier="../Models/cascade.xml", use_alignment=False, leave_blank_areas=False):
        self.use_alignment = use_alignment
        self.leave_blank_areas = leave_blank_areas
        if self.use_alignment:
            self.eye_cascade = cv2.CascadeClassifier(path_to_cascade_classifier)
          
    def preprocess(self, image, face_frames_coordinates, output_image_shape=(48,48)): 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.use_alignment:
            aligned_faces = []
            for xmin, ymin, xmax, ymax in face_frames_coordinates:
                transformed_image = self.transform_if_possible((xmin, ymin, xmax, ymax), image)
                image_after_resize = cv2.resize(transformed_image, output_image_shape)
                normalized_image = self.normalize(image_after_resize)
                aligned_faces.append(normalized_image)
                
            return np.array(aligned_faces)
        else:
            faces = np.array([cv2.resize(image[ymin:ymax,xmin:xmax], output_image_shape) 
            for xmin, ymin, xmax, ymax in face_frames_coordinates])
        return faces

    def transform_if_possible(self, face_frame_coordinates, image, shape_for_haar=(150, 150)):
        eyes_coordinates = self.search_eyes(self.cut_face(image,  face_frame_coordinates), shape_for_haar)
        if len(eyes_coordinates) > 1:
            rotation_matrix = self.get_rotation_parameters(eyes_coordinates, 
                                                           face_frame_coordinates, shape_for_haar)
            return self.transform(image, rotation_matrix, face_frame_coordinates)
        else:
            xmin, ymin, xmax, ymax = face_frame_coordinates
            return image[ymin:ymax,xmin:xmax]
                
    def search_eyes(self, face, shape_for_haar):
        face = cv2.resize(face, shape_for_haar)
        eyes = self.eye_cascade.detectMultiScale(face[0:95], 1.02, minSize=(20, 20), maxSize=(50, 50))
        eyes_coordinates = [(x + w//2, y + h//2) for x, y, w, h in eyes[:2]]
        eyes_coordinates.sort()
        return eyes_coordinates

    def cut_face(self, image,  face_frame_coordinates):
        xmin, ymin, xmax, ymax = face_frame_coordinates
        return image[ymin:ymax,xmin:xmax]    

    def get_rotation_parameters(self, eyes_coordinates, face_frame_coordinates, shape_for_haar, scale=0.8):
        xmin, ymin, xmax, ymax = face_frame_coordinates
        dx = eyes_coordinates[1][0] - eyes_coordinates[0][0]
        dy = eyes_coordinates[1][1] - eyes_coordinates[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        coordinates_of_point_between_eyes = ((eyes_coordinates[0][0]+eyes_coordinates[1][0])//2,
                                                 (eyes_coordinates[0][1]+eyes_coordinates[1][1])//2)
        relative_x = coordinates_of_point_between_eyes[0] / 150 
        relative_y = coordinates_of_point_between_eyes[1] / 150
        coordinates_of_point_between_eyes_on_origin_face = \
            (int((xmax - xmin) * relative_x), int((ymax - ymin) * relative_y))
        pivot_point_coordinates = (xmin + coordinates_of_point_between_eyes_on_origin_face[0],
                                   ymin + coordinates_of_point_between_eyes_on_origin_face[1])
        rotation_matrix = cv2.getRotationMatrix2D(pivot_point_coordinates, angle, scale)
        desired_position_of_point_between_eyes_in_image_x = shape_for_haar[1] * 0.45 
        rotation_matrix[0, 2] += (desired_position_of_point_between_eyes_in_image_x -
                                  coordinates_of_point_between_eyes[0])
        return rotation_matrix
    
    def transform(self, image, rotation_matrix, face_frame_coordinates):
        xmin, ymin, xmax, ymax = face_frame_coordinates
        rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[:2], flags=cv2.INTER_CUBIC)
        face = rotated_image[ymin:ymax,xmin:xmax]
        return face

    def normalize(self, face_image):
        return face_image / 255