from components.helpers import get_center_of_points, Point, \
    get_distance_of_points_tuple

import mediapipe
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FacialFeaturesExtractor:

    # landmark list: https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    # https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    def __init__(self, image: []):

        self.fore_head_landmark = 8
        self.nose_landmark = 19
        self.IRIS_WIDTH = 11.7  # mm.
        self.left_eye_pos: () = None
        self.right_eye_pos: () = None
        self.left_cheek_pos: () = None
        self.right_cheek_pos: () = None

        self.image = image
        self.image_size = self.image.shape[:2]
        self.faceLandmarks = None

        self.process_image()

    def process_image(self):

        assert self.image is not None, "No image file is loaded."

        drawingModule = mediapipe.solutions.drawing_utils
        faceModule = mediapipe.solutions.face_mesh

        circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0, 255, 0))

        with faceModule.FaceMesh(static_image_mode=True,
                                 refine_landmarks=True,
                                 max_num_faces=1) as face:
            results = face.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            if results is None:
                raise ValueError
            try:
                self.faceLandmarks = results.multi_face_landmarks[0]
            except TypeError:
                raise ValueError

            self.right_eye_pos, self.left_eye_pos = self.get_eye_coords()
            self.right_cheek_pos, self.left_cheek_pos = self.get_cheek_coords()

    def get_eye_coords(self):

        right_eye_landmark_list = [
            469, 470, 471, 472
        ]  # 133, 155, 173
        # 33, 7, 163, 144, 145, 153, 154,
        # 157, 158, 159, 160, 161, 246
        left_eye_landmark_list = [
            474, 475, 476, 477
        ]  # 362, 382
        # 380, 381, 384, 385, 398,
        # 386, 387, 388, 466, 263, 249, 390,
        # 373, 374

        right_eye_x, right_eye_y = get_center_of_points(
            np.array(self.faceLandmarks.landmark)[right_eye_landmark_list], self.image_size)
        left_eye_x, left_eye_y = get_center_of_points(
            np.array(self.faceLandmarks.landmark)[left_eye_landmark_list], self.image_size)
        return (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)

    def get_cheek_coords(self):

        right_cheek_landmark = [119]
        left_cheek_landmark = [348]

        right_cheek_x, right_cheek_y = get_center_of_points(
            np.array(self.faceLandmarks.landmark)[right_cheek_landmark], self.image_size)
        left_cheek_x, left_cheek_y = get_center_of_points(
            np.array(self.faceLandmarks.landmark)[left_cheek_landmark], self.image_size)

        return (right_cheek_x, right_cheek_y), (left_cheek_x, left_cheek_y)

    def mark_features(self):

        assert self.image is not None, "No image file is loaded."

        if self.faceLandmarks is None:
            self.process_image()
        height, width = self.image_size
        circle_rad = int(height / 800)

        # point eye pupils and cheeks
        cv2.circle(self.image, self.right_eye_pos, 3, (255, 0, 0), -1)
        cv2.circle(self.image, self.left_eye_pos, 3, (255, 0, 0), -1)
        cv2.circle(self.image, self.right_cheek_pos, 2, (0, 0, 255), 2)
        cv2.circle(self.image, self.left_cheek_pos, 2, (0, 0, 255), 2)

        p1 = Point(x=int(self.faceLandmarks.landmark[473].x * width),
                   y=int(self.faceLandmarks.landmark[473].y * height))
        p2 = Point(x=int(self.faceLandmarks.landmark[468].x * width),
                   y=int(self.faceLandmarks.landmark[468].y * height))
        left_eye = Point(x=self.left_eye_pos[0], y=self.left_eye_pos[1])
        right_eye = Point(x=self.right_eye_pos[0], y=self.right_eye_pos[1])
        cv2.circle(self.image, (p1.x, p1.y), circle_rad, (0, 255, 0), -1)
        cv2.circle(self.image, (p2.x, p2.y), circle_rad, (0, 255, 0), -1)
        cv2.line(self.image, (left_eye.x, left_eye.y), (right_eye.x, right_eye.y), (0, 100, 255), 2)

    def get_pupils_distance(self):
        assert self.left_eye_pos is not None and self.right_eye_pos is not None, "Eye positions arenot calculated " \
                                                                                 "yet. Process them first "
        return get_distance_of_points_tuple(self.left_eye_pos, self.right_eye_pos)

    def crop_forehead(self):
        assert self.image is not None, "Please load an image before calling crop"
        # bottom_left_i = 48
        top_left_i = 103 #67
        # top_right_i = 297
        bottom_right_i = 255#283
        height, width = self.image_size

        corrected_top_y = self.faceLandmarks.landmark[top_left_i].y - 1.5 * \
                          (self.faceLandmarks.landmark[bottom_right_i].y - self.faceLandmarks.landmark[top_left_i].y)

        corrected_top_y = 0 if corrected_top_y <= 0 else corrected_top_y

        return self.image[int(corrected_top_y * height):
                          int(self.faceLandmarks.landmark[bottom_right_i].y * height),
                          int(self.faceLandmarks.landmark[top_left_i].x * width):
                          int(self.faceLandmarks.landmark[bottom_right_i].x * width)]


