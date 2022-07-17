import math

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
        self.forehead_pad_perc = 0.9
        self.forehead_pad_perc_inc = 0.0
        self.head_crop_scale = 0.05

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
        assert self.left_eye_pos is not None and self.right_eye_pos is not None, "Eye positions are not calculated " \
                                                                                 "yet. Process them first "
        return get_distance_of_points_tuple(self.left_eye_pos, self.right_eye_pos)

    def crop_forehead(self, y_displacement: int = 0) -> ([], []):
        """ Returns forehead image with a list containing top_left and bottom_right coordinates in sequence"""
        assert self.image is not None, "Please load an image before calling crop"
        pad_percent = self.forehead_pad_perc + self.forehead_pad_perc_inc

        pupil_dist = int(self.get_pupils_distance())
        pad_y = 0
        pad_x = int(pupil_dist * pad_percent)
        side_length = int(pupil_dist * (1 + 2 * pad_percent))
        right_eye = Point(x=self.right_eye_pos[0], y=self.right_eye_pos[1])
        corrected_top = right_eye.y - side_length
        y_displacement = 100
        if corrected_top < 0:
            pad_y = int(math.fabs(corrected_top))
            corrected_top = 0
        corrected_top = corrected_top + y_displacement
        corrected_bottom = right_eye.y + pad_y + y_displacement
        corrected_left = right_eye.x - pad_x
        corrected_right = right_eye.x - pad_x + side_length
        return self.image[corrected_top: corrected_bottom,
               corrected_left: corrected_right], \
               [(corrected_top, corrected_left), (corrected_bottom, corrected_right)]

    def increase_forehead_crop_pad(self):
        self.forehead_pad_perc_inc += 0.1

    def reset_forehead_crop_pad(self):
        self.forehead_pad_perc_inc = 0.0

    def get_forehead_crop_pad_total(self) -> float:
        return self.forehead_pad_perc + self.forehead_pad_perc_inc

    def arrange_crop_length(self, param1: int, param2: int, ref_length: int):
        # param1 is either top or left edge. Min value is 0
        # param2 is either bottom or right edge. Max value is ref_length
        if param1 < 0 and param2 > ref_length:
            logger.warning(f"param1 and param2 are out of limits. Param1: {param1}, param2: {param2}, "
                           f"ref_length: {ref_length}")
            raise ValueError
        elif param1 < 0:
            param1 = 0
            param2 += -param1
        elif param2 > ref_length:
            param1 -= param2 - ref_length
            param2 = ref_length
        logger.debug(f"param1: {param1}, param2: {param2}, ref_length: {ref_length}")
        assert (0 <= param1 <= ref_length), "Calculated crop edges are out of image"
        assert (0 <= param2 <= ref_length), "Calculated crop edges are out of image"

        return param1, param2

    def crop_whole_head(self):
        # returns head image with an aspect ratio of 1
        assert self.image is not None, "Please load an image before calling crop"
        # left: 234, bottom: 152, right: 454, top: 10
        height, width = self.image_size
        width_pad = width * self.head_crop_scale
        height_pad = height * self.head_crop_scale
        top = int(self.faceLandmarks.landmark[10].y * height - 2 * height_pad)
        left = int(self.faceLandmarks.landmark[234].x * width - width_pad)
        bottom = int(self.faceLandmarks.landmark[152].y * height + height_pad)
        right = int(self.faceLandmarks.landmark[454].x * width + width_pad)

        size_x = right - left
        size_y = bottom - top
        logger.debug(f"size_x: {size_x}, size_y: {size_y}")
        logger.debug(f"top: {top}, bottom: {bottom}")
        logger.debug(f"left: {left}, right: {right}")

        if size_x > size_y:
            half_diff = int(size_x / 2)
            mid_point = top + int(size_y / 2)
            top = mid_point - half_diff
            bottom = mid_point + half_diff
        else:
            half_diff = int(size_y / 2)
            mid_point = left + int(size_x / 2)
            left = mid_point - half_diff
            right = mid_point + half_diff

        logger.debug(f"After making aspect ratio 1 => top: {top}, bottom: {bottom}")
        logger.debug(f"After making aspect ratio 1 => left: {left}, right: {right}")

        top, bottom = self.arrange_crop_length(top, bottom, height)
        left, right = self.arrange_crop_length(left, right, width)

        return self.image[top: bottom, left: right]