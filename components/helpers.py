import os
import math


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def filename_no_extension(path: str):
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)

    return filename[0]


def get_center_of_points(landmarks: [], image_shape: ()) -> (int, int):
    center_x = 0
    center_y = 0
    for l in landmarks:
        center_x += l.x * image_shape[1]
        center_y += l.y * image_shape[0]

    center_x = int(center_x / len(landmarks))
    center_y = int(center_y / len(landmarks))
    return center_x, center_y


def get_distance_of_points(point1: Point, point2: Point):
    return math.sqrt(math.fabs(point1.x - point2.x) ** 2 + math.fabs(point1.y - point2.y) ** 2)
