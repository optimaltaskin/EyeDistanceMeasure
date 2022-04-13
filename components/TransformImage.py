from PIL import Image, ImageEnhance, ImageFilter
from components.helpers import filename_no_extension
from skimage.util import random_noise

import os
import cv2
import numpy as np


class TransformImageBase:

    def transform_image(self) -> str:
        pass


class BlurImage(TransformImageBase):

    def __init__(self, blur_type: ImageFilter, f: str, suffix: str = "", path: str = "/data/training_set"):
        self.blur_filter = blur_type
        self.suffix = suffix
        self.filename = filename_no_extension(f)
        self.path = path

    def transform_image(self, image: Image) -> str:
        blurred_image = image.filter(self.blur_filter)
        file_name = f'{self.filename}_{self.suffix}.jpg'
        blurred_image.save(os.path.join(self.path, file_name))
        return file_name

class NoiseImage(TransformImageBase):

    def __init__(self, suffix: str, f: str, path: str = "/data/training_set", **kwargs):
        self.suffix = suffix
        self.filename = f
        self.path = path
        self.kwargs = kwargs["kwargs"].copy()
        self.kwargs.pop("suffix")

    def transform_image(self, image: []) -> str:

        try:
            noised_img = random_noise(image, **self.kwargs)
        except:
            print("Invalid parameters for random_noise function")
        final_noised_img = (255 * noised_img).astype(np.uint8)
        file_name = f'{filename_no_extension(self.filename)}_{self.suffix}.jpg'
        file_path = os.path.join(self.path, file_name)
        cv2.imwrite(file_path, final_noised_img)
        return file_name


class ContrastImage(TransformImageBase):

    def __init__(self, image: Image, enhance_val: float, f: str, suffix: str = "", path: str = "/data/training_set"):
        self.enhance_val = enhance_val
        self.enhancer = ImageEnhance.Contrast(image)
        self.suffix = suffix
        self.filename = filename_no_extension(f)
        self.path = path

    def transform_image(self) -> str:
        contrast_img = self.enhancer.enhance(self.enhance_val)
        file_name = f'{filename_no_extension(self.filename)}_{self.suffix}.jpg'
        contrast_img.save(os.path.join(self.path, file_name))
        return file_name


class ChangeColorSpace(TransformImageBase):

    def __init__(self, cspace: cv2, suffix: str, f: str, path: str = "/data/training_set"):
        self.new_color_space = cspace
        self.suffix = suffix
        self.filename = f
        self.path = path

    def transform_image(self, image: []) -> str:
        new_image = cv2.cvtColor(image.copy(), self.new_color_space)
        file_name = f'{filename_no_extension(self.filename)}_{self.suffix}.jpg'
        cv2.imwrite(os.path.join(self.path, file_name), new_image)
        return file_name


class ChangeBrightness(TransformImageBase):

    def __init__(self, delta: int, f: str, suffix: str = "", path: str = "/data/training_set"):
        self.delta: int = delta
        self.suffix: str = suffix
        self.filename = filename_no_extension(f)
        self.path = path

    def transform_image(self, image: []) -> str:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if self.delta < 0:
            lim = abs(self.delta)
            v[v < lim] = 0
            v[v >= lim] += np.uint16(self.delta)
        else:
            lim = 255 - abs(self.delta)
            v[v > lim] = 255
            v[v <= lim] += np.uint16(self.delta)

        final_hsv = cv2.merge((h, s, v))
        new_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        file_name = f'{filename_no_extension(self.filename)}_{self.suffix}.jpg'
        cv2.imwrite(os.path.join(self.path,
                                 file_name),
                    new_image)
        return file_name
