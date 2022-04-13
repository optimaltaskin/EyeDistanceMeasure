from pathlib import Path
from skimage.util import random_noise
from PIL import Image, ImageEnhance, ImageFilter
from components.helpers import filename_no_extension
from components.AugmentAnnotation import AugmentAnnotation
from components.TransformImage import BlurImage, NoiseImage, ContrastImage, ChangeColorSpace, ChangeBrightness

import numpy as np
import cv2
import os
import json
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-j", "--json_file", required=True,
                help="name of json file containing annotations")
ap.add_argument("-p", "--images_path", required=True,
                help="path to images")

args = vars(ap.parse_args())

RELATIVE_ROOT = Path(args["images_path"])
IMAGES_DIR = os.path.join(Path(__file__).parent, RELATIVE_ROOT)
WRITE_TO = Path(f"{args['images_path']}/{filename_no_extension(args['json_file'])}_augmented.json")
LOAD_FROM = Path(f"{args['images_path']}/{args['json_file']}")

BRIGHTNESS_FILTERS = [
    {"delta": 90, "suffix": "brightened"},
    {"delta": -90, "suffix": "darkened"}
]

BLUR_FILTERS = [
    # {"filter": ImageFilter.BLUR, "suffix": "blurred_light"},
    {"filter": ImageFilter.BoxBlur(4), "suffix": "blurred_moderate"},
    # {"filter": ImageFilter.BoxBlur(8), "suffix": "blurred_heavy"},
    {"filter": ImageFilter.BoxBlur(12), "suffix": "blurred_extreme"},
]

NOISE_FILTER = [
    {"mode": "gaussian", "var": 0.12, "suffix": "gaussian"},
    # {"mode": "gaussian", "var": 0.06, "suffix": "gaussian"},
    {"mode": 's&p', "amount": 0.12, "suffix": "snp"},
    # {"mode": 'speckle', "mean": 0.5, "suffix": "speckle"}
]

CONTRAST_FILTER = [
    {"enhance": 0.25, "suffix": "contrast_025"},
    # {"enhance": 0.6, "suffix": "contrast_060"},
    # {"enhance": 1.5, "suffix": "contrast_150"},
    {"enhance": 2.0, "suffix": "contrast_200"},
]

COLOR_SPACES = [
    # {"space": cv2.COLOR_BGR2LAB, "suffix": "cspace_lab"},
    {"space": cv2.COLOR_BGR2YCrCb, "suffix": "cspace_lYCrCb"},
    {"space": cv2.COLOR_BGR2HSV, "suffix": "cspace_hsv"},
    {"space": cv2.COLOR_BGR2GRAY, "suffix": "cspace_gray"},
]
with open(LOAD_FROM) as ann_file:
    training_set_data = json.load(ann_file)

last_image_id = training_set_data['images'][-1]['id']
images_list = training_set_data['images'].copy()
annotations_list = training_set_data['annotations']

augment_annot = AugmentAnnotation(images_list)

for index, image in enumerate(training_set_data['images']):
    f = image['file_name']
    print("Processing file: {}".format(os.path.join(IMAGES_DIR, f)))

    annotation = annotations_list[index]
    augment_annot.load_new_image_and_annot(image, annotation)

    new_image_pil = Image.open(os.path.join(IMAGES_DIR, f))
    new_image_cv = cv2.imread(os.path.join(IMAGES_DIR, f), 1)  # 1 for color image

    for filt in BRIGHTNESS_FILTERS:
        br_filter = ChangeBrightness(delta=filt["delta"],
                                     f=f,
                                     suffix=filt["suffix"],
                                     path=IMAGES_DIR)
        file_name = br_filter.transform_image(new_image_cv.copy())
        new_image_dict, new_annot = augment_annot.get_image_and_annot_no_scaling(file_name)
        images_list.append(new_image_dict)
        annotations_list.append(new_annot)

    for filter_dict in BLUR_FILTERS:
        blur_filter = BlurImage(filter_dict["filter"],
                                f,
                                suffix=filter_dict["suffix"],
                                path=IMAGES_DIR)
        file_name = blur_filter.transform_image(new_image_pil.copy())
        new_image_dict, new_annot = augment_annot.get_image_and_annot_no_scaling(file_name)
        images_list.append(new_image_dict)
        annotations_list.append(new_annot)

    for noise_filter_param in NOISE_FILTER:
        # print(noise_filter_param)
        noise_filter = NoiseImage(suffix=noise_filter_param["suffix"],
                                  f=f,
                                  path=IMAGES_DIR,
                                  kwargs=noise_filter_param)
        file_name = noise_filter.transform_image(new_image_cv.copy())
        new_image_dict, new_annot = augment_annot.get_image_and_annot_no_scaling(file_name)
        images_list.append(new_image_dict)
        annotations_list.append(new_annot)

    for contrast_params in CONTRAST_FILTER:
        contrast_filter = ContrastImage(image=new_image_pil,
                                        enhance_val=contrast_params["enhance"],
                                        f=f,
                                        suffix=contrast_params["suffix"],
                                        path=IMAGES_DIR)
        file_name = contrast_filter.transform_image()
        new_image_dict, new_annot = augment_annot.get_image_and_annot_no_scaling(file_name)
        images_list.append(new_image_dict)
        annotations_list.append(new_annot)

    for new_space in COLOR_SPACES:
        change_space = ChangeColorSpace(cspace=new_space["space"],
                                        suffix=new_space["suffix"],
                                        f=f,
                                        path=IMAGES_DIR)
        file_name = change_space.transform_image(image=new_image_cv)
        new_image_dict, new_annot = augment_annot.get_image_and_annot_no_scaling(file_name)
        images_list.append(new_image_dict)
        annotations_list.append(new_annot)

# first write to file than read all images and annoatations to scale all of them
# del training_set_data['images']
# del training_set_data['annotations']
# training_set_data['images'] = images_list
# training_set_data['annotations'] = annotations_list
# with open(WRITE_TO, 'w') as target_json:
#     json.dump(training_set_data, target_json)
#
# with open(WRITE_TO) as ann_file:
#     training_set_data = json.load(ann_file)

# last_image_id = training_set_data['images'][-1]['id']
# images_list = training_set_data['images'].copy()
# annotations_list = training_set_data['annotations']
#
# augment_annot = AugmentAnnotation(images_list)
#
# for index, image in enumerate(training_set_data['images']):
#     f = image['file_name']
#     annotation = annotations_list[index]
#     augment_annot.load_new_image_and_annot(image, annotation)
#
#     print("Processing file: {}".format(os.path.join(IMAGES_DIR, f)))
#     new_image = Image.open(os.path.join(IMAGES_DIR, f))
#
#     # for scale in np.arange(0.5, 1.0, 0.1):
#     for scale in [0.5]:
#         scaled_width = int(image['width'] * scale)
#         scaled_height = int(image['height'] * scale)
#         scaled_image = new_image.resize((scaled_width, scaled_height))
#         suffix = f"_scale_{str(scale).replace('.', '')}"
#         file_name = f'{filename_no_extension(f)}{suffix}.jpg'
#         scaled_image.save(os.path.join(IMAGES_DIR, file_name))
#
#         new_image_dict, new_annot = augment_annot.get_image_and_annot_scaled(file_name, scale)
#         images_list.append(new_image_dict)
#         annotations_list.append(new_annot)
#

training_set_data['images'] = images_list
training_set_data['annotations'] = annotations_list
with open(WRITE_TO, 'w') as target_json:
    json.dump(training_set_data, target_json)
