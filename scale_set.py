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
ap.add_argument("-s", "--scale", required=True,
                help="Amount of scaling")

args = vars(ap.parse_args())

WRITE_FOLDER = "scaled_set"
RELATIVE_ROOT = Path(args["images_path"])
IMAGES_DIR = os.path.join(Path(__file__).parent, RELATIVE_ROOT)
WRITE_TO = Path(f"{args['images_path']}/{WRITE_FOLDER}/{filename_no_extension(args['json_file'])}_augmented.json")
LOAD_FROM = Path(f"{args['images_path']}/{args['json_file']}")
scale = float(args["scale"])

with open(LOAD_FROM) as ann_file:
    training_set_data = json.load(ann_file)

last_image_id = training_set_data['images'][-1]['id']
images_list = training_set_data['images'].copy()
annotations_list = training_set_data['annotations']

augment_annot = AugmentAnnotation(images_list)
os. makedirs(f"{args['images_path']}/{WRITE_FOLDER}/", exist_ok=True)

for index, image in enumerate(training_set_data['images']):
    f = image['file_name']
    annotation = annotations_list[index]
    augment_annot.load_new_image_and_annot(image, annotation)

    print("Processing file: {}".format(os.path.join(IMAGES_DIR, f)))
    new_image = Image.open(os.path.join(IMAGES_DIR, f))

    scaled_width = int(new_image.size[0] * scale)
    scaled_height = int(new_image.size[1] * scale)
    scaled_image = new_image.resize((scaled_width, scaled_height))
    suffix = "" #f"_scale_{str(scale).replace('.', '')}"
    file_name = f'{filename_no_extension(f)}{suffix}.jpg'
    scaled_image.save(os.path.join(IMAGES_DIR, f'{WRITE_FOLDER}/{file_name}'))

    new_image_dict, new_annot = augment_annot.get_image_and_annot_scaled(file_name, scale)
    images_list.append(new_image_dict)
    annotations_list.append(new_annot)

del training_set_data['images']
del training_set_data['annotations']
training_set_data['images'] = images_list
training_set_data['annotations'] = annotations_list
with open(WRITE_TO, 'w') as target_json:
    json.dump(training_set_data, target_json)