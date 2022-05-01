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
ap.add_argument("-p", "--images_path", required=True,
                help="path to images")
ap.add_argument("-s", "--scale", required=False,
                help="Amount of scaling", default=None)
ap.add_argument("-w", "--width", required=False,
                help="Image width", default=None)

args = vars(ap.parse_args())

WRITE_FOLDER = "scaled_set"
RELATIVE_ROOT = Path(args["images_path"])
IMAGES_DIR = os.path.join(Path(__file__).parent, RELATIVE_ROOT)

if args["scale"] is not None and args["width"] is not None:
    print("Please enter only either scale or width")

if args["scale"] is None and args["width"] is None:
    print("Please enter one of scale or width")

if args["scale"]:
    scale = float(args["scale"])

os. makedirs(f"{args['images_path']}/{WRITE_FOLDER}/", exist_ok=True)

for filename in os.listdir(IMAGES_DIR):

    if filename.endswith(".jpg"):
        new_image = Image.open(os.path.join(IMAGES_DIR, filename))
        if args["width"]:
            if new_image.size[0] < float(args["width"]):
                print(f"{filename}'s width is smaller than required width which is {args['width']}. Keeping original "
                      f"width.")
                scale = 1.0
            else:
                scale = float(args["width"]) / new_image.size[0]

        scaled_width = int(new_image.size[0] * scale)
        scaled_height = int(new_image.size[1] * scale)
        scaled_image = new_image.resize((scaled_width, scaled_height))
        suffix = "" #f"_scale_{str(scale).replace('.', '')}"
        file_name = f'{filename_no_extension(filename)}{suffix}.jpg'
        scaled_image.save(os.path.join(IMAGES_DIR, f'{WRITE_FOLDER}/{file_name}'))

