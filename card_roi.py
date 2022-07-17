from __future__ import print_function
from components.FacialFeaturesExtractor import FacialFeaturesExtractor

import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.')
parser.add_argument('--output', help='Path to output image.')
args = parser.parse_args()
src = cv2.imread(args.input)

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

facial = FacialFeaturesExtractor(src)
facial.mark_features()
img = facial.crop_forehead()

cv2.imwrite(args.output, img)
