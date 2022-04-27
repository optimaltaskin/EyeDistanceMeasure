from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from scipy.spatial import ConvexHull
from components.CardMaskProcessor import CardMaskProcessor

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--conf_file", help="full path to the config file",
                    required=True)
parser.add_argument("--check_file", help="full path to the checkpoint file",
                    required=True)
parser.add_argument("--img", help="image file name",
                    required=True)

args = parser.parse_args()
# Choose to use a config and initialize the detector
config = args.conf_file
# Setup a checkpoint file to load
checkpoint = args.check_file

# Set the device to be used for evaluation
device = 'cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

img_filename = args.img
image = cv2.imread(img_filename)


print(f"Processing file: {img_filename}")
result = inference_detector(model, img_filename)
show_result_pyplot(model, img_filename, result, score_thr=0.1)
# model.show_result(img_filename, result, out_file=f"results/{img_filename}")
