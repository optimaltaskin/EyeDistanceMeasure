from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from components.CardMaskProcessor import CardMaskProcessor
from components.CardMaskProcessor import NoMaskError

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import os
import argparse


def binary_to_grayscale(arr: []):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--conf_file", help="full path to the config file",
                    required=True)
parser.add_argument("--check_file", help="full path to the checkpoint file",
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

img = 'data/base_set/validation/87.jpg'
image = cv2.imread(img)

maskProcessor = CardMaskProcessor(inference_detector(model, img), img)

maskProcessor.binary_to_grayscale()
maskProcessor.define_contours()
maskProcessor.create_convexhull(draw_contours=True)
maskProcessor.find_card_top_and_bottom(display_refinement=True)


# directory = 'data/base_set/validation/'
# # directory = 'data/'
# with open("results/failed_files.txt", "w+") as text_file:
#     pass
#
# for filename in os.listdir(directory):
#
#     if filename.endswith(".jpg"):
#         full_path = os.path.join(directory, filename)
#         print(f"Processing file: {full_path}")
#         # result = inference_detector(model, full_path)
#         # # show_result_pyplot(model, full_path, result, score_thr=0.7)
#         # model.show_result(full_path, result, out_file=f"results/{filename}")
#         maskProcessor = CardMaskProcessor(inference_detector(model, full_path), full_path)
#         try:
#             maskProcessor.binary_to_grayscale()
#         except NoMaskError:
#             print(f"Mask not found for file {filename}")
#             with open("results/failed_files.txt", "w+") as text_file:
#                 text_file.write(f"{filename}\n")
#             continue
#
#         maskProcessor.define_contours()
#         maskProcessor.create_convexhull(draw_contours=False)
#         maskProcessor.find_card_top_and_bottom(display_refinement=False)
