from datetime import datetime

from components.logger import init_logger
from components.CardMaskProcessor import CardMaskProcessor
from components.FacialFeaturesExtractor import FacialFeaturesExtractor
from components.helpers import filename_no_extension, load_image
from components.CardMaskProcessor import NoMaskError
from pathlib import Path
import cv2

import os
import argparse
import logging

CARD_HEIGHT = 53.98
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)



def arguement_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", help="full path to the config file",
                        required=True)
    parser.add_argument("--check_file", help="full path to the checkpoint file",
                        required=True)
    parser.add_argument("--print_mask", help="Prints mask on resulting images",
                        required=False, default=False)
    parser.add_argument("--single_step", help="Runs only for one iteration.",
                        required=False, default=False)
    parser.add_argument("--image_file", help="Processes only input image file. Consider using --single_step with that "
                                             "arguement.", required=False, default="")
    parser.add_argument("--disp_refinement", help="Either displays refinement process or not Consider .Consider using "
                                                  "--single_step with that arguement.", required=False, default=False)
    parser.add_argument("--save_to_folder", help="folder location to save results",
                        required=False, default="")
    parser.add_argument("--images_dir", help="Directory to images to process",
                        required=False, default='data/result_validation/')
    return parser.parse_args()


def measure_eye_distance(image, card_height_px, width, height, filename):
    facial_features = FacialFeaturesExtractor(image)

    facial_features.mark_features()
    pupils_distance_pixels = facial_features.get_pupils_distance()

    PRINT_START_POS_V = int(height / 20)
    PRINT_LINE_HEIGHT = int(height / 30)
    FONT_SIZE = height / 1500
    FONT_THICKNESS = int(height / 400)

    mm_per_pixel = CARD_HEIGHT / card_height_px
    pupil_distance = mm_per_pixel * pupils_distance_pixels

    cv2.putText(image, f"Pupils Distance(unscaled): {format(pupil_distance, '.1f')} mm",
                (30, PRINT_START_POS_V),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)
    cv2.putText(image, f"1 Pixel in mm: {format(mm_per_pixel, '.3f')}",
                (30, PRINT_START_POS_V + 3 * PRINT_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)

    cv2.putText(image, f"Pupils Distance(pixels): {int(pupils_distance_pixels)}",
                (30, PRINT_START_POS_V + 2 * PRINT_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)
    cv2.putText(image, f"Card Height(pixels): {int(card_height_px)}",
                (30, PRINT_START_POS_V + 1 * PRINT_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)

    logger.info(f"mm_per_pixel: {mm_per_pixel}")
    logger.info(f"Pupils distance in px: {pupils_distance_pixels}")
    logger.info(f"Pupils distance in mm: {pupil_distance}")
    # show the output image with the face detections + facial landmarks


def main():
    init_logger()
    args = arguement_parse()
    config = args.conf_file
    checkpoint = args.check_file
    if args.save_to_folder == "":
        target_folder = "results/"
    else:
        target_folder = args.save_to_folder

    mask_processor = CardMaskProcessor(config=config, checkpoint=checkpoint)
    directory = args.images_dir
    target_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 target_folder)
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    with open(f"{target_folder}failed_files.txt", "w+") as text_file:
        text_file.write("THIS FILE CONTAINS ALL FAILED DETECTIONS...\n\n")

    for filename in os.listdir(directory):

        if filename.endswith(".jpg"):


            if args.image_file != "":
                full_path = args.image_file
                args.single_step = True
                filename = f"{filename_no_extension(args.image_file)}.jpg"
            else:
                full_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {full_path}")
            image = load_image(full_path)
            height, width = image.shape[:2]

            try:
                mask_processor.initialize(full_path)
            except NoMaskError:
                logger.warning(f"Mask not found for file {filename}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Mask not found!\n")
                if args.single_step:
                    exit()
                continue
            logger.info(f"Mask processor object: {mask_processor}")
            mask_processor.binary_to_grayscale()
            mask_processor.define_contours()
            try:
                mask_processor.create_convexhull(draw_contours=False)
            except AssertionError:
                logger.warning(f"Convex hull was not formed for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Convex hull was not formed!\n")
                if args.single_step:
                    exit()
                continue

            mask_processor.find_card_top_and_bottom(display_refinement=args.disp_refinement,
                                                    save_result=True,
                                                    result_to_image=image)
            if args.print_mask:
                result = mask_processor.inference
                mask_processor.model.show_result(full_path, result, out_file=f"{target_folder}{filename}")
            try:
                card_height_px: float = mask_processor.measure_mean_height_px(mark_to_image=image)
            except AssertionError:
                logger.warning(f"No parallel lines were found for file {full_path}")
                logger.warning(f"Writing to file: {target_folder}failed_files.txt")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Parallel line not found!\n")
                if args.single_step:
                    exit()
                continue
            try:
                measure_eye_distance(image, card_height_px, width, height, full_path)
            except:
                logger.warning(f"Face not detected for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Face not detected!\n")
                if args.single_step:
                    exit()

            result_file_path = os.path.join(Path(__file__).parent,
                                            Path(f"{target_folder}{filename_no_extension(full_path)}--"
                                                 f"{datetime.now().strftime('%d-%m-%Y')}.jpg"))
            logger.info(f"Saving processed file to {result_file_path}")
            cv2.imwrite(result_file_path, image)

            if args.single_step:
                exit()


if __name__ == "__main__":
    main()
