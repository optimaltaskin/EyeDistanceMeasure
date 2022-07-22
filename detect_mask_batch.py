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
import time

CARD_HEIGHT = 53.98
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)

# python detect_mask_batch.py --conf_file=results/tuning_results/AdamW/mask7/set_base_set_bbone50_m_7_lr_0.0001_wd_0.001/config_set_base_set_bbone50_m_7_lr_0.0001_wd_0.001.py --check_file=results/tuning_results/AdamW/mask7/set_base_set_bbone50_m_7_lr_0.0001_wd_0.001/pth_set_base_set_bbone50_m_7_lr_0.0001_wd_0.001.pth --print_mask=True --save_to_folder=results/tuning_results/temp/ --images_dir=data/sil/



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
    parser.add_argument("--use_card_roi", help="Uses cropped card roi instead of full image for mask processing.",
                         default=False)
    return parser.parse_args()


def measure_eye_distance(facial_features, card_height_px, width, height, image):

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
    script_exec_time = time.time()
    init_logger()
    args = arguement_parse()
    config = args.conf_file
    checkpoint = args.check_file
    if args.save_to_folder == "":
        target_folder = "results/"
    else:
        target_folder = args.save_to_folder
    init_time = time.time()
    mask_processor = CardMaskProcessor(config=config, checkpoint=checkpoint)
    mask_processor_init_time = time.time()
    logger.info(f"Checkpoint loaded in {(mask_processor_init_time - init_time):.2f} seconds.")

    directory = args.images_dir
    print(f"Main work folder is: {directory}")
    target_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 target_folder)
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    with open(f"{target_folder}failed_files.txt", "w+") as text_file:
        text_file.write("THIS FILE CONTAINS ALL FAILED DETECTIONS...\n\n")

    num_processed_images: int = 0
    mean_process_time: float = 0.0
    for filename in os.listdir(directory):

        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):

            file_load_time = time.time()
            suffix = "jpg" if filename.lower().endswith(".jpg") else "png"
            if args.image_file != "":
                full_path = args.image_file
                args.single_step = True
                filename = f"{filename_no_extension(args.image_file)}.{suffix}"
            else:
                full_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {full_path}")
            image = load_image(full_path)
            file_load_complete_time = time.time()
            logger.info(f"File loaded in {(file_load_complete_time - file_load_time):.2f} seconds.")

            height, width = image.shape[:2]

            try:
                facial_features = FacialFeaturesExtractor(image)
            except ValueError:
                logger.warning(f"Face not detected for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Face not detected!\n")
                if args.single_step:
                    exit()
                else:
                    continue

            if args.use_card_roi:
                os.makedirs(os.path.join(directory, "card_rois"), exist_ok=True)
                card_roi_file_path = os.path.join(directory, f"card_rois/{filename}")
                image = facial_features.crop_whole_head()
                height, width = image.shape[:2]
                cv2.imwrite(card_roi_file_path, image)
                image_file = card_roi_file_path
            else:
                image_file = full_path

            try:
                mask_processor.initialize(image_file)
                inference_init_completed_time = time.time()
                logger.info(f"Inference completed in {(inference_init_completed_time - file_load_complete_time):.2f} "
                            f"seconds.")
            except NoMaskError:
                logger.warning(f"Mask not found for file {filename}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Mask not found!\n")
                if args.single_step:
                    exit()
                continue
            mask_processor.binary_to_grayscale()
            mask_processor.define_contours()
            preprocess_completed_time = time.time()
            logger.info(f"Image preprocessing completed in "
                        f"{(preprocess_completed_time - inference_init_completed_time):.2f} seconds.")
            try:
                mask_processor.create_convexhull(draw_contours=False)
                convex_hull_completed_time = time.time()
                logger.info(f"Convex hull calculated in {(convex_hull_completed_time - preprocess_completed_time):.2f} "
                            f"seconds.")
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
            card_top_bottom_found_time = time.time()
            logger.info(f"Card top and bottom edge detected in "
                        f"{(card_top_bottom_found_time - convex_hull_completed_time):.2f} seconds.")
            if args.print_mask:
                result = mask_processor.inference
                mask_processor.model.show_result(mask_processor.image_filename, result, out_file=f"{target_folder}{filename}")

                # card process time is being updated if mask is printed for proper time measurement of mean height
                # calculation
                card_top_bottom_found_time = time.time()
            try:
                card_height_px: float = mask_processor.measure_mean_height_px(mark_to_image=image)
                logger.info(f"Mean height measurement completed in "
                            f"{(card_top_bottom_found_time - convex_hull_completed_time):.2f} seconds.")
            except AssertionError:
                logger.warning(f"No parallel lines were found for file {full_path}")
                logger.warning(f"Writing to file: {target_folder}failed_files.txt")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Parallel line not found!\n")
                if args.single_step:
                    exit()
                continue
            try:
                measure_eye_distance(facial_features, card_height_px, width, height, image)
                eye_dist_measurement_time = time.time()
                logger.info(f"Eye distance measurement completed in "
                            f"{(eye_dist_measurement_time - card_top_bottom_found_time):.2f} seconds.")
            except:
                logger.warning(f"Face not detected for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Face not detected!\n")
                if args.single_step:
                    exit()

            result_file_path = os.path.join(Path(__file__).parent,
                                            Path(f"{target_folder}{filename_no_extension(full_path)}--"
                                                 f"{datetime.now().strftime('%d-%m-%Y')}.{suffix}"))
            logger.info(f"Saving processed file to {result_file_path}")
            cv2.imwrite(result_file_path, image)

            logger.info(f"{filename} is processed in {(time.time() - file_load_time):.2f} seconds")
            mean_process_time += time.time() - file_load_time
            num_processed_images += 1
            if args.single_step:
                break

    logger.info(f"Batch inference completed in {(time.time() - script_exec_time):.2f}")
    logger.info(f"Total {num_processed_images} files were processed.")

if __name__ == "__main__":
    main()

