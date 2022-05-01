from datetime import datetime
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from components.CardMaskProcessor import CardMaskProcessor
from components.helpers import get_center_of_points, \
    get_distance_of_points, \
    filename_no_extension, \
    Point
from components.CardMaskProcessor import NoMaskError
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import os
import argparse
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
faceModule = mediapipe.solutions.face_mesh

circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0, 255, 0))

right_eye_landmark_list = [
    469, 470, 471, 472
]  # 133, 155, 173
# 33, 7, 163, 144, 145, 153, 154,
# 157, 158, 159, 160, 161, 246

left_eye_landmark_list = [
    474, 475, 476, 477
]  # 362, 382
# 380, 381, 384, 385, 398,
# 386, 387, 388, 466, 263, 249, 390,
# 373, 374

right_cheek_landmark = [119]
left_cheek_landmark = [348]
fore_head_landmark = 8
nose_landmark = 19

IRIS_WIDTH = 11.7  # mm.
CARD_HEIGHT = 53.98


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


def load_image(img: str):
    try:
        image = cv2.imread(img)
        return image
    except:
        print(f"Image file not found! File name: {img}")
        exit()


def measure_eye_distance(image, card_height_px, width, height, filename):
    with faceModule.FaceMesh(static_image_mode=True,
                             refine_landmarks=True,
                             max_num_faces=1) as face:

        PRINT_START_POS_V = int(height / 20)
        PRINT_LINE_HEIGHT = int(height / 30)
        FONT_SIZE = height / 1500
        FONT_THICKNESS = int(height / 400)
        CIRCLE_RAD = int(height / 800)

        results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks != None:
            for faceLandmarks in results.multi_face_landmarks:
                right_eye_x, right_eye_y = get_center_of_points(
                    np.array(faceLandmarks.landmark)[right_eye_landmark_list],
                    image.shape[:2])
                left_eye_x, left_eye_y = get_center_of_points(np.array(faceLandmarks.landmark)[left_eye_landmark_list],
                                                              image.shape[:2])
                right_cheek_x, right_cheek_y = get_center_of_points(
                    np.array(faceLandmarks.landmark)[right_cheek_landmark],
                    image.shape[:2])
                left_cheek_x, left_cheek_y = get_center_of_points(np.array(faceLandmarks.landmark)[left_cheek_landmark],
                                                                  image.shape[:2])
                # point eye pupils and cheeks
                cv2.circle(image, (right_eye_x, right_eye_y), 3, (255, 0, 0), -1)
                cv2.circle(image, (left_eye_x, left_eye_y), 3, (255, 0, 0), -1)
                cv2.circle(image, (right_cheek_x, right_cheek_y), 2, (0, 0, 255), 2)
                cv2.circle(image, (left_cheek_x, left_cheek_y), 2, (0, 0, 255), 2)

                p1 = Point(x=int(faceLandmarks.landmark[473].x * width), y=int(faceLandmarks.landmark[473].y * height))
                p2 = Point(x=int(faceLandmarks.landmark[468].x * width), y=int(faceLandmarks.landmark[468].y * height))
                left_eye = Point(x=left_eye_x, y=left_eye_y)
                right_eye = Point(x=right_eye_x, y=right_eye_y)
                cv2.circle(image, (p1.x, p1.y), CIRCLE_RAD, (0, 255, 0), -1)
                cv2.circle(image, (p2.x, p2.y), CIRCLE_RAD, (0, 255, 0), -1)
                cv2.line(image, (left_eye.x, left_eye.y), (right_eye.x, right_eye.y), (0, 100, 255), 2)
            pupils_distance_pixels = get_distance_of_points(left_eye, right_eye)

            mm_per_pixel = CARD_HEIGHT / card_height_px
            pupil_distance = mm_per_pixel * pupils_distance_pixels

            cv2.putText(image, f"Pupils Distance(unscaled): {format(pupil_distance, '.1f')} mm",
                        (30, PRINT_START_POS_V),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)
            cv2.putText(image, f"1 Pixel in mm: {format(mm_per_pixel, '.3f')}",
                        (30, PRINT_START_POS_V + 3 * PRINT_LINE_HEIGHT),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)

        else:
            print("Face not detected")
            raise ValueError

        pupils_distance_pixels = get_distance_of_points(left_eye, right_eye)

        cv2.putText(image, f"Pupils Distance(pixels): {int(pupils_distance_pixels)}",
                    (30, PRINT_START_POS_V + 2 * PRINT_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)
        cv2.putText(image, f"Card Height(pixels): {int(card_height_px)}",
                    (30, PRINT_START_POS_V + 1 * PRINT_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)

        print(f"mm_per_pixel: {mm_per_pixel}")
        print(f"Pupils distance in px: {pupils_distance_pixels}")
        print(f"Pupils distance in mm: {pupil_distance}")
        # show the output image with the face detections + facial landmarks


def main():
    args = arguement_parse()
    config = args.conf_file
    checkpoint = args.check_file
    if args.save_to_folder == "":
        target_folder = "results/"
    else:
        target_folder = args.save_to_folder

    device = 'cuda:0'
    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

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
                filename = f"{filename_no_extension(args.image_file)}.jpg"
            else:
                full_path = os.path.join(directory, filename)
            print(f"Processing file: {full_path}")
            image = load_image(full_path)
            height, width = image.shape[:2]

            try:
                mask_processor = CardMaskProcessor(inference_detector(model, full_path), full_path)
            except NoMaskError:
                print(f"Mask not found for file {filename}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Mask not found!\n")
                if args.single_step:
                    exit()
                continue

            mask_processor.binary_to_grayscale()
            mask_processor.define_contours()
            try:
                mask_processor.create_convexhull(draw_contours=False)
            except AssertionError:
                print(f"Convex hull was not formed for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Convex hull was not formed!\n")
                if args.single_step:
                    exit()
                continue

            mask_processor.find_card_top_and_bottom(display_refinement=args.disp_refinement,
                                                    save_result=True,
                                                    result_to_image=image)
            if args.print_mask:
                result = inference_detector(model, full_path)
                model.show_result(full_path, result, out_file=f"{target_folder}{filename}")
            try:
                card_height_px: float = mask_processor.measure_mean_height_px(mark_to_image=image)
            except AssertionError:
                print(f"No parallel lines were found for file {full_path}")
                print(f"Writing to file: {target_folder}failed_files.txt")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Parallel line not found!\n")
                if args.single_step:
                    exit()
                continue
            try:
                measure_eye_distance(image, card_height_px, width, height, full_path)
            except ValueError:
                print(f"Face not detected for file {full_path}")
                with open(f"{target_folder}failed_files.txt", "a") as text_file:
                    text_file.write(f"{filename} - Face not detected!\n")
                if args.single_step:
                    exit()

            result_file_path = os.path.join(Path(__file__).parent,
                                            Path(f"{target_folder}{filename_no_extension(full_path)}--"
                                                 f"{datetime.now().strftime('%d-%m-%Y')}.jpg"))
            print(f"Saving processed file to {result_file_path}")
            cv2.imwrite(result_file_path, image)

            # cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO)
            # cv2.imshow("Result", image)
            # cv2.resizeWindow("Result", 960, 1280)
            # cv2.waitKey(0)
            if args.single_step:
                exit()


if __name__ == "__main__":
    main()
