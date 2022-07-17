from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector

import math
import random
import numpy as np
import cv2
import os
import logging
import mmcv
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# todo: implement a recursive algorith which will execute when no parallel lines were found.
#  The algorithm will reduce parameters such as min_segment_length, and will also try smaller
#  contours contained in self.hull_list. It will run until a calculation is done

# todo: Implement a logical check, when pupillary distance is for instance lower than 50 mm and
#  higher than 85 mm, it will re run as above. Such length will be assumed to be impossible for
#  any human.


# todo: Onden kart ile fotograf cekerken, ayni anda yandan da fotografini cek. Boylece en dogru sonucu
#  hangi acidan aldigini kontrol et.

class NoMaskError(Exception):
    pass


r = lambda: random.randint(0, 255)
VERTICAL_SLOPE = 9999.0

def imshow_revised(image, title: str):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, image)
    cv2.resizeWindow(title, 960, 1280)


class Segment:
    def __init__(self, point1: [], point2: []):
        self.length: float = 0.0
        self.slope: float = 0.0
        self.point1: [] = point1
        self.point2: [] = point2

        self.calculate_properties()

    def calculate_properties(self):
        self.length = math.dist(self.point1, self.point2)
        if abs(self.point2[0] - self.point1[0]) < 0.01:
            self.slope = VERTICAL_SLOPE
        else:
            self.slope = (self.point2[1] - self.point1[1]) / (self.point2[0] - self.point1[0])

        # self.slope = max(min(self.slope, 10000.0), 0.00001)


class ParallelSegment:

    def __init__(self):
        self.segments_list: [] = []

    def add_segment(self, segment: Segment):
        self.segments_list.append(segment)

    def len_segments(self) -> int:
        return len(self.segments_list)

    def contains_segment(self, segment: Segment) -> bool:
        return segment in self.segments_list

    def copy_list(self):
        return self.segments_list.copy()


class CardMaskProcessor:

    def __init__(self, config: str, checkpoint: str):
        logger.info("Card Mask Processor is being generated...")
        self.inference: () = None
        self.image_filename: str = None
        self.image = None
        self.grayscale_mask: [] = None
        self.contours: [] = None
        self.hierarchy = None
        self.grayscale_threshold: int = None
        self.hull_list: [] = None
        self.segments: [] = None
        self.parallel_segments: [] = None
        self.min_segment_length = None
        self.model = None

        device = 'cuda:0'
        config = mmcv.Config.fromfile(config)
        config.model.pretrained = None
        self.model = build_detector(config.model)
        checkpoint = load_checkpoint(self.model, checkpoint, map_location=device)
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.model.cfg = config
        self.model.to(device)
        self.model.eval()



    def initialize(self, image_filename: str, threshold: int = 100, min_segment_length: float = 20.0):
        logger.info("Initializing card mask processor")
        self.inference: () = inference_detector(self.model, image_filename)
        self.image_filename: str = image_filename
        self.grayscale_threshold: int = threshold
        self.min_segment_length = min_segment_length

        try:
            self.mask = self.inference[1][0][0]
            self.bounding_box = self.inference[0][0]
        except IndexError:
            logger.error(f"Warning! No mask found for file {self.image_filename}")
            raise NoMaskError

        self.min_slope_dif = 0.15

        # lines below delete any previously processed data. Otherwise, with each new processed file, previous data is
        # concatenated with new results
        self.hull_list: [] = []
        self.segments: [] = []
        self.parallel_segments: [] = []
        self.contours: [] = []
        logger.info("Card mask processor initialization completed.")

    def binary_to_grayscale(self):
        logger.info("Changing from binary to grayscale image")
        mask_np = (np.array(self.mask) * 255.0).astype('uint8')
        mask_stacked = np.stack((mask_np,) * 3, -1)
        self.grayscale_mask = cv2.cvtColor(mask_stacked, cv2.COLOR_BGR2GRAY)

    def define_contours(self):
        logger.info("Drawing contours of mask.")
        ret, thresh_img = cv2.threshold(self.grayscale_mask, self.grayscale_threshold, 255, cv2.THRESH_BINARY)
        self.contours, self.hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def create_convexhull(self, draw_contours: bool = True):
        logger.info("Creating convex hull.")
        self.hull_list = []
        for i in range(len(self.contours)):
            hull = cv2.convexHull(self.contours[i])
            self.hull_list.append(hull)

        if draw_contours:
            image = cv2.imread(self.image_filename)
            for i in range(len(self.contours)):
                color = (255, 0, 0)
                cv2.drawContours(image, self.hull_list, i, color)

            imshow_revised(image, 'Contours')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        assert len(self.hull_list) != 0, "No convex hull was formed!"

    def generate_segments(self):
        # sort contours according to their areas. Biggest contour first
        logger.info("Generating segments.")
        contours = sorted(self.hull_list, key=lambda x: cv2.contourArea(x), reverse=True)
        for i, h in enumerate(contours[0]):
            logger.debug(f"Segment {i} of {len(contours[0]) - 1}")
            p1 = h[0]
            i2 = (i + 1) % (len(contours[0]))
            p2 = contours[0][i2][0]
            s = Segment([p1[0], p1[1]], [p2[0], p2[1]])
            self.segments.append(s)
            logger.debug(f"{i}: {p1}   {i2}: {p2}")

    def get_first_non_zero_length_index(self) -> int:
        for i, s in enumerate(self.segments):
            if s.length != 0.0:
                return i

    def draw_segments(self, title: str, result_to_image: [] = None):
        if result_to_image is not None:
            image = result_to_image
        else:
            image = self.image.copy()
        for s in self.segments:
            cv2.line(image, s.point1, s.point2, (r(), r(), r()), 2)
        imshow_revised(image, title)

    def refine_by_similarity(self, display_refinement):
        logger.info("Refining by similarity...")
        if display_refinement:
            image = self.image.copy()
            # mark all points for debugging purpose
            for t, i in enumerate(self.segments):
                color = (255, 0, 0)
                cv2.circle(image, (i.point1[0], i.point1[1]), 2, color, 2)
            i = self.segments[-1]
            cv2.circle(image, (i.point2[0], i.point2[1]), 2, color, 2)

        for i in range(len(self.segments) - 1):

            s1 = self.segments[i]
            index_next = (i + 1) % (len(self.segments))
            s2 = self.segments[index_next]

            # check if segments are sequential
            if s1.point1 != s2.point2 and s1.point2 != s2.point1:
                continue

            slope_dif = abs(s1.slope - s2.slope)
            logger.debug(f"i1: {i}   and     i2: {index_next}")
            logger.debug(f"Slopes diff: {slope_dif}     Min slope dif: {self.min_slope_dif}")
            if slope_dif < self.min_slope_dif:
                logger.debug(f"segment no: {i} will be deleted.")
                logger.debug(f"Changing seg#{index_next}'s point1 from {self.segments[index_next].point1} to "
                             f"{self.segments[i].point1}")
                logger.debug(f"segment {i} data was: {self.segments[i].__dict__}")
                logger.debug(f"segment {index_next} data was: {self.segments[index_next].__dict__}")
                self.segments[index_next].point1 = self.segments[i].point1
                self.segments[index_next].calculate_properties()
                self.segments[i].length = 0.0

                if display_refinement:
                    cv2.putText(image, f"{index_next}", (s2.point1[0], s2.point1[1]),
                                cv2.FONT_HERSHEY_PLAIN,
                                0.5, (0, 255, 0), 1)

        # delete segments with length 0.0
        for i in range(len(self.segments) - 1, -1, -1):
            if self.segments[i].length == 0.0:
                del self.segments[i]

        if display_refinement:
            self.draw_segments('Segment lines - Refine by similarity')
            imshow_revised(image, 'Segment Points')

    def refine_by_length(self, display_refinement: bool):
        logger.info("Refining by length...")
        # filter segments by length
        delete_items: [] = []
        for i, s in enumerate(self.segments):
            if s.length < self.min_segment_length:
                delete_items.append(i)

        delete_items.sort(reverse=True)
        for i in delete_items:
            del self.segments[i]

        # draw segments
        if display_refinement:
            self.draw_segments(title='Segment Lines - Refinement by length')

    def is_segment_processed(self, segment: Segment) -> bool:
        for p in self.parallel_segments:
            if p.contains_segment(segment):
                return True
        return False

    def match_parallel_lines(self, save_result: bool, result_to_image):
        logger.info("Matching parallel lines...")
        for i, s in enumerate(self.segments):
            remaining_segments = self.segments.copy()
            remaining_segments.pop(i)
            parallel_s = ParallelSegment()
            parallel_s.add_segment(s)
            logger.debug(f"Processing segment:\n{s.__dict__}")

            # is the segment processed before?
            if self.is_segment_processed(s):
                continue

            if s.slope == VERTICAL_SLOPE:
                continue

            if self.check_segment_coincide_bbox(s):
                continue

            for rem_segment in remaining_segments:
                if self.check_segment_coincide_bbox(rem_segment):
                    continue
                slope_dif = abs(s.slope - rem_segment.slope)
                logger.debug(f"Slope difference is {slope_dif}")
                # multiplication below is required to prevent adding more than two segments in the
                # parallel segments list
                if slope_dif < self.min_slope_dif * 0.95:
                    logger.debug(f"Parallel segment found.")
                    logger.debug(f"Other seg: {rem_segment.__dict__}")
                    new_parallel_segments = ParallelSegment()
                    new_parallel_segments.segments_list = parallel_s.copy_list()
                    new_parallel_segments.add_segment(rem_segment)
                    self.parallel_segments.append(new_parallel_segments)

        if save_result:
            # draw parallel segments
            if result_to_image is not None:
                image_parallel = result_to_image
            elif not self.image:
                self.image = cv2.imread(self.image_filename)
                image_parallel = self.image.copy()
            else:
                image_parallel = self.image.copy()

            for p in self.parallel_segments:
                color = (r(), r(), r())
                logger.debug(f"Parallel segment:\n{p.__dict__}")

                for s in p.segments_list:
                    cv2.line(image_parallel, s.point1, s.point2, color, 2)
                    cv2.putText(image_parallel, f"{s.point1}", (s.point1[0], s.point1[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.putText(image_parallel, f"{s.point2}", (s.point2[0], s.point2[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    mid_point = (int((s.point1[0] + s.point2[0]) / 2), int((s.point1[1] + s.point2[1]) / 2))
                    cv2.putText(image_parallel, f"{mid_point}", (mid_point[0], mid_point[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # imshow_revised(image_parallel, "Parallel segments marked")
            if result_to_image is None:
                basename = os.path.basename(self.image_filename)
                filename = os.path.splitext(basename)
                cv2.imwrite(f"results/{filename[0]}.jpg", image_parallel)
                with open(f"results/{filename[0]}.jpg_parallel_segments.txt", "w+") as text_file:
                    text_file.write(f"{self.image_filename}\n")
                    for p in self.parallel_segments:
                        for s in p.segments_list:
                            text_file.write(f"{s.__dict__}\n")

    def find_card_top_and_bottom(self, display_refinement: bool = False, save_result: bool = False,
                                 result_to_image=None):
        logger.info("Trying to define card top and bottom edges...")
        self.generate_segments()
        if display_refinement:
            self.image = cv2.imread(self.image_filename)

        self.refine_by_similarity(display_refinement)
        self.refine_by_length(display_refinement)
        self.match_parallel_lines(save_result, result_to_image)

        if display_refinement:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def check_segment_coincide_bbox(self, s: Segment) -> bool:
        logger.info("Checking if segment coincides with bounding box")
        left = int(self.bounding_box[0][0]) + 1
        top = int(self.bounding_box[0][1]) + 1
        right = int(self.bounding_box[0][2]) - 1
        bottom = int(self.bounding_box[0][3]) - 1
        if s.point1[1] == s.point2[1]:
            if s.point1[1] <= top or s.point1[1] >= bottom:
                return True
        elif s.point1[0] == s.point2[0]:
            if s.point1[0] <= left or s.point1[0] >= right:
                return True
        return False

    def check_p_seg_coincide_bbox(self, p: ParallelSegment) -> bool:
        logger.info("Checking if parallel segment coincides with bounding box")
        left = int(self.bounding_box[0][0]) + 1
        top = int(self.bounding_box[0][1]) + 1
        right = int(self.bounding_box[0][2]) - 1
        bottom = int(self.bounding_box[0][3]) - 1
        for s in p.segments_list:
            if s.point1[1] == s.point2[1]:
                if s.point1[1] <= top or s.point1[1] >= bottom:
                    return True
            elif s.point1[0] == s.point2[0]:
                if s.point1[0] <= left or s.point1[0] >= right:
                    return True
        return False

    def select_parallel_segment(self) -> ParallelSegment:
        logger.info("Selecting strongest candidate for card top and bottom edges among parallel segments...")
        most_horizontal: float = 99.0
        p_segment: ParallelSegment = None
        p_segment_touching_bbox: ParallelSegment = None
        for p in self.parallel_segments:
            try:
                slope = p.segments_list[0].slope
                assert len(p.segments_list) >= 2, "Parallel segment contains too low segment lines..."
            except IndexError:
                logger.warning(f"Segment list contains too low or no segments!. Parallel Segment "
                               f"containing faulty segment: 'n{p}")
                raise ValueError
            if self.check_p_seg_coincide_bbox(p) or \
                    self.check_p_seg_coincide_bbox(p):
                p_segment_touching_bbox = p
                continue

            if slope < most_horizontal:
                if not self.check_p_seg_coincide_bbox(p):
                    p_segment = p
                    most_horizontal = slope

        assert (p_segment is not None) or (p_segment_touching_bbox is not None), \
            "Something went wrong with parallel segment selection. Multiple parallel " \
            "segments contains faulty segment lines. "

        if p_segment is None:
            return p_segment_touching_bbox

        # if a parallel segment contains more than two parallel lines, merge sequential lines
        if len(p_segment.segments_list) > 2:
            logger.info(f"Parallel segment contains more than two segments!")
            for s in p_segment.segments_list:
                logger.info(f"{s.__dict__}")
            # which segments have common point?
            for i in range(len(p_segment.segments_list)):
                s1 = p_segment.segments_list[i]
                index_next = (i + 1) % (len(p_segment.segments_list))
                s2 = p_segment.segments_list[index_next]
                if s1.point2 == s2.point1:
                    logger.debug(f"Merging segments: \n{s1.__dict__}\n{s2.__dict__}")
                    p_segment.segments_list[index_next].point1 = p_segment.segments_list[i].point1
                    p_segment.segments_list[index_next].calculate_properties()
                    p_segment.segments_list[i].length = 0.0
                elif s1.point1 == s2.point2:
                    logger.debug(f"Merging segments: \n{s1.__dict__}\n{s2.__dict__}")
                    p_segment.segments_list[index_next].point2 = p_segment.segments_list[i].point2
                    p_segment.segments_list[index_next].calculate_properties()
                    p_segment.segments_list[i].length = 0.0

            for i in range(len(p_segment.segments_list) - 1, -1, -1):
                if p_segment.segments_list[i].length == 0.0:
                    del p_segment.segments_list[i]

        if p_segment_touching_bbox is not None:
            if p.segments_list[0].slope > 1.0:
                return p_segment_touching_bbox
        return p_segment

    def measure_mean_height_px(self, mark_to_image=None) -> float:
        logger.info("Measuring mean height between top and bottom edges in pixels")
        assert len(self.parallel_segments) != 0, "No parallel segments were formed!"

        height_px = 0.0
        p_segment = self.select_parallel_segment()
        len_segments = len(p_segment.segments_list)

        per_line_top_end = (0, 50000)
        height_line_slope = 0.0
        for i in range(len_segments):
            s1 = p_segment.segments_list[i % len_segments]
            s2 = p_segment.segments_list[(i + 1) % len_segments]

            mid_point = (int((s1.point1[0] + s1.point2[0]) / 2), int((s1.point1[1] + s1.point2[1]) / 2))
            if mid_point[1] < per_line_top_end[1]:
                per_line_top_end = mid_point
                height_line_slope += s1.slope

            if mark_to_image is not None:
                cv2.circle(mark_to_image, (mid_point[0], mid_point[1]), 4, (255, 0, 0))

            x1, y1 = s2.point1[0], s2.point1[1]
            x2, y2 = s2.point2[0], s2.point2[1]
            x0, y0 = mid_point[0], mid_point[1]

            upper_part = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
            lower_part = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            d = upper_part / lower_part

            logger.info(f"Distance from point {mid_point} to segment line is:\n{d}")
            height_px += d

        height_px /= len_segments
        height_line_slope /= len_segments

        # draw calculated height on image - top point will be mid point of top line.
        # calculate lower end of height line
        if mark_to_image is not None:
            if math.fabs(height_line_slope) < 0.01:
                # line is nearly horizontal
                per_line_bottom_end = (per_line_top_end[0], per_line_top_end[1] + int(height_px))
            else:
                slope_perpendicular_line = - 1 / height_line_slope
                logger.debug(f"Segment slope: {height_line_slope}   -    Perpendicular line slope: {slope_perpendicular_line}")
                # normalize slope vector
                slope_vector = (1, slope_perpendicular_line)
                magnitude = math.sqrt(slope_vector[0] * slope_vector[0] +
                                      slope_vector[1] * slope_vector[1])
                normalized_slope = (slope_vector[0] / magnitude, slope_vector[1] / magnitude)
                logger.debug(f"Normalized slope vector: {normalized_slope}")
                if normalized_slope[1] >= 0.0:
                    per_line_bottom_end = (int(height_px * normalized_slope[0] + per_line_top_end[0]),
                                           (int(height_px * normalized_slope[1] + per_line_top_end[1])))
                else:
                    per_line_bottom_end = (int(height_px * normalized_slope[0] * -1 + per_line_top_end[0]),
                                           (int(height_px * normalized_slope[1] * -1 + per_line_top_end[1])))

            cv2.line(mark_to_image, per_line_top_end, per_line_bottom_end, (0, 0, 255), 3)
            validation_height_px: float = math.sqrt((per_line_bottom_end[0] - per_line_top_end[0]) ** 2 +
                                                    (per_line_bottom_end[1] - per_line_top_end[1]) ** 2)
            logger.info(f"Calculated height_px: {height_px}    -     Validation height_px: {validation_height_px}")

        logger.info(f"Mean height of card in px: {height_px}")
        return height_px
