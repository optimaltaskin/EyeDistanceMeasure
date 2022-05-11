import math
import random
import numpy as np
import cv2
import os


class NoMaskError(Exception):
    pass


r = lambda: random.randint(0, 255)


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
            self.slope = 9999.0
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


class CardMaskProcessor:

    def __init__(self, inference: (), image_filename: str, threshold: int = 100, min_segment_length: float = 100.0):

        self.inference: () = inference
        self.image_filename: str = image_filename
        self.image = None
        self.grayscale_mask: [] = None
        self.contours: [] = None
        self.hierarchy = None
        self.grayscale_threshold: int = threshold
        self.hull_list: [] = None
        self.segments: [] = []
        self.parallel_segments: [] = []
        self.min_segment_length = min_segment_length

        try:
            self.mask = self.inference[1][0][0]
            self.bounding_box = self.inference[0][0]
        except IndexError:
            print(f"Warning! No mask found for file {self.image_filename}")
            raise NoMaskError

        # self.min_slope_dif = 5
        self.min_slope_dif = 0.15

    def binary_to_grayscale(self):
        mask_np = (np.array(self.mask) * 255.0).astype('uint8')
        mask_stacked = np.stack((mask_np,) * 3, -1)
        self.grayscale_mask = cv2.cvtColor(mask_stacked, cv2.COLOR_BGR2GRAY)

    def define_contours(self):
        ret, thresh_img = cv2.threshold(self.grayscale_mask, self.grayscale_threshold, 255, cv2.THRESH_BINARY)
        self.contours, self.hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def create_convexhull(self, draw_contours: bool = True):
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
        # for h in self.hull_list[0]:
        #     print(f"{h[0][0]}   {h[0][1]}")
        for i, h in enumerate(self.hull_list[0]):
            # print(f"Segment {i} of {len(self.hull_list[0]) - 1}")
            p1 = h[0]
            i2 = (i + 1) % (len(self.hull_list[0]))
            p2 = self.hull_list[0][i2][0]
            s = Segment([p1[0], p1[1]], [p2[0], p2[1]])
            self.segments.append(s)
            # print(f"{i}: {p1}   {i2}: {p2}")

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
        if display_refinement:
            image = self.image.copy()
            image2 = self.image.copy()
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
            print(f"i1: {i}   and     i2: {index_next}")
            print(f"Slopes diff: {slope_dif}     Min slope dif: {self.min_slope_dif}")
            if slope_dif < self.min_slope_dif:
                print(f"segment no: {i} will be deleted.")
                print(
                    f"Changing seg#{index_next}'s point1 from {self.segments[index_next].point1} to {self.segments[i].point1}")
                print(f"segment {i} data was: {self.segments[i].__dict__}")
                print(f"segment {index_next} data was: {self.segments[index_next].__dict__}")
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
            # for t, i in enumerate(self.segments):
            #     color = (255, 0, 0)
            #     cv2.circle(image2, (i.point1[0], i.point1[1]), 2, color, 2)
            #     cv2.putText(image2, f"{t}", (i.point1[0], i.point1[1]), cv2.FONT_HERSHEY_PLAIN,
            #                 0.5, (0, 255, 0), 1)
            # i = self.segments[-1]
            # cv2.circle(image2, (i.point2[0], i.point2[1]), 2, color, 2)
            # cv2.putText(image2, f"{len(self.segments)}", (i.point2[0], i.point2[1]), cv2.FONT_HERSHEY_PLAIN,
            #             0.5, (0, 255, 0), 1)
            self.draw_segments('Segment lines - Refine by similarity')

            imshow_revised(image, 'Segment Points')
            # imshow_revised(image2, 'Segment Points - Refine by similarity')

    def refine_by_length(self, display_refinement: bool):

        # filter segments by length
        min_length: float = 20.0
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

    def refine_coinciding_with_bounding_box(self, result_to_image, display_refinement: bool = False):

        delete_items: [] = []
        # bounding box
        left = int(self.bounding_box[0][0])
        top = int(self.bounding_box[0][1])
        right = int(self.bounding_box[0][2])
        bottom = int(self.bounding_box[0][3])
        if result_to_image is not None:
            cv2.circle(result_to_image, (left, top), 3, (0, 255, 0), thickness=-1)
            cv2.circle(result_to_image, (left, bottom), 3, (0, 255, 0), thickness=-1)
            cv2.circle(result_to_image, (right, top), 3, (0, 255, 0), thickness=-1)
            cv2.circle(result_to_image, (right, bottom), 3, (0, 255, 0), thickness=-1)

        # apply a margin for bounding box filter
        top += 1
        bottom -= 1
        left += 1
        right -= 1

        for i, s in enumerate(self.segments):
            # print(f"Point1: {s.point1}")
            # print(f"Point2: {s.point2}")
            # print(f"Top: {top}")
            # print(f"Bottom: {bottom}")
            # print(f"Left: {left}")
            # print(f"Right: {right}")

            if s.point1[1] == s.point2[1]:
                if s.point1[1] <= top or s.point1[1] >= bottom:
                    print(f"Deleting segment coinciding with bbox: {s.__dict__}")
                    delete_items.append(i)
            elif s.point1[0] == s.point2[0]:
                if s.point1[0] <= left or s.point1[0] >= right:
                    print(f"Deleting segment coinciding with bbox: {s.__dict__}")
                    delete_items.append(i)

        delete_items.sort(reverse=True)
        for i in delete_items:
            del self.segments[i]

        # draw segments
        if display_refinement:
            self.draw_segments('Segment Lines - Refinement by Bounding box')

    def is_segment_processed(self, segment: Segment) -> bool:
        for p in self.parallel_segments:
            if p.contains_segment(segment):
                return True
        return False

    def match_parallel_lines(self, save_result: bool, result_to_image):
        for i, s in enumerate(self.segments):
            remaining_segments = self.segments.copy()
            remaining_segments.pop(i)
            parallel_s = ParallelSegment()
            parallel_s.add_segment(s)
            print(f"Processing segment:\n{s.__dict__}")

            # is the segment processed before?
            if self.is_segment_processed(s):
                continue

            for rem_segment in remaining_segments:
                slope_dif = abs(s.slope - rem_segment.slope)
                # print(f"Slope difference is {slope_dif}")
                # multiplication below is required to prevent adding more than two segments in the
                # parallel segments list
                if slope_dif < self.min_slope_dif * 0.95:
                    print(f"Parallel segment found.")
                    # print(f"Other seg: {rem_segment.__dict__}")
                    parallel_s.add_segment(rem_segment)

            if parallel_s.len_segments() > 1:
                print("Adding parallel segments.")
                self.parallel_segments.append(parallel_s)

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
                # print(f"Parallel segment:\n{p.__dict__}")

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

        self.generate_segments()
        if display_refinement:
            self.image = cv2.imread(self.image_filename)

        # print("Segments before refinement:\n")
        # for s in self.segments:
        #     print(f"{s.__dict__}")
        self.refine_by_similarity(display_refinement)
        # self.refine_coinciding_with_bounding_box(result_to_image, display_refinement)
        self.refine_by_length(display_refinement)
        # print("Remaining segments after refinement:\n")
        # for s in self.segments:
        #     print(f"{s.__dict__}")

        self.match_parallel_lines(save_result, result_to_image)

        if display_refinement:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def check_p_seg_coincide_bbox(self, p: ParallelSegment) -> bool:
        # result = True
        left = int(self.bounding_box[0][0]) + 1
        top = int(self.bounding_box[0][1]) + 1
        right = int(self.bounding_box[0][2]) - 1
        bottom = int(self.bounding_box[0][3]) - 1
        for s in p.segments_list:
            # if (s.point1[1] >= top and s.point2[1] >= top) or \
            #         (s.point1[1] <= bottom and s.point2[1] <= bottom):
            #     result = False
            if s.point1[1] == s.point2[1]:
                if s.point1[1] <= top or s.point1[1] >= bottom:
                    return True
            elif s.point1[0] == s.point2[0]:
                if s.point1[0] <= left or s.point1[0] >= right:
                    return True
        # return result
        return False

    def select_parallel_segment(self) -> ParallelSegment:

        most_horizontal: float = 99.0
        p_segment: ParallelSegment = None
        p_segment_touching_bbox: ParallelSegment = None
        for p in self.parallel_segments:
            try:
                slope = p.segments_list[0].slope
                assert len(p.segments_list) >= 2, "Parallel segment contains too low segment lines..."
            except IndexError:
                print(
                    f"Segment list contains too low or no segments!. Parallel Segment containing faulty segment: 'n{p}")
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
            print(f"Parallel segment contains more than two segments!")
            for s in p_segment.segments_list:
                print(f"{s.__dict__}")
            # which segments have common point?
            for i in range(len(p_segment.segments_list)):
                s1 = p_segment.segments_list[i]
                index_next = (i + 1) % (len(p_segment.segments_list))
                s2 = p_segment.segments_list[index_next]
                if s1.point2 == s2.point1:
                    print(f"Merging segments: \n{s1.__dict__}\n{s2.__dict__}")
                    p_segment.segments_list[index_next].point1 = p_segment.segments_list[i].point1
                    p_segment.segments_list[index_next].calculate_properties()
                    p_segment.segments_list[i].length = 0.0
                elif s1.point1 == s2.point2:
                    print(f"Merging segments: \n{s1.__dict__}\n{s2.__dict__}")
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

        assert len(self.parallel_segments) != 0, "No parallel segments were formed!"

        height_px = 0.0
        p_segment = self.select_parallel_segment()
        len_segments = len(p_segment.segments_list)

        per_line_top_end = (0, 50000)
        seg_slope = 0.0
        for i in range(len_segments):
            s1 = p_segment.segments_list[i % len_segments]
            s2 = p_segment.segments_list[(i + 1) % len_segments]

            mid_point = (int((s1.point1[0] + s1.point2[0]) / 2), int((s1.point1[1] + s1.point2[1]) / 2))
            if mid_point[1] < per_line_top_end[1]:
                per_line_top_end = mid_point
                seg_slope = s1.slope

            if mark_to_image is not None:
                cv2.circle(mark_to_image, (mid_point[0], mid_point[1]), 4, (255, 0, 0))

            x1, y1 = s2.point1[0], s2.point1[1]
            x2, y2 = s2.point2[0], s2.point2[1]
            x0, y0 = mid_point[0], mid_point[1]

            upper_part = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
            lower_part = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            d = upper_part / lower_part

            print(f"Distance from point {mid_point} to segment line is:\n{d}")
            height_px += d

        height_px /= 2.0

        # draw calculated height on image - top point will be mid point of top line.
        # calculate lower end of height line
        if mark_to_image is not None:
            if math.fabs(seg_slope) < 0.01:
                # line is nearly horizontal
                per_line_bottom_end = (per_line_top_end[0], per_line_top_end[1] + int(height_px))
            else:
                slope_perpendicular_line = - 1 / seg_slope
                print(f"Segment slope: {seg_slope}   -    Perpendicular line slope: {slope_perpendicular_line}")
                # normalize slope vector
                slope_vector = (1, slope_perpendicular_line)
                magnitude = math.sqrt(slope_vector[0] * slope_vector[0] +
                                      slope_vector[1] * slope_vector[1])
                normalized_slope = (slope_vector[0] / magnitude, slope_vector[1] / magnitude)
                print(f"Normalized slope vector: {normalized_slope}")
                if normalized_slope[1] >= 0.0:
                    per_line_bottom_end = (int(height_px * normalized_slope[0] + per_line_top_end[0]),
                                           (int(height_px * normalized_slope[1] + per_line_top_end[1])))
                else:
                    per_line_bottom_end = (int(height_px * normalized_slope[0] * -1 + per_line_top_end[0]),
                                           (int(height_px * normalized_slope[1] * -1 + per_line_top_end[1])))

            cv2.line(mark_to_image, per_line_top_end, per_line_bottom_end, (0, 0, 255), 3)
            validation_height_px: float = math.sqrt((per_line_bottom_end[0] - per_line_top_end[0]) ** 2 +
                                                    (per_line_bottom_end[1] - per_line_top_end[1]) ** 2)
            print(f"Calculated height_px: {height_px}    -     Validation height_px: {validation_height_px}")

        print(f"Mean height of card in px: {height_px}")
        return height_px
