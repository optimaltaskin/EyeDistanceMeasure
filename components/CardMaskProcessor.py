import math
import random
import numpy as np
import cv2


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


class ParallelSegments:

    def __init__(self):
        self.segments_list: [] = []

    def add_segment(self, segment: Segment):
        self.segments_list.append(segment)

    def len_segments(self) -> int:
        return len(self.segments_list)


class CardMaskProcessor:

    def __init__(self, inference: (), image_filename: str, threshold: int = 100, min_segment_length: float = 10.0):

        self.inference: () = inference
        self.image_filename: str = image_filename
        self.image = None
        self.grayscale_mask: [] = None
        self.contours: [] = None
        self.hierarchy = None
        self.threshold: int = threshold
        self.hull_list: [] = None
        self.segments: [] = []
        self.parallel_segments: [] = []
        self.min_segment_length = min_segment_length

    def binary_to_grayscale(self):
        mask = self.inference[1][0][0]
        mask_np = (np.array(mask) * 255.0).astype('uint8')
        mask_stacked = np.stack((mask_np,) * 3, -1)
        self.grayscale_mask = cv2.cvtColor(mask_stacked, cv2.COLOR_BGR2GRAY)

    def define_contours(self):
        ret, thresh_img = cv2.threshold(self.grayscale_mask, self.threshold, 255, cv2.THRESH_BINARY)
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

    def generate_segments(self):
        for h in self.hull_list[0]:
            print (f"{h[0][0]}   {h[0][1]}")
        for i, h in enumerate(self.hull_list[0]):
            print(f"Segment {i} of {len(self.hull_list[0]) - 1}")
            p1 = h[0]
            i2 = (i + 1) % (len(self.hull_list[0]))
            p2 = self.hull_list[0][i2][0]
            s = Segment([p1[0], p1[1]], [p2[0], p2[1]])
            self.segments.append(s)
            print(f"{i}: {p1}   {i2}: {p2}")

    # def refine_by_similarity(self, display_refinement: bool):
    #     if display_refinement:
    #         image = self.image.copy()
    #         image2 = self.image.copy()
    #         for t, i in enumerate(self.segments):
    #             color = (255, 0, 0)
    #             cv2.circle(image, (i.point1[0], i.point1[1]), 2, color, 2)
    #         i = self.segments[-1]
    #         cv2.circle(image, (i.point2[0], i.point2[1]), 2, color, 2)
    #
    #     min_slope_dif = 0.05
    #     delete_items: [] = []
    #
    #     for i in range(len(self.segments) - 1):
    #         s1 = self.segments[i]
    #         index_next = (i + 1) % (len(self.segments))
    #         s2 = self.segments[index_next]
    #         slope_dif = abs(s1.slope - s2.slope)
    #         if slope_dif < min_slope_dif:
    #
    #             if i in delete_items:
    #                 # this segment will be deleted. So merge next segment with previous one
    #                 origin = i - 1
    #             else:
    #                 # merge two segments, since they are too similar
    #                 origin = i
    #             self.segments[origin].point2 = self.segments[index_next].point2
    #             self.segments[origin].calculate_properties()
    #             delete_items.append(index_next)
    #
    #             if display_refinement:
    #                 cv2.putText(image, f"{index_next}", (s2.point1[0], s2.point1[1]),
    #                             cv2.FONT_HERSHEY_PLAIN,
    #                             0.5, (0, 255, 0), 1)
    #
    #     # delete segments pointed by indexes in delete_items
    #     delete_items.sort(reverse=True)
    #
    #     for i in delete_items:
    #         del self.segments[i]
    #
    #     if display_refinement:
    #         for t, i in enumerate(self.segments):
    #             color = (255, 0, 0)
    #             cv2.circle(image2, (i.point1[0], i.point1[1]), 2, color, 2)
    #             cv2.putText(image2, f"{t}", (i.point1[0], i.point1[1]), cv2.FONT_HERSHEY_PLAIN,
    #                         0.5, (0, 255, 0), 1)
    #         i = self.segments[-1]
    #         cv2.circle(image2, (i.point2[0], i.point2[1]), 2, color, 2)
    #         cv2.putText(image2, f"{len(self.segments)}", (i.point2[0], i.point2[1]), cv2.FONT_HERSHEY_PLAIN,
    #                     0.5, (0, 255, 0), 1)
    #
    #         imshow_revised(image, 'Segment Points')
    #         imshow_revised(image2, 'Segment Points - Refine by similarity')

    def get_first_non_zero_length_index(self) -> int:
        for i, s in enumerate(self.segments):
            if s.length != 0.0:
                return i

    def refine_by_similarity(self, display_refinement):
        if display_refinement:
            image = self.image.copy()
            image2 = self.image.copy()
            for t, i in enumerate(self.segments):
                color = (255, 0, 0)
                cv2.circle(image, (i.point1[0], i.point1[1]), 2, color, 2)
            i = self.segments[-1]
            cv2.circle(image, (i.point2[0], i.point2[1]), 2, color, 2)

        min_slope_dif = 0.1

        for i in range(len(self.segments) - 1):
            s1 = self.segments[i]
            index_next = (i + 1) % (len(self.segments))
            s2 = self.segments[index_next]
            slope_dif = abs(s1.slope - s2.slope)
            print(f"i1: {i}   and     i2: {index_next}")
            print(f"Slopes diff: {slope_dif}     Min slope dif: {min_slope_dif}")
            if slope_dif < min_slope_dif:

                if i == len(self.segments):
                    # this boundary condition has a special case
                    next_index = self.get_first_non_zero_length_index()
                    print(f"Index next changed to: {index_next}")
                print(f"segment no: {i} will be deleted.")
                print(f"Changing seg#{index_next}'s point1 from {self.segments[index_next].point1} to {self.segments[i].point1}")
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
            for t, i in enumerate(self.segments):
                color = (255, 0, 0)
                cv2.circle(image2, (i.point1[0], i.point1[1]), 2, color, 2)
                cv2.putText(image2, f"{t}", (i.point1[0], i.point1[1]), cv2.FONT_HERSHEY_PLAIN,
                            0.5, (0, 255, 0), 1)
            i = self.segments[-1]
            cv2.circle(image2, (i.point2[0], i.point2[1]), 2, color, 2)
            cv2.putText(image2, f"{len(self.segments)}", (i.point2[0], i.point2[1]), cv2.FONT_HERSHEY_PLAIN,
                        0.5, (0, 255, 0), 1)

            imshow_revised(image, 'Segment Points')
            imshow_revised(image2, 'Segment Points - Refine by similarity')

    def refine_by_length(self, display_refinement: bool):

        # filter segments by length
        min_length: float = 20.0
        delete_items: [] = []
        for i, s in enumerate(self.segments):
            if s.length < min_length:
                delete_items.append(i)

        delete_items.sort(reverse=True)
        for i in delete_items:
            del self.segments[i]

        # draw segments
        if display_refinement:
            image3 = self.image.copy()
            for s in self.segments:
                cv2.line(image3, s.point1, s.point2, (r(), r(), r()), 2)

            imshow_revised(image3, 'Segment Lines - Refinement by length')

    def find_card_top_and_bottom(self, display_refinement: bool = False):

        self.generate_segments()
        if display_refinement:
            self.image = cv2.imread(self.image_filename)

        self.refine_by_similarity(display_refinement)
        self.refine_by_length(display_refinement)

        for i, s in enumerate(self.segments):
            slope = s.slope
            remaining_segments = self.segments.copy()
            remaining_segments.pop(i)
            parallel_s = ParallelSegments()
            parallel_s.add_segment(s)
            print(f"Processing segment:\n{s.__dict__}")

            for rem_segment in remaining_segments:
                slope_dif = abs(s.slope - rem_segment.slope)
                print(f"Slope difference is {slope_dif}")
                if slope_dif < 0.15:
                    print(f"Parallel segment found.")
                    print(f"Other seg: {rem_segment.__dict__}")
                    parallel_s.add_segment(rem_segment)
            if parallel_s.len_segments() > 1:
                print("Adding parallel segments.")
                self.parallel_segments.append(parallel_s)

        # draw parallel segments
        image_parallel = self.image.copy()
        for p in self.parallel_segments:
            color = (r(), r(), r())
            print(f"Parallel segment:\n{p.__dict__}")

            for s in p.segments_list:
                cv2.line(image_parallel, s.point1, s.point2, color, 2)

        imshow_revised(image_parallel, "Parallel segments marked")

        if display_refinement:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # for each segment check if there is any parallel segment.
        # If there are two or more sets of parallel segments, than choose the set which is closest to horizontal axis
        # if there are more than two segments with same slope, then filter it to two
        # to calculate the distance, project a line from mid part of each segment to other. You will have two measurements
        # take mean of those two measurements.

