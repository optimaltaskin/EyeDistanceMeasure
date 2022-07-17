class AugmentAnnotation:

    def __init__(self, images_list: []):
        self.last_image_id = images_list[-1]['id']
        self.image = {}
        self.annotation = {}

    def load_new_image_and_annot(self, image: {}, annotation: {}):
        self.image = image
        self.annotation = annotation

    def get_image_and_annot_no_scaling(self, file_name: str) -> ({}, {}):

        new_annot = self.annotation.copy()
        new_annot['id'] = self.last_image_id
        self.last_image_id += 1
        new_annot['image_id'] = self.last_image_id

        new_image_dict = self.image.copy()
        new_image_dict['id'] = self.last_image_id
        new_image_dict['file_name'] = file_name

        return new_image_dict, new_annot

    def get_image_and_annot_scaled(self, file_name: str, scale: float) -> ({}, {}):

        new_annot = self.annotation.copy()
        for index, point in enumerate(self.annotation['segmentation'][0]):
            new_annot['segmentation'][0][index] = point * scale

        for index, point in enumerate(self.annotation['bbox']):
            new_annot['bbox'][index] = point * scale

        new_annot['area'] *= scale * scale

        new_annot['id'] = self.last_image_id
        self.last_image_id += 1
        new_annot['image_id'] = self.last_image_id

        new_image_dict = self.image.copy()
        new_image_dict['id'] = self.last_image_id
        new_image_dict['file_name'] = file_name
        new_image_dict['width'] = int(scale * new_image_dict['width'])
        new_image_dict['height'] = int(scale * new_image_dict['height'])

        return new_image_dict, new_annot
