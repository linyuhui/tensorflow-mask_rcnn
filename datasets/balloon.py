import numpy as np
import os
import skimage
import json
import cv2
from data import load_example


class BalloonDataset:
    def __init__(self, dataset_dir, annotation_file=None, config=None,
                 name='balloon',
                 export_info=True):
        self.name = name
        self.class_name_to_id = {
            'ballon': 1
        }
        self.class_id_to_name = {
            v: k for k, v in self.class_name_to_id.items()
        }
        self.image_paths = []
        self.polygons = []
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]
            self.image_paths.append(image_path)
            self.polygons.append(polygons)

        self.ids = list(range(len(self.image_paths)))
        self.config = config

        if export_info:
            info = {
                'num_training_examples': len(self.ids)
            }
            with open('./datasets/balloon_info.json', 'w', encoding='utf-8') as fw:
                json.dump(info, fw, indent=4)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        polygons = self.polygons[idx]
        image = skimage.io.imread(image_path)  # rgb
        height, width = image.shape[:2]

        if image.ndim < 3:
            image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rows, cols = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rows, cols, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return image, mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        mask = mask.astype(np.bool)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)

        example = load_example(idx, image, mask, class_ids, self.config,
                               use_mini_mask=self.config.use_mini_mask)
        padded_label = np.zeros(self.config.max_num_gt_instances,
                                dtype=example.label.dtype)
        padded_box = np.zeros((self.config.max_num_gt_instances, 4),
                              dtype=example.box.dtype)

        mask_size = example.mask.shape[0:2]
        padded_mask = np.zeros(
            (mask_size[0], mask_size[1], self.config.max_num_gt_instances),
            dtype=example.mask.dtype)

        # fill data
        padded_box[:example.box.shape[0], :] = example.box
        padded_mask[:, :, :example.mask.shape[2]] = example.mask
        padded_label[:example.label.shape[0]] = example.label

        # tf.float32, tf.float32, tf.int32, tf.float32, tf.bool
        return {
            'image': example.image,
            'image_meta': example.image_meta,
            'padded_label': padded_label,
            'padded_box': padded_box,
            'padded_mask': padded_mask
        }

    def __len__(self):
        return len(self.image_paths)

