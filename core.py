import tensorflow as tf


def roi_align(feature_maps, boxes, box_indices, output_size, sample_ratio):
    """Implement ROI align with different sample_ratio support.
    Args:
        feature_maps: tensor, shape[batch_size, height, width, channels]
        boxes: shape [num_boxes, (y1, x1, y2, x2)] y and x normalized by (height -1) or
            (width -1)
        box_indices: indicate which feature_map to crop.
        output_size: roi_align output size.
        sample_ratio: sample ratio in each row in each bin. If sample 4 regular locations,
            sample_ratio is 2.
    """

    def _roi_align_core():
        """
        Return:
            cropped image sampled by bilinear interpolation with shape [num_boxes, crop_height,
            crop_width, depth], totally samples (output_size * sample_ratio)**2 points.
        """
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        bin_height = (y2 - y1) / output_size[0]
        bin_width = (x2 - x1) / output_size[1]

        grid_center_y1 = (y1 + 0.5 * bin_height / sample_ratio)
        grid_center_x1 = (x1 + 0.5 * bin_width / sample_ratio)

        grid_center_y2 = (y2 - 0.5 * bin_height / sample_ratio)
        grid_center_x2 = (x2 - 0.5 * bin_width / sample_ratio)

        new_boxes = tf.concat([grid_center_y1, grid_center_x1, grid_center_y2, grid_center_x2],
                              axis=1)

        crop_size = tf.constant([output_size[0] * sample_ratio, output_size[1] * sample_ratio])
        return tf.image.crop_and_resize(feature_maps, new_boxes, box_ind=box_indices,
                                        crop_size=crop_size, method='bilinear')

    sampled = _roi_align_core()

    aligned = tf.layers.average_pooling2d(sampled, sample_ratio, sample_ratio)
    return aligned
