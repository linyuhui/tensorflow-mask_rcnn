import numpy as np
import tensorflow as tf
import math
import cv2
import logging
from distutils.version import LooseVersion
import losses

from utils import tensor_utils
from modules import build_proposal_graph
from modules import build_detection_target_graph, build_box_head_graph
from modules import build_mask_head_graph, build_detection_graph
from modules import build_rpn_graph, build_rpn_target_graph
from collections import namedtuple
from resnet import resnet_arg_scope, resnet_v1_101
from tensorflow.contrib import slim

Stage1Outputs = namedtuple(
    'stage1_outputs',
    ['rpn_proposals', 'rpn_logits', 'rpn_deltas', 'anchors', 'stage2_feature_maps']
)
TrainingOutputs = namedtuple(
    'training_outputs',
    ['logits',
     'target_class_ids', 'deltas',
     'target_deltas', 'masks', 'target_masks',
     'rpn_target_matches', 'rpn_target_deltas'])
LossOutputs = namedtuple(
    'loss_outputs',
    ['rpn_class_loss', 'rpn_box_loss', 'class_loss',
     'box_loss', 'mask_loss', 'loss'
     ]
)


class MaskRCNN:
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """

        self.config = config

    def build_stage1_graph(self,
                           image,
                           image_meta,
                           is_training):

        with slim.arg_scope(resnet_arg_scope()):
            with slim.arg_scope([slim.batch_norm],
                                is_training=(is_training and self.config.is_training_bn)):
                _, endpoints = resnet_v1_101(image, global_pool=False)
                resnet_name = 'resnet_v1_101'
                c2, c3, c4, c5 = (
                    endpoints[resnet_name + '/block1'],
                    endpoints[resnet_name + '/block2'],
                    endpoints[resnet_name + '/block3'],
                    endpoints[resnet_name + '/block4']
                )

        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=None):
            p5 = slim.conv2d(c5, self.config.top_down_pyramid_size, (1, 1), padding='VALID')
            p4 = (tf.image.resize_images(p5, [p5.shape.as_list()[1] * 2, p5.shape.as_list()[2] * 2])
                  + slim.conv2d(c4, self.config.top_down_pyramid_size, (1, 1),
                                padding='VALID'))
            p3 = (tf.image.resize_images(p4, [p4.shape.as_list()[1] * 2, p4.shape.as_list()[2] * 2])
                  + slim.conv2d(c3, self.config.top_down_pyramid_size, (1, 1), padding='VALID'))
            p2 = (tf.image.resize_images(p3, [p3.shape.as_list()[1] * 2, p3.shape.as_list()[2] * 2])
                  + slim.conv2d(c2, self.config.top_down_pyramid_size, (1, 1), padding='VALID'))

            p2 = slim.conv2d(p2, self.config.top_down_pyramid_size, (3, 3), padding='SAME')
            p3 = slim.conv2d(p3, self.config.top_down_pyramid_size, (3, 3), padding='SAME')
            p4 = slim.conv2d(p4, self.config.top_down_pyramid_size, (3, 3), padding='SAME')
            p5 = slim.conv2d(p5, self.config.top_down_pyramid_size, (3, 3), padding='SAME')

            p6 = slim.max_pool2d(p5, kernel_size=(1, 1), stride=2)

        stage1_feature_maps = [p2, p3, p4, p5, p6]
        stage2_feature_maps = [p2, p3, p4, p5]

        backbone_sizes = [(int(math.ceil(image.shape.as_list()[1] / stride)),
                           int(math.ceil(image.shape.as_list()[2] / stride)))
                          for stride in self.config.backbone_strides]
        anchors = tensor_utils.generate_pyramid_anchors(
            self.config.rpn_anchor_scales,
            self.config.rpn_anchor_ratios,
            backbone_sizes,
            self.config.backbone_strides)

        rpn_logits, rpn_probs, rpn_deltas = build_rpn_graph(
            stage1_feature_maps,
            anchors_per_location=len(self.config.rpn_anchor_ratios))

        if is_training:
            num_proposal = self.config.num_training_proposals
        else:
            num_proposal = self.config.num_inference_proposals

        # num_proposal = tf.cond(is_training,
        #                        lambda: self.config.num_training_proposals,
        #                        lambda: self.config.num_inference_proposals)
        # use normed anchors, normed boxes
        # or use unnormed anchors then apply norm to boxes
        # to avoid gen anchors again, use the unnormed anchors
        # return normed proposals
        with tf.variable_scope('proposal_graph'):
            rpn_proposals = build_proposal_graph(
                rpn_probs, rpn_deltas, anchors, num_proposal,
                nms_threshold=self.config.proposal_nms_threshold, config=self.config)

        return Stage1Outputs(
            rpn_proposals=rpn_proposals,
            rpn_logits=rpn_logits,
            rpn_deltas=rpn_deltas,
            stage2_feature_maps=stage2_feature_maps,
            anchors=anchors
        )

    def build_training_stage2_graph(self,
                                    rpn_proposals,
                                    stage2_feature_maps,
                                    image_meta,
                                    anchors,
                                    gt_class_ids,
                                    gt_boxes,
                                    gt_masks,
                                    is_training,
                                    reuse=tf.AUTO_REUSE):
        # unnormed anchors and unnormed gt_boxes
        rpn_target_matches, rpn_target_deltas = build_rpn_target_graph(
            anchors, gt_boxes, self.config)

        # normed rois
        rois, target_class_ids, target_deltas, target_masks = build_detection_target_graph(
            rpn_proposals, gt_class_ids, gt_boxes, gt_masks, self.config
        )
        # print('rois', rois)
        logits, probs, deltas = build_box_head_graph(
            rois, stage2_feature_maps, image_meta, self.config,
            is_training=is_training, reuse=reuse
        )
        masks = build_mask_head_graph(
            rois, stage2_feature_maps, image_meta, self.config,
            is_training=is_training, reuse=reuse
        )
        return TrainingOutputs(
            logits=logits,
            target_class_ids=target_class_ids,
            deltas=deltas,
            target_deltas=target_deltas,
            masks=masks,
            target_masks=target_masks,
            rpn_target_matches=rpn_target_matches,
            rpn_target_deltas=rpn_target_deltas
        )

    def build_inference_stage2_graph(self,
                                     rpn_proposals,
                                     stage2_feature_maps,
                                     image_meta,
                                     is_training=False,
                                     reuse=tf.AUTO_REUSE
                                     ):
        logits, probs, deltas = build_box_head_graph(
            rpn_proposals, stage2_feature_maps, image_meta, self.config,
            is_training=is_training, reuse=reuse
        )

        detection = build_detection_graph(
            rpn_proposals, probs, deltas, image_meta, self.config
        )
        boxes = detection[:, :, :4]
        mask = build_mask_head_graph(
            boxes, stage2_feature_maps, image_meta, self.config,
            is_training=is_training, reuse=reuse
        )

        return {
            'detection': detection,
            'mask': mask
        }

    def compute_loss(self, rpn_logits, rpn_deltas, rpn_target_matches, rpn_target_deltas,
                     logits, target_class_ids, deltas, target_deltas, masks, target_masks):
        rpn_class_loss = losses.rpn_class_loss(rpn_target_matches, rpn_logits)
        rpn_box_loss = losses.rpn_box_loss(self.config, rpn_target_deltas, rpn_target_matches,
                                           rpn_deltas)
        class_loss = losses.class_loss(target_class_ids, logits)
        box_loss = losses.box_loss(target_deltas, target_class_ids, deltas)
        mask_loss = losses.mask_loss(target_masks, target_class_ids, masks)
        loss = rpn_class_loss + rpn_box_loss + class_loss + box_loss + mask_loss

        return LossOutputs(
            rpn_class_loss=rpn_class_loss,
            rpn_box_loss=rpn_box_loss,
            class_loss=class_loss,
            box_loss=box_loss,
            mask_loss=mask_loss,
            loss=loss
        )

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta datasets]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = resize_image(
                image,
                min_dim=self.config.image_min_size,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.image_max_size,
                mode=self.config.image_resize_mode)

            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.num_classes], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = normalize_box(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference" or self.mode == "export", "Create model in inference mode."
        assert len(
            images) == self.config.batch_size, "len(images) must be equal to batch_size"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        # GMJ_test
        GMJ_test = False
        if GMJ_test:
            fs = cv2.FileStorage("/home/sf/GMJ/infer_image.xml", cv2.FILE_STORAGE_WRITE)
            fs.write('mat', np.reshape(molded_images[0], [512, 512, 3]))
            fs.release()
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check image_resize_mode and " \
                "image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.batch_size,) + anchors.shape)
        reshape_anchors = np.reshape(anchors, [1, -1])
        reshape_image_metas = np.reshape(image_metas, [1, -1])
        # GMJ_test

        if GMJ_test:
            fs = cv2.FileStorage("/home/sf/GMJ/infer_anchors.xml", cv2.FILE_STORAGE_WRITE)
            fs.write('mat', np.reshape(anchors, [65472, 4]))
            fs.release()
            fs = cv2.FileStorage("/home/sf/GMJ/infer_image_meta.xml", cv2.FILE_STORAGE_WRITE)
            fs.write('mat', np.reshape(image_metas, [1, 14]))
            fs.release()
            # np.savetxt('/home/sf/anchors.txt',reshape_anchors)
            # np.savetxt('/home/sf/image_metas.txt', reshape_image_metas)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _, mrcnn_class_logits = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,

            })
        return results, mrcnn_class_logits

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta datasets, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.batch_size, \
            "Number of images must be equal to batch_size"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.batch_size,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def postprocess(self, detection, mask, image_meta, window):
        """

        Args:
            detection: tensor with shape [1, max_num_detections, 6]
            mask: tensor with shape[1, max_num_detections, mask_size, mask_size, num_classes]
            image_meta: tensor with shape [1, 12]
            window: tuple

        Returns:
            full_mask: tensor with shape [original height, original width, num_detections]
            box: tensor with shape [num_detections, 4]
            class_id: tensor with shape [num_detections]
            score: tensor with shape [num_detections]
        """
        detection = tf.squeeze(detection, axis=0)
        mask = tf.squeeze(mask, axis=0)
        logging.info('{}{}{}'.format(detection.dtype, detection.shape, mask.shape))
        zero_ids = tf.cast(tf.where(tf.equal(detection[:, 4], 0))[:, 0], tf.int32)
        non_zero_count = tf.cond(tf.greater(tf.shape(zero_ids)[0], 0),
                                 lambda: zero_ids[0], lambda: tf.shape(detection)[0])
        box = detection[:non_zero_count, :4]

        class_ids = tf.cast(detection[:non_zero_count, 4], tf.int32, name='class_ids')
        logging.info('{}'.format(detection[:non_zero_count, 4].shape))
        scores = detection[:non_zero_count, 5]

        # put class_id to dim1 from last dim.
        mask = tf.transpose(mask, [0, 3, 1, 2])
        indices = tf.stack([tf.range(0, non_zero_count), class_ids], axis=1)
        mask = tf.gather_nd(mask, indices)  # reduce to 3 dims

        # detect box is normalized by padded image
        window = tensor_utils.normalize_box(tf.constant(window, tf.float32), image_meta[0, 4:6])
        logging.info('{}'.format(window.shape))
        wy1, wx1, wy2, wx2 = window[0], window[1], window[2], window[3]
        shift = tf.stack([wy1, wx1, wy1, wx1])
        window_height = wy2 - wy1
        window_width = wx2 - wx1
        scale = tf.stack([window_height, window_width, window_height, window_width])
        # normalized by window
        box = (box - shift) / scale
        box = tensor_utils.denormalize_box(box, image_meta[0, 1:3])

        original_image_size = image_meta[0, 1:3]

        def _postprocess_mask(_i, _m):

            _box = tf.cast(box[_i], tf.int32)
            print('_box', _box)
            # TypeError: Tensor objects are only iterable when eager execution is enabled.
            # y1, x1, y2, x2 = _box
            y1, x1, y2, x2 = _box[0], _box[1], _box[2], _box[3]
            _mask = mask[_i]
            _mask = tf.expand_dims(_mask, axis=2)
            _mask = tf.image.resize_images(_mask, (y2 - y1, x2 - x1))
            _mask = tf.squeeze(_mask, axis=2)

            # Put the mask in the right location.
            _full_mask = tf.zeros(original_image_size, dtype=np.float32)
            # TypeError: 'Tensor' object does not support item assignment
            # _full_mask[y1:y2, x1:x2] = _mask
            _x_indices, _y_indices = tf.meshgrid(tf.range(x1, x2), tf.range(y1, y2))
            _indices = tf.reshape(tf.stack([_y_indices, _x_indices], axis=2), shape=[-1, 2],
                                  name='mask_indices')
            # scatter_update_tensor updates dtype can't be bool
            _full_mask = tensor_utils.scatter_update_tensor(
                _full_mask, tf.Print(_indices, [_indices], 'indices', summarize=16), tf.reshape(_mask, [-1]))
            print('full mask', _full_mask)
            _full_mask = tf.expand_dims(_full_mask, axis=2)
            return _i + 1, tf.concat([_m, _full_mask], axis=2)

        m0 = tf.zeros(shape=[original_image_size[0], original_image_size[1], 0])
        i0 = tf.constant(0)
        _, full_mask = tf.while_loop(
            lambda i, m: tf.less(i, non_zero_count),
            _postprocess_mask, [i0, m0],
            shape_invariants=[
                i0.get_shape(),
                tf.TensorShape([original_image_size[0], original_image_size[1], None])]
        )
        threshold = 0.5

        full_mask = tf.cast(
            tf.where(tf.greater(full_mask, threshold),
                     tf.ones_like(full_mask), tf.zeros_like(full_mask)), tf.bool
        )

        return {
            'box': box,
            'class_id': class_ids,
            'mask': full_mask,
            'score': scores
        }
