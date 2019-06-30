import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from utils import tensor_utils
from core import roi_align
import logging


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the datasets into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def build_rpn_graph(feature_maps, anchors_per_location):
    """

    Args:
        feature_maps: N x [B, H, W, C]
        anchors_per_location:

    Returns:
        rpn_logits, rpn_probs, rpn_deltas
    """
    rpn_outputs = []
    for i, feature_map in enumerate(feature_maps):
        s = 'pyramid{}'.format(i)
        # height, width = shape[1], shape[2]
        # here use relu
        x = slim.conv2d(feature_map, 256, [3, 3], stride=1, activation_fn=tf.nn.relu,
                        scope=s + '/rpn'.format(i))
        raw_deltas = slim.conv2d(x, anchors_per_location * 4, [1, 1], stride=1,
                                 scope=s + '/rpn/raw_deltas', padding='VALID',
                                 weights_initializer=slim.variance_scaling_initializer(),
                                 activation_fn=None)
        raw_class = slim.conv2d(x, anchors_per_location * 2, [1, 1], stride=1,
                                scope=s + '/rpn/class',
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=None)
        logits = tf.reshape(raw_class, [tf.shape(raw_class)[0], -1, 2],
                            name=s + '/rpn/logits')
        probs = slim.softmax(logits, scope=s + '/rpn/probs')
        deltas = tf.reshape(raw_deltas, [tf.shape(raw_deltas)[0], -1, 4],
                            name=s + '/rpn/deltas')
        rpn_outputs.append([logits, probs, deltas])
    logging.info('rpn_outputs {}'.format(rpn_outputs))

    rpn_logits, rpn_probs, rpn_deltas = [
        tf.concat(values, axis=1) for values in zip(*rpn_outputs)
    ]
    return rpn_logits, rpn_probs, rpn_deltas


def build_proposal_graph(rpn_probs, rpn_deltas, anchors, num_proposals,
                         nms_threshold, config):
    """ only build in training phase.
    return:
        proposals: [batch, num porposals, 4]
    """
    # Point:   use unnormed anchors
    scores = rpn_probs[:, :, 1]
    deltas = rpn_deltas

    def _build_proposals(_scores, _deltas):
        pre_nms_limit = tf.minimum(config.pre_nms_limit, tf.shape(anchors)[0])
        ids = tf.nn.top_k(_scores, pre_nms_limit, sorted=True,
                          name="top_anchors").indices
        _scores = tf.gather(_scores, ids)
        _deltas = tf.gather(_deltas, ids)
        _anchors = tf.gather(anchors, ids)

        _boxes = tensor_utils.decode_boxes(_anchors, _deltas)

        # normalize box
        _normed_boxes = tensor_utils.normalize_box(_boxes,
                                                   (config.image_max_size, config.image_max_size))
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)

        _normed_boxes = tensor_utils.clip_boxes(_normed_boxes, window)

        # Non-max suppression
        indices = tf.image.non_max_suppression(_normed_boxes, _scores, num_proposals,
                                               nms_threshold, name="rpn_non_max_suppression")
        _proposals = tf.gather(_normed_boxes, indices)
        # Pad if needed
        padding = tf.maximum(num_proposals - tf.shape(_proposals)[0], 0)
        _proposals = tf.pad(_proposals, [(0, padding), (0, 0)])
        return _proposals

    proposals = [
        _build_proposals(scores[i], deltas[i]) for i in range(config.batch_size)
    ]

    proposals = tf.stack(proposals, name='proposals')
    # print('proposals', proposals)
    return proposals


def build_detection_target_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """
    Note: do normalize to gt_boxes after gt_boxes trimmed.
    Args:
        proposals: [batch, num_proposals, 4] in normed.
        gt_class_ids: [batch, max_num_gt_instances]
        gt_boxes:
        gt_masks:
        config:

    Returns:
        rois, targets for detection
    """

    def _build_detection_target_graph(_proposals, _gt_class_ids, _gt_boxes, _gt_masks):
        asserts = [
            tf.Assert(tf.greater(tf.shape(_proposals)[0], 0), [_proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            _proposals = tf.identity(_proposals)

        # Remove zero padding
        _proposals, _ = tensor_utils.trim_zeros(_proposals, name="trim_proposals")
        _gt_boxes, non_zeros = tensor_utils.trim_zeros(_gt_boxes, name="trim_gt_boxes")
        _gt_class_ids = tf.boolean_mask(_gt_class_ids, non_zeros, name="trim_gt_class_ids")
        _gt_masks = tf.gather(_gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                              name="trim_gt_masks")

        # normalize gt_boxes since proposals have been normed.
        _normed_gt_boxes = tensor_utils.normalize_box(_gt_boxes,
                                                      (config.image_max_size, config.image_max_size))

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(_gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(_gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(_normed_gt_boxes, crowd_ix)
        _gt_class_ids = tf.gather(_gt_class_ids, non_crowd_ix)
        _normed_gt_boxes = tf.gather(_normed_gt_boxes, non_crowd_ix)
        _gt_masks = tf.gather(_gt_masks, non_crowd_ix, axis=2)
        # Compute overlaps matrix [proposals, _normed_gt_boxes]
        overlaps = tensor_utils.compute_overlaps(_proposals, _normed_gt_boxes)
        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = tensor_utils.compute_overlaps(_proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive 33% is 1:2, better than 1:3?
        # Positive ROIs
        positive_count = int(config.num_detection_targets * config.detection_target_positive_ratio)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / config.detection_target_positive_ratio  # left will be padding below
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(_proposals, positive_indices)
        negative_rois = tf.gather(_proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        # get gt_box_ids corresponding to positive_overlaps
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_ids = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[0], 0),  # tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        # get gt_box corresponding to positive_overlaps
        roi_gt_boxes = tf.gather(_normed_gt_boxes, roi_gt_box_ids)
        roi_gt_class_ids = tf.gather(_gt_class_ids, roi_gt_box_ids)

        # Compute bbox refinement for positive ROIs
        # TODO: put the std_dev to encode box
        deltas = tensor_utils.encode_boxes(positive_rois, roi_gt_boxes)

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(_gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_ids)

        # Compute mask targets
        boxes = positive_rois
        if config.use_mini_mask:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids,
                                         config.mask_target_size)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        num_negative = tf.shape(negative_rois)[0]
        padding = tf.maximum(config.num_detection_targets - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, padding), (0, 0)])
        # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, num_negative + padding), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, num_negative + padding)])
        deltas = tf.pad(deltas, [(0, num_negative + padding), (0, 0)])
        masks = tf.pad(masks, [[0, num_negative + padding], (0, 0), (0, 0)])

        roi_gt_class_ids = tf.stop_gradient(roi_gt_class_ids)
        deltas = tf.stop_gradient(deltas)
        masks = tf.stop_gradient(masks)
        return rois, roi_gt_class_ids, deltas, masks

    names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
    outputs = batch_slice(
        [proposals, gt_class_ids, gt_boxes, gt_masks],
        lambda p, c, b, m: _build_detection_target_graph(
            p, c, b, m),
        config.batch_size, names=names)
    return outputs


def build_rpn_target_graph(anchors, gt_boxes, config):
    """
    Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    Args
        anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
        gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image
            coordinates. batch_size = 1 usually
    Returns
        rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
            1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
        rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))]
            Anchor bbox deltas.
    """

    def _build_rpn_target(_gt_boxes):
        """Compute targets per instance.

            anchors: [num_anchors, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

            target_matchs: [num_anchors]
            target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))]
        """
        _gt_boxes, _ = tensor_utils.trim_zeros(_gt_boxes)

        match = tf.zeros(anchors.shape[0], dtype=tf.int32)  # [326393]

        # Compute overlaps [num_anchors, num_gt_boxes] 326393 vs 10 => [326393, 10]
        overlaps = tensor_utils.compute_overlaps(anchors, _gt_boxes)

        # 1. Negative match, may get overwritten bellow
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])
        # Mark -1 if max iou < rpn_target_negative_threshold
        match = tf.where(anchor_iou_max < config.rpn_target_negative_threshold,
                         -tf.ones(anchors.shape[0], dtype=tf.int32), match)

        # 2. Find closest anchor for each gt box. (despite of iou value)
        # if iou between this anchor and other box less than
        # this anchor and current gt box.
        # the matched gt box of this anchor is current gt box.
        # [num gt]
        gt_iou_argmax = tf.cast(tf.argmax(overlaps, axis=0), tf.int32)

        def scatter_update_tensor(x, indices, updates):
            """
            Utility function similar to `tf.scatter_update`, but performing on Tensor
            indices is nd. Avoid error occured when using tf.scatter_update.
            """
            x_shape = tf.shape(x)
            patch = tf.scatter_nd(indices, updates, x_shape)
            # mark 1 where need update
            mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), x_shape), 0)
            return tf.where(mask, patch, x)

        # update match to 1 of respective anchor
        match = scatter_update_tensor(match, tf.reshape(gt_iou_argmax, [-1, 1]),
                                      tf.ones(tf.shape(gt_iou_argmax)[0], dtype=tf.int32))
        # match = tf.scatter_update(tf.Variable(match), gt_iou_argmax, 1)

        # 3. Positive match.
        match = tf.where(anchor_iou_max >= config.rpn_target_positive_threshold,
                         tf.ones(anchors.shape[0], dtype=tf.int32), match)

        # 4. Subsample.
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(match, 1))
        ids = tf.cast(ids, tf.int32)
        extra = tf.shape(ids)[0] - (
                config.num_rpn_targets // 2)

        # https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond
        def _update_pos_match():
            indices = tf.random.shuffle(ids)[:extra]
            updates = tf.zeros([extra], dtype=tf.int32)
            shape = tf.shape(match)
            patch = tf.scatter_nd(indices, updates, shape)
            # mark 1 where need update
            mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), shape), 0)
            return tf.where(mask, patch, match)

        match = tf.cond(
            tf.greater(extra, 0),
            # use tf.Variable wrap it
            # avoid AttributeError: 'Tensor' object has no attribute '_lazy_read'
            _update_pos_match,
            lambda: tf.identity(match)
        )

        # Same for negative proposals
        ids = tf.where(tf.equal(match, -1))
        ids = tf.cast(ids, tf.int32)
        extra = tf.shape(ids)[0] - (
                config.num_rpn_targets -
                tf.reduce_sum(tf.cast(tf.equal(match, 1), tf.int32)))

        def _update_neg_match():
            indices = tf.random.shuffle(ids)[:extra]
            updates = tf.zeros([extra], dtype=tf.int32)
            shape = tf.shape(match)
            patch = tf.scatter_nd(indices, updates, shape)
            # mark 1 where need update
            mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), shape), 0)
            return tf.where(mask, patch, match)

        match = tf.cond(
            tf.greater(extra, 0),
            _update_neg_match,
            lambda: tf.identity(match)
        )

        # 6. deltas
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.where(tf.equal(match, 1))
        positive_anchors = tf.gather_nd(anchors, ids)
        # TODO: Update closest_gt_box_ids by gt_iou_argmax. closest_gt_box_ids[gt_iou_argmax] = indices of gt_iou_argmax
        #  the closest anchor of gt_box be overwritten here.
        #  this implement is follow Mask_RCNN repo, but there is something wrong.
        #  Not obey Principle 1 correct.
        closest_gt_box_ids = tf.gather_nd(anchor_iou_argmax, ids)
        closest_gt_box = tf.gather(_gt_boxes, closest_gt_box_ids)
        delta = tensor_utils.encode_boxes(
            positive_anchors, closest_gt_box)

        padding = tf.maximum(config.num_rpn_targets - tf.shape(delta)[0], 0)
        delta = tf.pad(delta, [(0, padding), (0, 0)])

        return match, delta

    rpn_target_matches, rpn_target_deltas = batch_slice(
        [gt_boxes],
        lambda x: _build_rpn_target(x), config.batch_size,
        names=['rpn_target_match', 'rpn_target_delta'])

    rpn_target_matches = tf.stop_gradient(rpn_target_matches)
    rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)

    return rpn_target_matches, rpn_target_deltas


def build_roi_align_graph(rois, feature_maps, image_meta, crop_size, config):
    """
    Implement roi align. (support different sample ratio.)
    Args:
        rois: tensor with shape [batch_size, num_rois, 4]
        feature_maps: list of feature_map, each one is a tensor with shape [batch_size, H, W, 256]
        image_meta: tensor with shape [batch_size, 12]
        crop_size: output size after roi align.
        config:

    Returns:
        list of pooled tensor, each one with shape [batch_size, crop_H, crop_W, 256]
    """

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
    h = y2 - y1
    w = x2 - x1
    # Use shape of first image. Images in a batch must have the same size.
    image_shape = tensor_utils.parse_image_meta(image_meta)['image_shape'][0]
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    roi_level = tensor_utils.log2(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
    roi_level = tf.minimum(5, tf.maximum(
        2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
    roi_level = tf.squeeze(roi_level, 2)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(rois, ix)

        # Box indices for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32)

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)

        pooled.append(roi_align(feature_maps[i], level_boxes,
                                box_indices=box_indices,
                                output_size=crop_size,
                                sample_ratio=config.sample_ratio))  # use 2

    # Pack pooled features into one tensor
    pooled = tf.concat(pooled, axis=0)
    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = tf.concat(box_to_level, axis=0)
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                             axis=1)

    # Rearrange pooled features to match the order of the original boxes
    # Sort box_to_level by batch then box index
    # TF doesn't have a way to sort by two columns, so merge them and sort.
    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    ix = tf.gather(box_to_level[:, 2], ix)
    pooled = tf.gather(pooled, ix)

    # Re-add the batch dimension
    shape = tf.concat([tf.shape(rois)[:2], tf.shape(pooled)[1:]], axis=0)
    pooled = tf.reshape(pooled, shape)
    return pooled


def build_box_head_graph(rois, stage2_feature_maps, image_meta, config,
                         is_training=False, weight_decay=1e-5,
                         reuse=tf.AUTO_REUSE):
    batch_norm_params = {
        'decay': 0.0001,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training and config.is_training_bn,
        'fused': None,  # Use fused batch norm if possible.
    }
    aligned = build_roi_align_graph(rois, stage2_feature_maps, image_meta,
                                    crop_size=config.box_head_crop_size,
                                    config=config)

    def _build_box_head(_aligned):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with tf.variable_scope('box_head', reuse=reuse):
                # print('per align: ', _aligned)
                x = slim.conv2d(_aligned, 1024, config.box_head_crop_size, stride=2,
                                padding='VALID')

                x = slim.conv2d(x, 1024, [1, 1], stride=2, padding='VALID')
                x = tf.squeeze(tf.squeeze(x, 2), 1)
                _logits = slim.fully_connected(
                    x, config.num_classes,
                    activation_fn=None,
                    weights_initializer=slim.variance_scaling_initializer(),
                    scope='logits')
                _raw_deltas = slim.fully_connected(
                    x, config.num_classes * 4,
                    activation_fn=None,
                    weights_initializer=slim.variance_scaling_initializer())
                _probs = slim.softmax(_logits, scope='probs')
                _deltas = tf.reshape(_raw_deltas, [tf.shape(_raw_deltas)[0], config.num_classes, 4],
                                     name='deltas')
                return _logits, _probs, _deltas

    outputs = [_build_box_head(aligned[i]) for i in range(config.batch_size)]
    outputs = list(zip(*outputs))
    output_names = ['predict_logits', 'predict_probs', 'predict_deltas']
    logits, probs, deltas = [tf.stack(x, axis=0, name=n) for x, n in zip(outputs, output_names)]

    return logits, probs, deltas


def build_mask_head_graph(rois, feature_maps, image_meta, config, is_training,
                          weight_decay=1e-5, reuse=tf.AUTO_REUSE):
    batch_norm_params = {
        'decay': 0.0001,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': (is_training and config.is_training_bn),
        'fused': None,  # Use fused batch norm if possible.
    }

    aligned = build_roi_align_graph(rois, feature_maps, image_meta, config.mask_head_crop_size,
                                    config)

    def _build_mask_head(_x):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with tf.variable_scope('mask_head', reuse=reuse):
                for _ in range(4):
                    _x = slim.conv2d(_x, 256, [3, 3])
                _x = slim.conv2d_transpose(_x, 256, 2, stride=2, activation_fn=tf.nn.relu)
                _x = slim.conv2d(_x, config.num_classes, [1, 1], padding='VALID',
                                 activation_fn=tf.nn.sigmoid)
            return _x

    outputs = [_build_mask_head(aligned[i]) for i in range(config.batch_size)]
    outputs = tf.stack(outputs, axis=0, name='predict_masks')
    return outputs


def build_detection_graph(proposals, probs, deltas, image_meta, config):
    # Get windows of images in normalized coordinates. Windows are the area
    # in the image that excludes the padding.
    # Use the shape of the first image in the batch to normalize the window
    # because we know that all images get resized to the same size.
    # print('image_meta', image_meta)
    m = tensor_utils.parse_image_meta(image_meta)
    image_size = m['image_shape'][0][:2]
    # print('image_shape', m['image_shape'])
    window = tensor_utils.normalize_box(m['window'], image_size)

    # Run detection refinement graph on each item in the batch
    detections_batch = batch_slice(
        [proposals, probs, deltas, window],
        lambda x, y, w, z: refine_detections_graph(x, y, w, z, config),
        config.batch_size)

    # Reshape output
    # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
    # normalized coordinates
    return tf.reshape(
        detections_batch,
        [config.batch_size, config.max_num_detection_instances, 6])


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.
    Args:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.
    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    rois.set_shape([config.num_inference_proposals, 4])
    probs.set_shape([config.num_inference_proposals, config.num_classes])
    logging.info('probs {} '.format(probs))
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    logging.info('class_ids {}'.format(class_ids))
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = tensor_utils.decode_boxes(
        rois, deltas_specific)
    # Clip boxes to image window
    refined_rois = tensor_utils.clip_boxes(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.detection_min_confidence:
        conf_keep = tf.where(class_scores >= config.detection_min_confidence)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.max_num_detection_instances,
            iou_threshold=config.detection_nms_threshold)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.max_num_detection_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.max_num_detection_instances])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.max_num_detection_instances
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < max_num_detection_instances
    gap = config.max_num_detection_instances - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)])
    return detections
