import logging
import tensorflow as tf
import argparse
import logging
import sys
import time
import json
import math
import os
from mask_rcnn import MaskRCNN
from configs import Config
from functools import partial


def _parse_example_proto(example_serialized, config):
    keys_to_features = {
        'image': tf.FixedLenFeature(shape=[config.image_max_size, config.image_max_size, 3],
                                    dtype=tf.float32),
        'image_meta': tf.FixedLenFeature(shape=[12], dtype=tf.float32),
        'padded_label': tf.FixedLenFeature(shape=[config.max_num_gt_instances], dtype=tf.float32),
        'padded_box': tf.FixedLenFeature(shape=[config.max_num_gt_instances, 4], dtype=tf.float32),
        'padded_mask': tf.FixedLenFeature(shape=[56, 56, config.max_num_gt_instances],
                                          dtype=tf.float32),
    }

    features = tf.parse_single_example(serialized=example_serialized,
                                       features=keys_to_features)
    input_image = tf.cast(features['image'], dtype=tf.float32)
    input_image_meta = tf.cast(features['image_meta'], dtype=tf.float32)
    input_label = tf.cast(features['padded_label'], dtype=tf.int32)
    input_gt_box = tf.cast(features['padded_box'], dtype=tf.float32)
    input_gt_mask = tf.cast(features['padded_mask'], dtype=tf.bool)

    return {
        'image': input_image,
        'image_meta': input_image_meta,
        'label': input_label,
        'box': input_gt_box,
        'mask': input_gt_mask
    }


def _get_training_variables(scopes=None):
    if scopes is None:
        return tf.trainable_variables()

    training_variables = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        training_variables.extend(variables)
    return training_variables


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='./_checkpoints/balloon')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--file_pattern', default='.../tfrecord/balloon_train*')
    parser.add_argument('--dataset_info', default='datasets/balloon_info.json')
    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stdout,
                        format='%(levelname)s %(filename)s: %(lineno)s: %(message)s')
    config = Config()
    is_training = True
    checkpoint_dir = args.checkpoint_dir
    num_epochs = args.num_epochs

    with open(args.dataset_info, 'r', encoding='utf-8') as fr:
        dataset_info = json.load(fr)
        num_training_examples = dataset_info['num_training_examples']

    dataset = tf.data.Dataset.list_files(args.file_pattern, shuffle=True)
    dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                 cycle_length=4)
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.map(partial(_parse_example_proto, config=config),
                          num_parallel_calls=4)
    dataset = dataset.batch(config.batch_size)
    iterator = dataset.make_initializable_iterator()

    inputs = iterator.get_next()

    model = MaskRCNN(config=config)
    stage1_outputs = model.build_stage1_graph(
        inputs['image'],
        inputs['image_meta'],
        is_training=is_training
    )
    # Note: better not use batch normalization
    training_outputs = model.build_training_stage2_graph(
        stage1_outputs.rpn_proposals,
        stage1_outputs.stage2_feature_maps,
        image_meta=inputs['image_meta'],
        anchors=stage1_outputs.anchors,
        gt_class_ids=inputs['label'],
        gt_boxes=inputs['box'],
        gt_masks=inputs['mask'],
        is_training=is_training,
        reuse=tf.AUTO_REUSE
    )

    loss_outputs = model.compute_loss(
        rpn_logits=stage1_outputs.rpn_logits,
        rpn_deltas=stage1_outputs.rpn_deltas,
        rpn_target_matches=training_outputs.rpn_target_matches,
        rpn_target_deltas=training_outputs.rpn_target_deltas,
        logits=training_outputs.logits,
        target_class_ids=training_outputs.target_class_ids,
        deltas=training_outputs.deltas,
        target_deltas=training_outputs.target_deltas,
        masks=training_outputs.masks,
        target_masks=training_outputs.target_masks
    )

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    global_step = tf.train.create_global_step()
    # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    regular_loss = tf.losses.get_regularization_loss()
    total_loss = loss_outputs.loss + regular_loss

    steps_per_epoch = int(math.ceil(num_training_examples / config.batch_size))

    first_decay_epochs = 1
    learning_rate = tf.train.cosine_decay_restarts(
        0.001, global_step,
        first_decay_steps=first_decay_epochs * steps_per_epoch,
        alpha=0.3
    )

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_variables = _get_training_variables()
    with tf.control_dependencies(update_ops):
        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=training_variables)
        grads_and_vars = [(tf.clip_by_norm(g, config.gradient_clip_norm), v)
                          for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step,
                                             name='train_op')

    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            start_time = time.time()
            sess.run(iterator.initializer)
            epoch_losses = []
            try:
                while True:

                    loss_value, _, lr_value = sess.run(
                        [loss_outputs.loss, train_op, learning_rate]
                    )
                    epoch_losses.append(loss_value)
            except tf.errors.OutOfRangeError:
                logging.info('Epoch: {}  take time: {:.3f}, lr: {:.6f}, loss: {}'.format(
                    epoch, time.time() - start_time, lr_value, sum(epoch_losses) / len(epoch_losses)))
                saver.save(sess, checkpoint_dir, global_step=global_step)


if __name__ == '__main__':
    main(parse_args())
