import random
import math
import tensorflow as tf
import os
import numpy as np
import sys

sys.path.append('.')
from multiprocessing import Pool
from datasets import balloon
from functools import partial
from configs import *


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _get_tfrecord_filename(dataset_name, tfrecord_dir, split_name, shard_id, num_shards=4):
    output_filename = dataset_name + '_{}_{:05}-of-{:05}.tfrecord'.format(
        split_name, shard_id, num_shards
    )
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    return os.path.join(tfrecord_dir, output_filename)


def get_tfexample(example):
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image': float_list_feature(example['image'].flatten()),
            'image_meta': float_list_feature(example['image_meta'].flatten()),
            'padded_label': float_list_feature(
                example['padded_label'].flatten().astype(np.float32)),
            'padded_box': float_list_feature(example['padded_box'].flatten()),
            'padded_mask': float_list_feature(example['padded_mask'].flatten().astype(np.float32))
        }))


def run_per_shard(pair, num_shards=None, dataset=None, split_name=None,
                  tfrecord_dir=None):
    shard_id, data_ids = pair
    output_filename = _get_tfrecord_filename(dataset.name, shard_id=shard_id,
                                             num_shards=num_shards,
                                             tfrecord_dir=tfrecord_dir,
                                             split_name=split_name)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i, idx in enumerate(data_ids):
            example = dataset[idx]
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i + 1, len(data_ids), shard_id))
            sys.stdout.flush()
            tf_example = get_tfexample(example)
            tfrecord_writer.write(tf_example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()
    print('shard {} complete.'.format(shard_id))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='Config')
    parser.add_argument('--num_shards', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--split_name', default='train')
    parser.add_argument('--dataset_dir', default='.../balloon/train')
    parser.add_argument('--tfrecord_dir', default='.../balloon/tfrecord')
    args = parser.parse_args()
    return args


def main(args):
    config = eval(args.config_name)()
    balloon_dataset = balloon.BalloonDataset(args.dataset_dir,
                                             os.path.join(args.dataset_dir, 'via_region_data.json'),
                                             config=config,
                                             export_info=True)

    indices = balloon_dataset.ids
    random.shuffle(indices)
    size_per_shard = math.ceil(len(indices) / num_shards)
    ids_list = [indices[i * size_per_shard:(i + 1) * size_per_shard]
                for i in range(args.num_shards)]
    shard_ids = list(range(args.num_shards))
    pool = Pool(args.num_workers)
    for _ in pool.imap_unordered(  # imap_unordered may occur error
            partial(run_per_shard, dataset=balloon_dataset,
                    num_shards=args.num_shards, split_name=args.split_name,
                    tfrecord_dir=args.tfrecord_dir),
            zip(shard_ids, ids_list)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    main(parse_args())
