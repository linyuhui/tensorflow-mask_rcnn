import tensorflow as tf
import cv2
import numpy as np
import logging
import sys
import colorsys
import matplotlib.pyplot as plt
import random
from mask_rcnn import MaskRCNN
from matplotlib import patches, lines
from configs import *
from data import compose_image_meta
from skimage.measure import find_contours


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display(image, boxes, masks, class_ids, class_names,
            scores=None, title="",
            figsize=(16, 16), fig_and_ax=None,
            show_mask=True, show_bbox=True,
            colors=None, captions=None,
            result_file=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [num_instances, height, width]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        #
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not fig_and_ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    else:
        fig, ax = fig_and_ax

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            class_name = class_names[class_id]
            caption = "{} {:.3f}".format(class_name, score) if score else class_name
        else:
            caption = captions[i]
        ax.text(x1, y1 + 10, caption,
                color=color, size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    # notice class and prob is add in ax not in mask image
    ax.imshow(masked_image.astype(np.uint8))
    # plt.show()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(result_file,
                bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))

    if auto_show:
        plt.show()


def prepare_input(image_file, config):
    original_image = cv2.imread(image_file)
    original_image = original_image[:, :, ::-1]  # to rgb
    height, width = original_image.shape[:2]
    min_size, max_size = 200, 256
    scale = max(1, min_size / min(height, width))
    scale = min(scale, max_size / max(height, width))
    # print(scale)
    resized_image = cv2.resize(original_image, (round(width * scale), round(height * scale)))
    # square mode
    h, w = resized_image.shape[:2]
    top_pad = (max_size - h) // 2
    bottom_pad = max_size - h - top_pad
    left_pad = (max_size - w) // 2
    right_pad = max_size - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(resized_image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    image = image.astype(np.float32)

    # norm
    image = (image - config.image_mean) / config.image_std

    image_meta = compose_image_meta(0, original_image.shape, image.shape, window, scale)
    image = image[None, ...]
    image_meta = image_meta[None, ...]
    return image, image_meta, window, original_image


def inference(image_path, checkpoint):
    input_image = tf.placeholder(tf.float32, [None, 256, 256, 3])
    input_image_meta = tf.placeholder(tf.float32, [None, 12])

    config_name = 'Config'
    config = eval(config_name)()

    image, image_meta, window, original_image = prepare_input(image_path, config)

    model = MaskRCNN(config=config)

    stage1_outputs = model.build_stage1_graph(
        input_image,
        input_image_meta,
        is_training=False
    )
    # Note: better not use batch normalization
    inference_outputs = model.build_inference_stage2_graph(
        stage1_outputs.rpn_proposals,
        stage1_outputs.stage2_feature_maps,
        input_image_meta,
        reuse=tf.AUTO_REUSE
    )

    result = model.postprocess(inference_outputs['detection'], inference_outputs['mask'],
                               image_meta, window)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, checkpoint)
        box, class_id, mask, score, _ = sess.run(
            [result['box'], result['class_id'], result['mask'], result['score'], print_op],
            feed_dict={
                input_image: image,
                input_image_meta: image_meta
            }
        )

        display(original_image, box, mask, class_id, scores=score,
                class_names={1: 'balloon'}, result_file='a.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='24631331976_defa3bb61f_k.jpg')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stdout,
                        format='%(levelname)s %(filename)s: %(lineno)s: %(message)s')
    inference(args.image_path, args.checkpoint)


if __name__ == '__main__':
    main(parse_args())
