class Config:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    # ##################
    # # backbone and fpn
    # ##################
    # backbone = "resnet101"  # or 'resnet50'
    #
    # # The strides of each layer of the FPN Pyramid.
    # # equal to subsample ratio of p2, p3, p4, p5, p6
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # # Size of the top-down layers used to build the feature pyramid
    # TOP_DOWN_PYRAMID_SIZE = 256
    #
    # # Number of classification classes (including background)
    # NUM_CLASSES = 1 + 1  # Override in sub-classes

    # ############################################
    # # RPN
    # ############################################
    # # Anchor
    # # Length of square anchor side in pixels
    # rpn_anchor_scales = (32, 64, 128, 256, 512)
    # # Ratios of anchors at each cell (width/height)
    # # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    # rpn_anchor_ratios = [0.5, 1, 2]
    #
    # # Target
    # rpn_target_neg_threshold = 0.3
    # rpn_target_pos_threshold = 0.7
    # # How many anchors per image to use for RPN training
    # # Number of targets of rpn
    # num_rpn_targets = 8    # 256 (pos + neg)
    # # Bounding box refinement standard deviation for RPN and final detections.
    # rpn_bbox_std_dev = (0.1, 0.1, 0.2, 0.2)

    # ############################################
    # # Proposal
    # ############################################
    # # Non-max suppression threshold to filter RPN proposals.
    # PROPOSAL_NMS_THRESHOLD = 0.7
    # # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # PRE_NMS_LIMIT = 6000  # used in proposal graph.
    # # ROIs kept after non-maximum suppression (training and inference)
    # POST_NMS_ROIS_TRAINING = 2000
    # POST_NMS_ROIS_INFERENCE = 1000

    ############################################
    # # Input
    # ############################################
    # # If enabled, resizes instance masks to a smaller size to reduce
    # # memory load. Recommended when using high-resolution images.
    # use_mini_mask = True
    # mini_mask_size = (56, 56)  # (height, width) of the mini-mask
    # # Input image resizing
    # # Generally, use the "square" resizing mode for training and predicting
    # # and it should work well in most cases. In this mode, images are scaled
    # # up such that the small side is = IMAGE_MIN_SIZE, but ensuring that the
    # # scaling doesn't make the long side > IMAGE_MAX_SIZE. Then the image is
    # # padded with zeros to make it a square so multiple images can be put
    # # in one batch.
    # # Available resizing modes:
    # # none:   No resizing or padding. Return the image unchanged.
    # # square: Resize and pad with zeros to get a square image
    # #         of size [max_dim, max_dim].
    # # pad64:  Pads width and height with zeros to make them multiples of 64.
    # #         If IMAGE_MIN_SIZE or IMAGE_MIN_SCALE are not None, then it scales
    # #         up before padding. IMAGE_MAX_SIZE is ignored in this mode.
    # #         The multiple of 64 is needed to ensure smooth scaling of feature
    # #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # # crop:   Picks random crops from the image. First, scales the image based
    # #         on IMAGE_MIN_SIZE and IMAGE_MIN_SCALE, then picks a random crop of
    # #         size IMAGE_MIN_SIZE x IMAGE_MIN_SIZE. Can be used in training only.
    # #         IMAGE_MAX_SIZE is not used in this mode.
    # image_resize_mode = "square"
    # IMAGE_MIN_SIZE = 100
    # IMAGE_MAX_SIZE = 128
    # # Image mean (RGB)
    # # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    # IMAGE_MEAN = (123.675, 116.28, 103.53)
    # IMAGE_STD = (58.395, 57.12, 57.375)
    # # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 5 # use to padding input gt

    # ############################################
    # # Detection
    # ############################################
    # # Number of ROIs per image to feed to classifier/mask heads
    # # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # # enough positive proposals to fill this and keep a positive:negative
    # # ratio of 1:3. You can increase the number of proposals by adjusting
    # # the RPN NMS threshold.
    # NUM_DETECTION_TARGETS = 4   # 200 # number of stage 2 roi targets
    # # Percent of positive ROIs used to train classifier/mask heads
    # ROI_POSITIVE_RATIO = 0.33

    ############################################
    # BOX HEAD AND MASK HEAD
    ############################################
    # # ROI align
    # CROP_SIZE = (7, 7) # box head
    # MASK_CROP_SIZE = (14, 14) # mask head
    #
    # # Shape of output mask
    # # To change this you also need to change the neural network mask branch
    # MASK_SHAPE = [28, 28]
    #
    # BBOX_STD_DEV = (0.1, 0.1, 0.2, 0.2)
    #
    # # detection
    # # Max number of final detections
    # DETECTION_MAX_INSTANCES = 50
    # # Minimum probability value to accept a detected instance
    # # ROIs below this threshold are skipped
    # DETECTION_MIN_CONFIDENCE = 0.7
    # # Non-maximum suppression threshold for detection
    # DETECTION_NMS_THRESHOLD = 0.3

    # ###########################################
    # # Train
    # ###########################################
    # # Learning rate and momentum
    # # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # # weights to explode. Likely due to differences in optimizer
    # # implementation.
    # LEARNING_RATE = 0.001
    # LEARNING_MOMENTUM = 0.9
    #
    # # Weight decay regularization
    # WEIGHT_DECAY = 0.0001
    #
    # # Loss weights for more precise optimization.
    # # Can be used for R-CNN training setup.
    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 1.,
    #     "rpn_bbox_loss": 1.,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.,
    #     "mrcnn_mask_loss": 1.
    # }
    # # Gradient norm clipping
    # GRADIENT_CLIP_NORM = 5.0

    def __init__(self, params=None):
        self.name = 'Config'

        ###########################################
        # Basic training config
        ###########################################
        self.num_gpus = 1
        self.batch_size = 1
        self.num_images_per_gpu = self.batch_size // self.num_gpus
        self.is_training_bn = False

        ###########################################
        # Train Hyper-parameters
        ###########################################
        # Number of classification classes (including background)
        self.num_classes = 1 + 1  # Override in sub-classes

        # Learning rate and momentum
        # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        # weights to explode. Likely due to differences in optimizer
        # implementation.
        self.learning_rate = 0.001
        self.learning_momentum = 0.9

        # weight decay regularization
        self.weight_decay = 0.0001

        # loss weights for more precise optimization.
        # can be used for r-cnn training setup.
        self.loss_weights = {
            "rpn_class_loss": 1.,
            "rpn_bbox_loss": 1.,
            "mrcnn_class_loss": 1.,
            "mrcnn_bbox_loss": 1.,
            "mrcnn_mask_loss": 1.
        }
        # gradient norm clipping
        self.gradient_clip_norm = 5.0

        ############################################
        # Input
        ############################################
        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        self.use_mini_mask = True
        self.mini_mask_size = (56, 56)  # (height, width) of the mini-mask
        self.image_resize_mode = "square"
        self.image_min_size = 200
        self.image_max_size = 256
        self.image_min_scale = 1.0
        self.image_mean = (123.675, 116.28, 103.53)  # RGB
        self.image_std = (58.395, 57.12, 57.375)  # RGB
        self.max_num_gt_instances = 30  # use to padding input gt

        ###########################################
        # backbone and fpn
        ###########################################
        self.backbone = "resnet101"  # or 'resnet50'

        # The strides of each layer of the FPN Pyramid.
        # equal to subsample ratio of p2, p3, p4, p5, p6
        self.backbone_strides = [4, 8, 16, 32, 64]
        # Size of the top-down layers used to build the feature pyramid
        self.top_down_pyramid_size = 256

        ############################################
        # RPN
        ############################################
        # Anchor
        self.rpn_anchor_scales = (32, 64, 128, 256, 512)  # h * w = scale^2
        self.rpn_anchor_ratios = (0.5, 1, 2)

        # Target
        self.rpn_target_negative_threshold = 0.3
        self.rpn_target_positive_threshold = 0.7
        self.num_rpn_targets = 256  # 256 (pos + neg)

        self.rpn_box_delta_std = (0.1, 0.1, 0.2, 0.2)
        self.rpn_box_delta_std = (0., 0., 0., 0.)

        ############################################
        # Proposal
        ############################################
        self.proposal_nms_threshold = 0.7
        self.pre_nms_limit = 6000  # choose top k before nms
        self.num_training_proposals = 2000
        self.num_inference_proposals = 1000

        ############################################
        # Detection
        ############################################
        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        # Target
        self.num_detection_targets = 200  # 200 # number of stage 2 roi targets
        self.detection_target_positive_ratio = 0.25  # not 0.33, to keep pos:neg ratio of 1:3

        # BOX HEAD AND MASK HEAD
        # ROI align
        self.sample_ratio = 2  # sample ratio of roi align
        self.box_head_crop_size = (7, 7)  # box head
        self.mask_head_crop_size = (14, 14)  # mask head
        self.mask_target_size = (28, 28)  # equal to mask head output size
        self.detection_box_delta_std = (0.1, 0.1, 0.2, 0.2)

        #############################################
        # Detection in Inference
        #############################################
        self.max_num_detection_instances = 30
        self.detection_min_confidence = 0.7
        self.detection_nms_threshold = 0.3  # Non-maximum suppression threshold for detection

        if params is not None:
            for key, val in params.items():
                self.__setattr__(key, val)

    def copy(self, update_params: dict):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in update_params.items():
            ret.__setattr__(key, val)

        return ret

    def update(self, update_params):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(update_params, Config):
            update_params = vars(update_params)

        for key, val in update_params.items():
            self.__setattr__(key, val)

    def __str__(self):
        message = '\n-------{} Info-------\n'.format(self.name)
        for k, v in vars(self).items():
            message += '{} = {}'.format(k, v) + '\n'
        return message


if __name__ == '__main__':
    config = Config()
    print(config)
