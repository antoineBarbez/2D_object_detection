import tensorflow as tf
import utils.boxes as box_utils
import utils.post_processing as postprocess_utils
import utils.sampling as sampling_utils

from models.detectors.abstract_detector import AbstractDetector
from utils.targets import TargetGenerator


class FastRCNNDetector(AbstractDetector):
    def __init__(self, image_shape, num_classes, name="fast_rcnn_detector"):
        """
        Instantiate a Fast-RCNN detector.

        Args:
            - image_shape: Shape of the input images.
            - num_classes: Number of classes without background.
        """
        super(FastRCNNDetector, self).__init__(name=name)

        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        regularizer = tf.keras.regularizers.l2(0.0005)

        self._roi_pooling = ROIPooling(pooled_size=5, kernel_size=2, name="regions_of_interest_pooling")

        self._cls_layer = tf.keras.layers.Dense(
            units=num_classes + 1,
            activation="softmax",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="fast_rcnn_classification_head",
        )

        self._reg_layer = tf.keras.layers.Dense(
            units=4 * num_classes,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="fast_rcnn_regression_head",
        )
        self._reg_reshape = tf.keras.layers.Reshape(
            target_shape=(-1, num_classes, 4), name="fast_rcnn_regression_head_reshape"
        )

        self._image_shape = image_shape

        self._target_generator = TargetGenerator(
            image_shape=image_shape,
            num_classes=num_classes,
            foreground_iou_interval=(0.5, 1.0),
            background_iou_interval=(0.0, 0.5),
        )

    def call(self, feature_maps, rois):
        """
        Args:
            - feature_maps: Output of the feature extractor. A tensor of shape 
                [batch_size, height, width, channels].
            - rois: A tensor of shape [batch_size, num_rois, 4] representing the 
                Regions of interest in relative coordinates.

        Returns:
            - rois: Regions to be used for postprocessing in absolute coordinates.
                Shape = [batch_size, num_rois, 4].
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_rois, num_classes + 1] representing classification scores 
                for each Region of Interest.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape 
                [batch_size, num_rois, num_classes, 4] representing encoded predicted box 
                coordinates for each Region of Interest.
        """
        pooled_features_flat = self._roi_pooling(feature_maps, rois, True, True)

        pred_class_scores = self._cls_layer(pooled_features_flat)

        pred_boxes_encoded = self._reg_layer(pooled_features_flat)
        pred_boxes_encoded = self._reg_reshape(pred_boxes_encoded)

        rois = box_utils.to_absolute(rois, self._image_shape)

        return rois, pred_class_scores, pred_boxes_encoded

    def get_training_samples(self, rois, gt_class_labels, gt_boxes, pred_class_scores, pred_boxes_encoded):
        target_class_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: self._target_generator.generate_targets(x[0], x[1], x[2]),
            elems=(gt_class_labels, gt_boxes, rois),
            dtype=(tf.float32, tf.float32),
        )

        return tf.map_fn(
            fn=lambda x: sampling_utils.sample_image(x[0], x[1], x[2], x[3], x[4], 64, 0.25),
            elems=(target_class_labels, target_boxes_encoded, pred_class_scores, pred_boxes_encoded, rois),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

    def postprocess_output(self, rois, pred_class_scores, pred_boxes_encoded, training):
        """
        Postprocess the output of the Faster-RCNN

        Args:
            - rois: A tensor of shape [num_anchors, 4] representing the Regions of Interest
                in absolute coordinates.
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_rois, num_classes + 1] representing classification scores 
                for each Region of Interest.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape 
                [batch_size, num_rois, num_classes, 4] representing encoded predicted box 
                coordinates for each Region of Interest.

        Returns:
            - boxes: A [batch_size, max_total_size, 4] float32 tensor containing the 
                predicted bounding boxes.
            - scores: A [batch_size, max_total_size] float32 tensor containing the 
                class scores for the boxes.
            - classes: A [batch_size, max_total_size] float32 tensor containing the 
                class indices for the boxes.

            With max_total_size = 300.
        """

        nmsed_boxes, nmsed_scores, nmsed_classes, _ = postprocess_utils.postprocess_output(
            regions=rois,
            image_shape=self._image_shape,
            pred_class_scores=pred_class_scores,
            pred_boxes_encoded=pred_boxes_encoded,
            max_output_size_per_class=100,
            max_total_size=300,
            iou_threshold=0.6,
        )

        return nmsed_boxes, nmsed_scores, nmsed_classes


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, pooled_size, kernel_size, **kwargs):
        super(ROIPooling, self).__init__(**kwargs)
        self._pooled_size = pooled_size
        self._kernel_size = kernel_size

        self._max_pool = tf.keras.layers.MaxPool2D(kernel_size, name="max_pool_2d")
        self._flatten = tf.keras.layers.Flatten(name="flatten")

    def call(self, feature_maps, rois, flatten=False, keep_batch_dim=False):
        """
        Args:
            - feature_maps: [batch_size, height, width, channels]
            - rois: [batch_size, num_rois, 4] normalized possibly padded
            - flatten: Boolean value indicating whether to flatten the roi pooled
                features, i.e., the last three dimensions of the output tensor.
            - keep_batch_dim: Boolean value indicating whether to keep the batch
                dimension and the rois dimension separate. If true, the output 
                tensor will be of shape [batch_size, num_rois, ...] else, it will
                be of shape [batch_size * num_rois, ...].
        """
        batch_size = tf.shape(rois)[0]
        num_rois = tf.shape(rois)[1]

        rois = tf.reshape(rois, [-1, 4])
        rois = tf.gather(rois, [1, 0, 3, 2], axis=-1)

        cropped_and_resized_features = tf.image.crop_and_resize(
            image=feature_maps,
            boxes=rois,
            box_indices=tf.repeat(tf.range(batch_size), num_rois),
            crop_size=[self._pooled_size * self._kernel_size, self._pooled_size * self._kernel_size],
            name="crop_and_resize",
        )

        pooled_features = self._max_pool(cropped_and_resized_features)

        if flatten:
            pooled_features = self._flatten(pooled_features)

        if keep_batch_dim:
            size_splits = tf.tile(tf.expand_dims(num_rois, 0), [batch_size])
            pooled_features = tf.stack(tf.split(pooled_features, size_splits))

        return pooled_features
