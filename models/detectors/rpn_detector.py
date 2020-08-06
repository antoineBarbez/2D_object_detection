import tensorflow as tf
import utils.anchors as anchor_utils
import utils.boxes as box_utils
import utils.post_processing as postprocess_utils
import utils.sampling as sampling_utils

from models.detectors.abstract_detector import AbstractDetector
from utils.targets import TargetGenerator


class RPNDetector(AbstractDetector):
    def __init__(
        self,
        image_shape,
        grid_shape,
        window_size,
        scales,
        aspect_ratios,
        base_anchor_shape,
        name="region_proposal_network_detector",
    ):
        """
        Instantiate a Region Proposal Network.

        Args:
            - image_shape: Shape of the input images.
            - grid_shape: Tuple of integers. Shape of the anchors grid, 
                i.e., height and width of the feature maps.
            - window_size: Size of the sliding window.
            - scales: Anchors' scales.
            - aspect_ratios: Anchors' aspect ratios.
            - base_anchor_shape: Tuple of integers. Shape of the base anchor. 
        """
        super(RPNDetector, self).__init__(name=name)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        regularizer = tf.keras.regularizers.l2(0.0005)
        self._intermediate_layer = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=window_size,
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="rpn_intermediate_layer",
        )

        num_anchors_per_location = len(scales) * len(aspect_ratios)
        self._cls_layer = tf.keras.layers.Conv2D(
            filters=2 * num_anchors_per_location,
            kernel_size=1,
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="rpn_classification_head",
        )
        self._cls_reshape = tf.keras.layers.Reshape(target_shape=(-1, 2), name="rpn_classification_head_reshape")
        self._cls_activation = tf.keras.layers.Activation(
            activation="softmax", name="rpn_classification_head_activation"
        )

        self._reg_layer = tf.keras.layers.Conv2D(
            filters=4 * num_anchors_per_location,
            kernel_size=1,
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="rpn_regression_head",
        )
        self._reg_reshape = tf.keras.layers.Reshape(target_shape=(-1, 1, 4), name="rpn_regression_head_reshape")

        self._image_shape = image_shape
        self._anchors = anchor_utils.generate_anchors(
            scales=scales,
            aspect_ratios=aspect_ratios,
            grid_shape=grid_shape,
            stride_shape=(16, 16),
            base_anchor_shape=base_anchor_shape,
        )

        self._target_generator = TargetGenerator(
            image_shape=image_shape, foreground_iou_interval=(0.7, 1.0), background_iou_interval=(0.0, 0.3),
        )

    @property
    def anchors(self):
        return self._anchors

    def call(self, feature_maps, training=False):
        """
        Args:
            - feature_maps: Tensor of shape [batch_size, im_height, im_width, channels].
                Output of the feature extractor.
            - training: Boolean value indicating whether the training version of the
                computation graph should be constructed.

        Returns:
            Dictionary with keys:
                - regions: Tensor of shape [num_anchors, 4].
                    Anchor boxes to be used for postprocessing.
                - pred_scores: Tensor of shape [batch_size, num_anchors, 2].
                    Output of the classification head, representing classification
                    scores for each anchor.
                - pred_boxes: Tensor of shape [batch_size, num_anchors, 1, 4].
                    Output of the regression head, representing encoded predicted box
                    coordinates for each anchor.
        """
        features = self._intermediate_layer(feature_maps)

        pred_class_scores = self._cls_layer(features)
        pred_class_scores = self._cls_reshape(pred_class_scores)
        pred_class_scores = self._cls_activation(pred_class_scores)

        pred_boxes_encoded = self._reg_layer(features)
        pred_boxes_encoded = self._reg_reshape(pred_boxes_encoded)

        if training:
            anchors, pred_class_scores, pred_boxes_encoded = self._remove_invalid_anchors_and_predictions(
                pred_class_scores, pred_boxes_encoded
            )
        else:
            image_height, image_width, _ = self._image_shape
            anchors = box_utils.clip_to_window(self._anchors, [0, 0, image_width, image_height])

        output = {}
        output["regions"] = anchors
        output["pred_scores"] = pred_class_scores
        output["pred_boxes"] = pred_boxes_encoded
        return output

    def get_training_samples(self, gt_class_labels, gt_boxes, output):
        gt_objectness_labels = tf.one_hot(indices=tf.cast(tf.reduce_sum(gt_class_labels, -1), dtype=tf.int32), depth=2)

        target_objectness_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: self._target_generator.generate_targets(x[0], x[1], output["regions"]),
            elems=(gt_objectness_labels, gt_boxes),
            dtype=(tf.float32, tf.float32),
        )

        sample_indices = tf.map_fn(
            fn=lambda y: sampling_utils.sample_image(y, 256, 0.5), elems=target_objectness_labels, dtype=tf.int64,
        )

        samples = {}
        samples["target_labels"] = tf.gather(target_objectness_labels, sample_indices, batch_dims=1)
        samples["pred_scores"] = tf.gather(output["pred_scores"], sample_indices, batch_dims=1)
        samples["target_boxes"] = tf.gather(target_boxes_encoded, sample_indices, batch_dims=1)
        samples["pred_boxes"] = tf.gather(output["pred_boxes"], sample_indices, batch_dims=1)
        return samples

    def postprocess_output(self, output, training):
        """
        Postprocess the output of the RPN

        Args:
            - output: Output dictionary of the call function.
            - training: A boolean value indicating whether we are in training mode.

        Returns:
            - rois: Tensor of shape [batch_size, max_predictions, 4] possibly zero padded 
                representing region proposals.
            - roi_scores: Tensor of shape [batch_size, max_predictions] possibly zero padded 
                representing objectness scores for each proposal.
        """
        max_predictions = 500 if training else 300

        nmsed_boxes, nmsed_scores, _, _ = postprocess_utils.postprocess_output(
            regions=output["regions"],
            image_shape=self._image_shape,
            pred_class_scores=output["pred_scores"],
            pred_boxes_encoded=output["pred_boxes"],
            max_output_size_per_class=max_predictions,
            max_total_size=max_predictions,
            iou_threshold=0.7,
        )

        return nmsed_boxes, nmsed_scores

    def _remove_invalid_anchors_and_predictions(self, pred_class_scores, pred_boxes_encoded):
        """
        Remove anchors that overlap with the image boundaries, as well as
        the corresponding predictions.

        Args:
            - pred_class_scores: Tensor of shape [batch_size, num_anchors, 2].
                Output of the classification head.
            - pred_boxes_encoded: Tensor of shape [batch_size, num_anchors, 1, 4].
                Output of the regression head.

        Returns:
            filtered anchors, pred_class_scores, and pred_boxes_encoded.
        """
        image_height, image_width, _ = self._image_shape
        inds_to_keep = tf.reshape(
            tf.where(
                (self._anchors[:, 0] >= 0)
                & (self._anchors[:, 1] >= 0)
                & (self._anchors[:, 2] <= image_width)
                & (self._anchors[:, 3] <= image_height)
            ),
            [-1],
        )

        anchors = tf.gather(self._anchors, inds_to_keep)
        pred_class_scores = tf.gather(pred_class_scores, inds_to_keep, axis=1)
        pred_boxes_encoded = tf.gather(pred_boxes_encoded, inds_to_keep, axis=1)

        return anchors, pred_class_scores, pred_boxes_encoded
