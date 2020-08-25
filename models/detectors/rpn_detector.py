import tensorflow as tf

from utils.boxes import clip_to_window
from utils.training import generate_targets, get_sample_indices


class RPNDetector(tf.keras.Model):
    def __init__(
        self, image_shape, feature_maps_shape, config, name="region_proposal_network_detector",
    ):
        """
        Instantiate a Region Proposal Network detector.

        Args:
            - image_shape: Shape of the input images.
            - feature_maps_shape: Output shape of the feature extractor.
            - config: Region proposal network configuration dictionary.
        """
        super(RPNDetector, self).__init__(name=name)
        self._image_shape = image_shape
        _, grid_height, grid_width, _ = feature_maps_shape
        self._anchors = self._generate_anchors(grid_shape=(grid_height, grid_width), **config["anchors"])

        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        regularizer = tf.keras.regularizers.l2(config["weight_decay"])
        self._intermediate_layer = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=config["window_size"],
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="rpn_intermediate_layer",
        )

        num_anchors_per_location = len(config["anchors"]["scales"]) * len(config["anchors"]["aspect_ratios"])
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

    def call(self, feature_maps, training=False):
        """
        Args:
            - feature_maps: Tensor of shape [batch_size, height, width, channels].
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

        pred_scores = self._cls_layer(features)
        pred_scores = self._cls_reshape(pred_scores)
        pred_scores = self._cls_activation(pred_scores)

        pred_boxes_encoded = self._reg_layer(features)
        pred_boxes_encoded = self._reg_reshape(pred_boxes_encoded)

        if training:
            anchors, pred_scores, pred_boxes_encoded = self._remove_invalid_anchors_and_predictions(
                pred_scores, pred_boxes_encoded
            )
        else:
            image_height, image_width, _ = self._image_shape
            anchors = clip_to_window(self._anchors, [0, 0, image_width, image_height])

        return {"regions": anchors, "pred_scores": pred_scores, "pred_boxes": pred_boxes_encoded}

    def get_training_samples(
        self,
        gt_labels,
        gt_boxes,
        regions,
        pred_scores,
        pred_boxes,
        foreground_iou_interval,
        background_iou_interval,
        num_samples,
        foreground_proportion,
    ):
        """
        Args:
            - gt_labels: A tensor of shape [batch_size, max_num_objects, num_classes + 1] representing the
                ground-truth class labels possibly passed with zeros.
            - gt_boxes: A tensor of shape [batch_size, max_num_objects, 4] representing the
                ground-truth bounding boxes possibly passed with zeros.
            - regions: Tensor of shape [num_anchors, 4], representing the reference anchor boxes.
            - pred_scores: Tensor of shape [batch_size, num_anchors, 2].
                Output of the classification head, representing classification
                scores for each anchor.
            - pred_boxes: Tensor of shape [batch_size, num_anchors, 1, 4].
                Output of the regression head, representing encoded predicted box
                coordinates for each anchor.
            - foreground_iou_interval: Regions that have an IoU overlap with a ground-truth
                bounding box in this interval are labeled as foreground. Note that if there are no regions
                within this interval, the region with the highest IoU with any ground truth box will be labeled
                as foreground to ensure that there is always at least one positive example per image.
            - background_iou_interval: Regions that have an IoU overlap with a ground-truth
                bounding box in this interval are labeled as background. Regions that are neither labeled as
                foreground, nor background are ignored.
            - num_samples: Number of examples (regions) to sample per image.
            - foreground_proportion: Maximum proportion of foreground vs background
                examples to sample per image.

        Returns:
            Dictionary containing the randomly sampled targets and predictions to be used for computing the loss:
                - target_labels: Tensor of shape [batch_size, num_samples, 2].
                - pred_scores: Tensor of shape [batch_size, num_samples, 2].
                - target_boxes: Tensor of shape [batch_size, num_samples, 1, 4].
                - pred_boxes: Tensor of shape [batch_size, num_samples, 1, 4].
        """
        gt_objectness_labels = tf.one_hot(indices=tf.cast(tf.reduce_sum(gt_labels, -1), dtype=tf.int32), depth=2)

        target_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: generate_targets(
                x[0], x[1], regions, self._image_shape, foreground_iou_interval, background_iou_interval
            ),
            elems=(gt_objectness_labels, gt_boxes),
            dtype=(tf.float32, tf.float32),
        )

        sample_indices = tf.map_fn(
            fn=lambda l: get_sample_indices(l, num_samples, foreground_proportion), elems=target_labels, dtype=tf.int64,
        )

        samples = {}
        samples["target_labels"] = tf.gather(target_labels, sample_indices, batch_dims=1)
        samples["pred_scores"] = tf.gather(pred_scores, sample_indices, batch_dims=1)
        samples["target_boxes"] = tf.gather(target_boxes_encoded, sample_indices, batch_dims=1)
        samples["pred_boxes"] = tf.gather(pred_boxes, sample_indices, batch_dims=1)
        return samples

    def _generate_anchors(self, grid_shape, scales, aspect_ratios, base_anchor_shape, stride_shape=(16, 16)):
        """
        Creates the anchor boxes

        Args:
            - grid_shape: Shape of the anchors grid, i.e., feature maps grid shape.
            - scales: Anchors scales.
            - aspect_ratios: Anchors aspect ratios.
            - base_anchor_shape: Shape of the base anchor.
            - stride_shape: Shape of a pixel projected in the input image.

        Returns
            A Tensor of shape [num_anchors, 4] representing the box coordinates of the anchors.
            Nb: num_anchors = grid_shape[0] * grid_shape[1] * len(self.scales) * len(self.aspect_ratios)
        """

        scales, aspect_ratios = tf.meshgrid(scales, aspect_ratios)
        scales = tf.reshape(scales, [-1])
        aspect_ratios = tf.reshape(aspect_ratios, [-1])

        ratio_sqrts = tf.sqrt(aspect_ratios)
        heights = scales / ratio_sqrts * base_anchor_shape[0]
        widths = scales * ratio_sqrts * base_anchor_shape[1]

        x_centers = tf.range(grid_shape[1], dtype=tf.float32) * stride_shape[1]
        y_centers = tf.range(grid_shape[0], dtype=tf.float32) * stride_shape[0]
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        widths, x_centers = tf.meshgrid(widths, x_centers)
        heights, y_centers = tf.meshgrid(heights, y_centers)

        centers = tf.stack([x_centers, y_centers], axis=2)
        centers = tf.reshape(centers, [-1, 2])

        sizes = tf.stack([widths, heights], axis=2)
        sizes = tf.reshape(sizes, [-1, 2])

        return tf.concat([centers - 0.5 * sizes, centers + 0.5 * sizes], 1)

    def _remove_invalid_anchors_and_predictions(self, pred_scores, pred_boxes_encoded):
        """
        Remove anchors that overlap with the image boundaries, as well as
        the corresponding predictions.

        Args:
            - pred_scores: Tensor of shape [batch_size, num_anchors, 2].
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
        pred_scores = tf.gather(pred_scores, inds_to_keep, axis=1)
        pred_boxes_encoded = tf.gather(pred_boxes_encoded, inds_to_keep, axis=1)

        return anchors, pred_scores, pred_boxes_encoded
