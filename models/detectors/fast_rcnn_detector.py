import tensorflow as tf

from utils.boxes import to_absolute
from utils.training import generate_targets, get_sample_indices


class FastRCNNDetector(tf.keras.Model):
    def __init__(self, image_shape, num_classes, config, name="fast_rcnn_detector"):
        """
        Instantiate a Fast-RCNN detector.

        Args:
            - image_shape: Shape of the input images.
            - num_classes: Number of classes without background.
            - config: Fast RCNN configuration dictionary.
        """
        super(FastRCNNDetector, self).__init__(name=name)
        self._image_shape = image_shape

        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        regularizer = tf.keras.regularizers.l2(config["weight_decay"])

        self._roi_pooling = ROIPooling(name="regions_of_interest_pooling", **config["roi_pooling"])

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

    def call(self, feature_maps, rois):
        """
        Args:
            - feature_maps: Tensor of shape [batch_size, height, width, channels].
                Output of the feature extractor.
            - rois: Tensor of shape [batch_size, num_rois, 4] representing the
                Regions of interest in relative coordinates.

        Returns:
            Dictionary with keys:
                - regions: Tensor of shape [batch_size, num_rois, 4]. Regions to be used for
                    postprocessing in absolute coordinates.
                - pred_scores: Tensor of shape [batch_size, num_rois, num_classes + 1].
                    Output of the classification head representing classification scores.
                - pred_boxes: Tensor of shape [batch_size, num_rois, num_classes, 4].
                    Output of the regression head representing encoded predicted boxes.
        """
        pooled_features_flat = self._roi_pooling(feature_maps, rois, True, True)

        pred_scores = self._cls_layer(pooled_features_flat)

        pred_boxes_encoded = self._reg_layer(pooled_features_flat)
        pred_boxes_encoded = self._reg_reshape(pred_boxes_encoded)

        rois = to_absolute(rois, self._image_shape)

        return {"regions": rois, "pred_scores": pred_scores, "pred_boxes": pred_boxes_encoded}

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
            - regions: Tensor of shape [batch_size, num_rois, 4], representing the reference regions.
            - pred_scores: Tensor of shape [batch_size, num_rois, num_classes +1].
                Output of the classification head, representing classification
                scores for each RoI.
            - pred_boxes: Tensor of shape [batch_size, num_rois, num_classes, 4].
                Output of the regression head, representing encoded predicted boxes.
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
                - target_labels: Tensor of shape [batch_size, num_samples, num_classes + 1].
                - pred_scores: Tensor of shape [batch_size, num_samples, num_classes + 1].
                - target_boxes: Tensor of shape [batch_size, num_samples, num_classes, 4].
                - pred_boxes: Tensor of shape [batch_size, num_samples, num_classes, 4].
        """
        target_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: generate_targets(
                x[0], x[1], x[2], self._image_shape, foreground_iou_interval, background_iou_interval
            ),
            elems=(gt_labels, gt_boxes, regions),
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
