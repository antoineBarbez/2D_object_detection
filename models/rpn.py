import tensorflow as tf
import utils.sampling as sampling_utils

from models.abstract_detection_model import AbstractDetectionModel
from models.detectors.rpn_detector import RPNDetector
from models.feature_extractor import preprocess_input, ResNet50FeatureExtractor
from utils.targets import TargetGenerator


class RPN(AbstractDetectionModel):
    def __init__(
        self,
        image_shape,
        window_size=3,
        scales=[0.25, 0.5, 1.0, 2.0],
        aspect_ratios=[0.5, 1.0, 2.0],
        base_anchor_shape=(256, 256),
        name="region_proposal_network",
    ):
        """
        Instantiate a Region Proposal Network.

        Args:
            - image_shape: Shape of the input images.
            - window_size: Size of the sliding window.
            - scales: Anchors' scales.
            - aspect_ratios: Anchors' aspect ratios.
            - base_anchor_shape: Shape of the base anchor. 
        """
        super(RPN, self).__init__(image_shape=image_shape, name=name)

        self._target_generator = TargetGenerator(
            image_shape=image_shape,
            num_classes=1,
            foreground_iou_interval=(0.7, 1.0),
            background_iou_interval=(0.0, 0.3),
        )

        """self.feature_extractor = ResNet50FeatureExtractor(
            kernel_regularizer=tf.keras.regularizers.l2(0.00005), input_shape=image_shape
        )"""

        resnet_50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=image_shape)

        for i, layer in enumerate(resnet_50.layers):
            if layer.name == "conv4_block6_out":
                index = i

        self.feature_extractor = tf.keras.Model(inputs=resnet_50.input, outputs=resnet_50.layers[index].output)

        for layer in self.feature_extractor.layers:
            print(layer.name)

        _, grid_height, grid_width, _ = self.feature_extractor.output_shape
        grid_shape = (grid_height, grid_width)

        self.detector = RPNDetector(
            image_shape=image_shape,
            grid_shape=grid_shape,
            window_size=window_size,
            scales=scales,
            aspect_ratios=aspect_ratios,
            base_anchor_shape=base_anchor_shape,
        )

    def call(self, images, training):
        preprocessed_images = preprocess_input(images)

        feature_maps = self.feature_extractor(preprocessed_images, training=training)

        return self.detector(feature_maps, training=training)

    def postprocess_output(self, anchors, pred_class_scores, pred_boxes_encoded, training=False):
        """
        Postprocess the output of the RPN

        Args:
            - anchors: A tensor of shape [num_anchors, 4] representing the anchor boxes.
            - pred_class_scores: Output of the classification head. A tensor of shape 
                [batch_size, num_anchors, 2] representing classification scores.
            - pred_boxes_encoded: Output of the regression head. A tensor of shape
                [batch_size, num_anchors, 1, 4] representing encoded predicted box coordinates.
            - training: A boolean value indicating whether we are in training mode.

        Returns:
            - rois: A tensor of shape [batch_size, max_predictions, 4] possibly zero padded 
                representing region proposals.
            - roi_scores: A tensor of shape [batch_size, max_predictions] possibly zero padded 
                representing objectness scores for each proposal.
        """
        return self.detector.postprocess_output(anchors, pred_class_scores, pred_boxes_encoded, training)

    @tf.function
    def train_step(self, images, gt_class_labels, gt_boxes, optimizer, num_samples_per_image):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_class_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - optimizer: A tf.keras.optimizers.Optimizer object.
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            - cls_loss: A scalar, the training classification loss.
            - reg_loss: A scalar, the training regression loss.
            - pred_boxes: predicted box proposals.
            - pred_scores: predicted objectness scores.
        """
        with tf.GradientTape() as tape:
            anchors, pred_objectness_scores, pred_boxes_encoded = self.call(images, True)

            gt_objectness_labels = tf.one_hot(
                indices=tf.cast(tf.reduce_sum(gt_class_labels, -1), dtype=tf.int32), depth=2
            )

            target_objectness_labels, target_boxes_encoded = tf.map_fn(
                fn=lambda x: self._target_generator.generate_targets(x[0], x[1], anchors),
                elems=(gt_objectness_labels, gt_boxes),
                dtype=(tf.float32, tf.float32),
            )

            (
                target_objectness_labels_sample,
                target_boxes_encoded_sample,
                pred_objectness_scores_sample,
                pred_boxes_encoded_sample,
                _,
            ) = tf.map_fn(
                fn=lambda x: sampling_utils.sample_image(x[0], x[1], x[2], x[3], anchors, num_samples_per_image, 0.5),
                elems=(target_objectness_labels, target_boxes_encoded, pred_objectness_scores, pred_boxes_encoded),
                dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
            )

            cls_loss = self._classification_loss(target_objectness_labels_sample, pred_objectness_scores_sample)
            reg_loss = self._regression_loss(
                target_objectness_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
            )
            multi_task_loss = cls_loss + reg_loss + sum(self.losses)

        gradients = tape.gradient(multi_task_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        pred_boxes, pred_scores = self.postprocess_output(anchors, pred_objectness_scores, pred_boxes_encoded, False)

        return cls_loss, reg_loss, pred_boxes, pred_scores

    @tf.function
    def test_step(self, images, gt_class_labels, gt_boxes, num_samples_per_image):
        """
        Args:
            - images: Input images. A tensor of shape [batch_size, height, width, 3].
            - gt_class_labels: Ground-truth class labels padded with background one hot encoded.
                A tensor of shape [batch_size, num_objects, num_classes + 1].
            - gt_boxes: Ground-truth bounding boxes padded. A tensor of shape 
                [batch_size, num_objects, 4].
            - num_samples_per_image: Number of examples (regions) to sample per image.

        Returns:
            - cls_loss: A scalar, the training classification loss.
            - reg_loss: A scalar, the training regression loss.
            - pred_boxes: predicted box proposals.
            - pred_scores: predicted objectness scores.
            - foreground_anchors: Anchors labeled as foreground.
            - background_anchors: Anchors labeled as background.
        """
        anchors, pred_objectness_scores, pred_boxes_encoded = self.call(images, False)

        gt_objectness_labels = tf.one_hot(indices=tf.cast(tf.reduce_sum(gt_class_labels, -1), dtype=tf.int32), depth=2)

        target_objectness_labels, target_boxes_encoded = tf.map_fn(
            fn=lambda x: self._target_generator.generate_targets(x[0], x[1], anchors),
            elems=(gt_objectness_labels, gt_boxes),
            dtype=(tf.float32, tf.float32),
        )

        (
            target_objectness_labels_sample,
            target_boxes_encoded_sample,
            pred_objectness_scores_sample,
            pred_boxes_encoded_sample,
            anchors_sample,
        ) = tf.map_fn(
            fn=lambda x: sampling_utils.sample_image(x[0], x[1], x[2], x[3], anchors, num_samples_per_image, 0.5),
            elems=(target_objectness_labels, target_boxes_encoded, pred_objectness_scores, pred_boxes_encoded),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        cls_loss = self._classification_loss(target_objectness_labels_sample, pred_objectness_scores_sample)
        reg_loss = self._regression_loss(
            target_objectness_labels_sample, target_boxes_encoded_sample, pred_boxes_encoded_sample
        )

        pred_boxes, pred_scores = self.postprocess_output(anchors, pred_objectness_scores, pred_boxes_encoded, False)

        foreground_inds = tf.where(target_objectness_labels_sample[..., 0] == 0.0)
        foreground_anchors = tf.gather_nd(anchors_sample, foreground_inds)

        background_inds = tf.where(target_objectness_labels_sample[..., 0] == 1.0)
        background_anchors = tf.gather_nd(anchors_sample, background_inds)

        return cls_loss, reg_loss, pred_boxes, pred_scores, foreground_anchors, background_anchors
